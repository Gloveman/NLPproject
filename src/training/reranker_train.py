"""
HotpotQA Cross-Encoder Reranker Training Script
Binary Cross Entropy Loss 버전 (Hard Negatives Mining 제외)
- 단순 BCE Loss 사용
- 최적화: 24시간 내 학습 완료
- FP32 전용 (Titan XP 최적화)
"""

import pickle
import logging
import math
from pathlib import Path
from typing import List, Tuple
from datasets import Dataset
import torch

from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments
)
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator
)
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss

# ============================================================================
# 설정 및 로깅
# ============================================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 경로 설정
TRAIN_DATA_PATH = Path("train_samples_with_title.pkl")
OUTPUT_DIR = Path("hotpotqa_reranker_title_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 학습 하이퍼파라미터
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
VAL_SPLIT_RATIO = 0.1
MAX_LENGTH = 512

# 데이터 샘플링 설정
USE_DATA_SAMPLING = False
SAMPLING_RATIO = 0.5

# Mixed Precision (FP32 고정)
USE_FP16 = False
USE_BF16 = False

# 체크포인트 설정
CHECKPOINT_SAVE_STEPS = 6000
CHECKPOINT_SAVE_LIMIT = 2


# ============================================================================
# 데이터 로딩 및 전처리
# ============================================================================

def load_training_data(pickle_path: Path) -> List[InputExample]:
    """Pickle 파일에서 학습 데이터 로드"""
    if not pickle_path.exists():
        raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {pickle_path}")
    
    logger.info(f"📂 학습 데이터 로딩: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        train_samples = pickle.load(f)
    
    logger.info(f"✓ 총 {len(train_samples):,}개의 샘플 로드 완료")
    
    return train_samples


def sample_balanced_data(
    samples: List[InputExample],
    sampling_ratio: float = 0.5,
    seed: int = 42
) -> List[InputExample]:
    """Positive/Negative 비율 유지하면서 샘플링"""
    import random
    random.seed(seed)
    
    positive_samples = [s for s in samples if s.label == 1.0]
    negative_samples = [s for s in samples if s.label == 0.0]
    
    pos_sample_size = int(len(positive_samples) * sampling_ratio)
    neg_sample_size = int(len(negative_samples) * sampling_ratio)
    
    sampled_positive = random.sample(positive_samples, pos_sample_size)
    sampled_negative = random.sample(negative_samples, neg_sample_size)
    
    sampled_data = sampled_positive + sampled_negative
    random.shuffle(sampled_data)
    
    logger.info("✂️  데이터 샘플링 완료:")
    logger.info(f"   - 원본: {len(samples):,}")
    logger.info(f"   - 샘플링 후: {len(sampled_data):,} ({sampling_ratio:.0%})")
    logger.info(f"   - Positive: {len(sampled_positive):,}")
    logger.info(f"   - Negative: {len(sampled_negative):,}")
    
    return sampled_data


def analyze_data_distribution(samples: List[InputExample]) -> dict:
    """데이터 분포 분석"""
    positive_count = sum(1 for s in samples if s.label == 1.0)
    negative_count = len(samples) - positive_count
    
    stats = {
        "total": len(samples),
        "positive": positive_count,
        "negative": negative_count,
        "pos_ratio": positive_count / len(samples) if samples else 0
    }
    
    logger.info("📊 데이터 분포:")
    logger.info(f"   - 총 샘플: {stats['total']:,}")
    logger.info(f"   - Positive: {stats['positive']:,}")
    logger.info(f"   - Negative: {stats['negative']:,}")
    logger.info(f"   - Positive 비율: {stats['pos_ratio']:.1%}")
    
    return stats


def create_train_val_split(
    samples: List[InputExample],
    val_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[InputExample], List[InputExample]]:
    """학습/검증 데이터 분할"""
    import random
    random.seed(seed)
    
    positive_samples = [s for s in samples if s.label == 1.0]
    negative_samples = [s for s in samples if s.label == 0.0]
    
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)
    
    val_pos_size = int(len(positive_samples) * val_ratio)
    val_neg_size = int(len(negative_samples) * val_ratio)
    
    val_samples = positive_samples[:val_pos_size] + negative_samples[:val_neg_size]
    train_samples = positive_samples[val_pos_size:] + negative_samples[val_neg_size:]
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    logger.info("🔀 Train/Val 분할 완료:")
    logger.info(f"   - 학습 데이터: {len(train_samples):,}개")
    logger.info(f"   - 검증 데이터: {len(val_samples):,}개")
    
    return train_samples, val_samples


# ============================================================================
# 모델 및 학습 설정
# ============================================================================

def check_gpu_info():
    """GPU 정보 확인"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info("🖥️  GPU 정보:")
        logger.info(f"   - 장치: {device_name}")
        logger.info(f"   - 총 메모리: {total_memory:.1f} GB")
        logger.info(f"   - CUDA 버전: {torch.version.cuda}")
        logger.info(f"   - Mixed Precision: FP32")
    else:
        logger.warning("⚠️  GPU를 사용할 수 없습니다.")


def create_training_arguments(output_dir: Path) -> CrossEncoderTrainingArguments:
    """학습 설정 생성"""
    
    logger.info("🎯 학습 설정:")
    logger.info(f"   - Loss: Binary Cross Entropy")
    logger.info(f"   - Precision: FP32")
    logger.info(f"   - Batch Size: {BATCH_SIZE}")
    logger.info(f"   - Gradient Accumulation: {GRADIENT_ACCUMULATION}")
    logger.info(f"   - Effective Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    logger.info(f"   - Epochs: {NUM_EPOCHS}")
    logger.info(f"   - Data Sampling: {'Yes' if USE_DATA_SAMPLING else 'No'} ({SAMPLING_RATIO:.0%})")
    
    args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        
        # ===== Mixed Precision =====
        fp16=False,
        bf16=False,
        
        # ===== 배치 설정 =====
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        
        # ===== 학습 파라미터 =====
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # ===== 평가 및 저장 =====
        eval_strategy="steps",
        eval_steps=CHECKPOINT_SAVE_STEPS,
        save_strategy="steps",
        save_steps=CHECKPOINT_SAVE_STEPS,
        save_total_limit=CHECKPOINT_SAVE_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # ===== 로깅 =====
        logging_dir=str(output_dir / "logs"),
        logging_steps=200,
        logging_first_step=True,
        report_to=[],
        
        # ===== 데이터 로더 =====
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        dataloader_prefetch_factor=2,
        
        # ===== 재현성 =====
        seed=42,
        data_seed=42,
    )
    
    return args


# ============================================================================
# 학습 실행
# ============================================================================

def train_cross_encoder(
    train_samples: List[InputExample],
    val_samples: List[InputExample]
) -> CrossEncoder:
    """Cross-Encoder 학습 (BCE Loss, Hard Negatives 제외)"""
    
    logger.info("="*80)
    logger.info("🚀 Cross-Encoder 학습 시작 (BCE Loss)")
    logger.info("="*80)
    
    # 1. 모델 로드
    logger.info(f"📦 모델 로드: {MODEL_NAME}")
    model = CrossEncoder(
        MODEL_NAME,
        num_labels=1,  # BCE는 단일 출력
        max_length=MAX_LENGTH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    #dataset으로 변환
    logger.info("🔄 데이터 형식 변환 중...")

    train_data = {
        "sentence1": [sample.texts[0] for sample in train_samples],  # query
        "sentence2": [sample.texts[1] for sample in train_samples],  # document
        "label": [float(sample.label) for sample in train_samples]   # 1.0 or 0.0
    }
    train_dataset = Dataset.from_dict(train_data)
    
    val_data = {
        "sentence1": [sample.texts[0] for sample in val_samples],
        "sentence2": [sample.texts[1] for sample in val_samples],
        "label": [float(sample.label) for sample in val_samples]
    }
    val_dataset = Dataset.from_dict(val_data)
    
    logger.info(f"✓ Train dataset: {len(train_dataset):,}개")
    logger.info(f"✓ Val dataset: {len(val_dataset):,}개")

    # 2. BCE Loss 정의
    # pos_weight는 데이터셋의 positive:negative 비율로 설정
    positive_count = sum(1 for s in train_samples if s.label == 1.0)
    negative_count = len(train_samples) - positive_count
    pos_weight_value = negative_count / positive_count if positive_count > 0 else 1.0
    
    loss = BinaryCrossEntropyLoss(
        model=model,
        pos_weight=torch.tensor(pos_weight_value)
    )
    logger.info(f"📐 BCE Loss 설정")
    logger.info(f"   - Positive: {positive_count:,}")
    logger.info(f"   - Negative: {negative_count:,}")
    logger.info(f"   - pos_weight: {pos_weight_value:.2f}")
    
    # 3. Evaluator 생성
    logger.info("📏 Evaluator 설정")
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        val_samples,
        name='HotpotQA-val',
        write_csv=True
    )
    
    # 초기 성능 평가
    logger.info("📊 초기 모델 성능 평가 중...")
    try:
        initial_score = evaluator(model, output_path=str(OUTPUT_DIR / "initial_eval"))
        logger.info(f"✓ 초기 평가 점수: {initial_score:.4f}")
    except Exception as e:
        logger.warning(f"초기 평가 실패 (무시하고 진행): {e}")
        initial_score = None
    
    # 4. 학습 설정
    args = create_training_arguments(OUTPUT_DIR)
    
    # 5. Trainer 초기화
    logger.info("🎓 Trainer 초기화")
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,  # BCE Loss 전달
        evaluator=evaluator,
    )
    
    # 6. 메모리 확인
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info(f"💾 GPU 메모리: {allocated:.2f} GB")
    
    # 7. 학습 시간 예측
    total_samples = len(train_dataset)
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(total_samples / effective_batch)
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    samples_per_second = 13.5
    est_time_hours = (total_samples * NUM_EPOCHS / samples_per_second) / 3600
    
    logger.info("⏱️  학습 예상 정보:")
    logger.info(f"   - 총 스텝: {total_steps:,}")
    logger.info(f"   - 총 샘플: {total_samples:,}")
    logger.info(f"   - 실질 배치: {effective_batch}")
    logger.info(f"   - 예상 시간: {est_time_hours:.1f}시간 ({est_time_hours/24:.1f}일)")
    
    # 8. 학습 시작
    logger.info("\n" + "="*80)
    logger.info("🏋️  학습 시작")
    logger.info("="*80 + "\n")
    
    trainer.train()
    
    # 9. 최종 평가
    logger.info("\n" + "="*80)
    logger.info("📊 최종 모델 평가")
    logger.info("="*80)
    
    try:
        final_score = evaluator(model, output_path=str(OUTPUT_DIR / "final_eval"))
        logger.info(f"✓ 최종 평가 점수: {final_score:.4f}")
        
        if initial_score is not None:
            improvement = final_score - initial_score
            logger.info(f"✓ 성능 향상: {improvement:+.4f} ({improvement/abs(initial_score)*100:+.1f}%)")
    except Exception as e:
        logger.warning(f"최종 평가 실패: {e}")
    
    # 10. 최종 모델 저장
    final_model_path = OUTPUT_DIR / "final"
    final_model_path.mkdir(exist_ok=True, parents=True)
    model.save(str(final_model_path))
    logger.info(f"💾 최종 모델 저장: {final_model_path}")
    
    return model


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    
    logger.info("\n" + "="*80)
    logger.info("🏁 HotpotQA Cross-Encoder with BCE Loss (단순 버전)")
    logger.info("="*80 + "\n")
    
    try:
        # 1. GPU 확인
        check_gpu_info()
        
        # 2. 데이터 로드
        train_samples = load_training_data(TRAIN_DATA_PATH)
        
        # 3. 데이터 샘플링
        if USE_DATA_SAMPLING:
            train_samples = sample_balanced_data(
                train_samples,
                sampling_ratio=SAMPLING_RATIO
            )
        
        # 4. 데이터 분석
        stats = analyze_data_distribution(train_samples)
        
        if stats['pos_ratio'] < 0.1 or stats['pos_ratio'] > 0.9:
            logger.warning(f"⚠️  데이터 불균형 감지: Positive 비율 {stats['pos_ratio']:.1%}")
        
        # 5. Train/Val 분할
        train_data, val_data = create_train_val_split(
            train_samples,
            val_ratio=VAL_SPLIT_RATIO
        )
        
        # 6. 학습 실행
        model = train_cross_encoder(
            train_samples=train_data,
            val_samples=val_data
        )
        
        # 7. 완료 메시지
        logger.info("\n" + "="*80)
        logger.info("🎉 학습 완료!")
        logger.info("="*80)
        logger.info(f"📁 모델: {OUTPUT_DIR / 'final'}")
        logger.info(f"📊 TensorBoard: {OUTPUT_DIR / 'logs'}")
        logger.info("\n💡 사용 방법:")
        logger.info("  from sentence_transformers import CrossEncoder")
        logger.info(f"  model = CrossEncoder('{OUTPUT_DIR / 'final'}')")
        logger.info("  scores = model.predict([['query', 'document']])")
        
        logger.info("\n📋 설정 요약:")
        logger.info(f"  - Loss Function: Binary Cross Entropy")
        logger.info(f"  - Hard Negatives: No (제외)")
        logger.info(f"  - Epochs: {NUM_EPOCHS}")
        logger.info(f"  - Data Sampling: {SAMPLING_RATIO:.0%}")
        logger.info(f"  - 예상 시간: 15-18시간 (Titan XP 기준)")
        
    except FileNotFoundError as e:
        logger.error(f"❌ 파일 없음: {e}")
        logger.info("\n💡 해결 방법:")
        logger.info("  1. 데이터 생성 스크립트를 먼저 실행하세요")
        logger.info("  2. train_samples.pkl 파일 경로를 확인하세요")
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
