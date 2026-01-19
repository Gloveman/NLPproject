import json
import re
import string
import collections
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. 표준 정규화 및 점수 계산 함수 (변동 없음)
# ==========================================

def normalize_answer(s):
    """답변 정규화"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def compute_exact_match(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    
    if precision + recall == 0:
        return 0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# ==========================================
# 2. 평가 실행 함수 (JSON 저장 제거)
# ==========================================

def evaluate_standard_metrics(json_path: str):
    """
    JSON 파일을 읽어 EM과 F1 Score 통계만 출력합니다.
    """
    json_path = Path(json_path)
    
    print('\n' + '='*80)
    print(f'📏 Standard Metrics Evaluation: {json_path.name}')
    print('='*80)
    
    if not json_path.exists():
        print(f"❌ Error: File not found - {json_path}")
        return

    # 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_records = json.load(f)

    em_scores = []
    f1_scores = []

    # 평가 루프
    for record in tqdm(qa_records, desc="Calculating"):
        # Key 유연 처리
        query = record.get('question', record.get('query', ''))
        pred = record.get('predicted_answer', record.get('generated_answer', record.get('answer', '')))
        truth = record.get('ground_truth', '')
        
        # 점수 계산
        em = compute_exact_match(pred, truth)
        f1 = compute_f1(pred, truth)
        
        em_scores.append(em)
        f1_scores.append(f1)

    # 평균 계산
    avg_em = np.mean(em_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100
    
    # 통계 출력
    print('-' * 80)
    print(f'✅ Total Samples   : {len(qa_records)}')
    print(f'🎯 Exact Match (EM): {avg_em:.2f}%')
    print(f'⚖️  F1 Score        : {avg_f1:.2f}%')
    print('='*80)
        
    return {
        "em": avg_em,
        "f1": avg_f1
    }

# ==========================================
# 3. 메인 실행
# ==========================================

if __name__ == "__main__":
    # Baseline 평가
    if Path('rag_baseline_results.json').exists():
        evaluate_standard_metrics('rag_baseline_results.json')
        
    # Ours 평가
    if Path('title_full_results.json').exists():
        evaluate_standard_metrics('rag_ours_results.json')
