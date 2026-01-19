import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset
from sentence_transformers import InputExample
from tqdm import tqdm

META_PATH = Path("train_samples_with_title.pkl")

def _build_context_dict(context: Iterable) -> Dict[str, Sequence[str]]:
    """
    HotpotQA context는 일반적으로 [[title, [sentences]]] 형태이지만,
    dict 기반 포맷도 존재하므로 모두 지원한다.
    """
    context_dict: Dict[str, Sequence[str]] = {}
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("paragraphs") or []
        for title, sent_list in zip(titles, sentences):
            context_dict[title] = sent_list
    else:
        for item in context or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sentences = item
                context_dict[title] = sentences
    return context_dict

def _build_gold_set(supporting_facts) -> Tuple[Tuple[str, int], ...]:
    """
    supporting_facts가 dict({'title': [...], 'sent_id': [...]})
    혹은 [[title, sent_id], ...] 둘 다 올 수 있으므로 통합 처리.
    """
    gold_pairs = []
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", [])
        gold_pairs = list(zip(titles, sent_ids))
    else:
        gold_pairs = supporting_facts or []
    return tuple((title, int(sent_idx)) for title, sent_idx in gold_pairs)

def create_training_data(dataset_split):
    """
    HotpotQA 데이터를 CrossEncoder 학습용 InputExample 리스트로 변환합니다.
    """
    train_samples: List[InputExample] = []
    
    for data in tqdm(dataset_split, desc="Generating Training Data"):
        query = data['question']
        
        context_dict = _build_context_dict(data.get('context'))
        gold_set = set(_build_gold_set(data.get('supporting_facts')))
        supporting_titles = {title for title, _ in gold_set}

        for title, sent_id in gold_set:
            sentences = context_dict.get(title)
            if not sentences or sent_id >= len(sentences):
                continue
            positive_text = sentences[sent_id]
            train_samples.append(InputExample(texts=[query, title+":"+positive_text], label=1.0))

        # Hard Negatives (같은 문서 내의 다른 문장) 찾기
        for title in supporting_titles:
            sentences = context_dict.get(title)
            if not sentences:
                continue
            for idx, sent in enumerate(sentences):
                if (title, idx) not in gold_set:
                    train_samples.append(InputExample(texts=[query, title+":"+sent], label=0.0))
    
    return train_samples

def main():
    full_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    # 학습 시간을 고려하여 일부만 샘플링하여 테스트해보는 것을 추천합니다 (예: 10%)
    # dataset_slice = full_dataset.select(range(1000))
    train_samples = create_training_data(full_dataset)

    print(f"총 학습 데이터 쌍 개수: {len(train_samples)}")
    if train_samples:
        print(f"예시 1 (Positive): {train_samples[0].texts}, Label: {train_samples[0].label}")

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "wb") as f:
        pickle.dump(train_samples, f)

if __name__ == "__main__":
    main()