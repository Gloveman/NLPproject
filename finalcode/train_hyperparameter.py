from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import faiss
import numpy as np
import os
import random as rd
import pickle
from tqdm import tqdm
from pathlib import Path
from itertools import product
import pandas as pd
from collections import defaultdict

# ==========================================
# 1. 유틸리티 함수
# ==========================================

def normalize_title(title: str) -> str:
    return " ".join(title.strip().split())

def normalize_text(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    return text

def extract_supporting_facts(data):
    """Ground Truth 추출"""
    supporting_facts = []
    sf_titles = data['supporting_facts']['title']
    sf_sent_ids = data['supporting_facts']['sent_id']
    context_titles = data['context']['title']
    context_sentences = data['context']['sentences']
    
    context_dict = {
        normalize_title(title): [normalize_text(s) for s in sentences]
        for title, sentences in zip(context_titles, context_sentences)
    }
    
    for sf_title, sf_sent_id in zip(sf_titles, sf_sent_ids):
        normalized_sf_title = normalize_title(sf_title)
        if normalized_sf_title in context_dict:
            sentences = context_dict[normalized_sf_title]
            if sf_sent_id < len(sentences):
                supporting_facts.append(sentences[sf_sent_id])
    
    return supporting_facts

# [추가] Sent Metadata를 Title별로 묶어주는 함수 (속도 최적화)
def create_title_to_sentences_map(sent_metadata):
    print("🔄 Building Title-to-Sentences Map...")
    title_to_sents = defaultdict(list)
    
    # sent_metadata 구조가 [{'title': '...', 'text': '...'}, ...] 라고 가정
    for item in tqdm(sent_metadata, desc="Mapping"):
        title = item.get('title', '').strip()
        text = item.get('text', '').strip()
        if title and text:
            title_to_sents[title].append(text)
            
    return title_to_sents

# ==========================================
# 2. Threshold 선택 로직 (이전과 동일)
# ==========================================
# ... (select_by_relative_threshold 등은 생략 - 이전 코드와 동일하게 사용) ...

def select_by_relative_threshold(docs_with_scores, ratio=0.7, min_docs=2, max_docs=5):
    if not docs_with_scores: return []
    sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    top_score = sorted_docs[0][1]
    threshold = top_score * ratio if top_score > 0 else top_score - abs(top_score) * (1 - ratio)
    selected = [(doc, score) for doc, score in sorted_docs if score >= threshold]
    return apply_min_max(selected, sorted_docs, min_docs, max_docs)

def select_by_adaptive_threshold(docs_with_scores, std_multiplier=0.5, min_docs=2, max_docs=5):
    if not docs_with_scores: return []
    sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    scores = [score for _, score in sorted_docs]
    threshold = np.mean(scores) + std_multiplier * np.std(scores)
    selected = [(doc, score) for doc, score in sorted_docs if score >= threshold]
    return apply_min_max(selected, sorted_docs, min_docs, max_docs)

def select_by_score_gap(docs_with_scores, gap_threshold=1.0, min_docs=2, max_docs=5):
    if not docs_with_scores: return []
    sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    selected = [sorted_docs[0]]
    for i in range(1, len(sorted_docs)):
        score_diff = sorted_docs[i-1][1] - sorted_docs[i][1]
        if score_diff > gap_threshold and len(selected) >= min_docs: break
        selected.append(sorted_docs[i])
        if len(selected) >= max_docs: break
    return selected

def select_by_absolute_threshold(docs_with_scores, threshold=0.0, min_docs=2, max_docs=5):
    if not docs_with_scores: return []
    sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    selected = [(doc, score) for doc, score in sorted_docs if score >= threshold]
    return apply_min_max(selected, sorted_docs, min_docs, max_docs)

def apply_min_max(selected, sorted_docs, min_docs, max_docs):
    if len(selected) < min_docs: return sorted_docs[:min_docs]
    elif len(selected) > max_docs: return selected[:max_docs]
    return selected

# ==========================================
# 3. 전처리 및 점수 계산 (Metadata Linking 적용)
# ==========================================

def precompute_reranking_scores(
    selected_data, doc_metadata, title_to_sents_map, gpu_index, embedder, reranker, doc_reranker, top_k_docs=10
):
    precomputed_results = []
    
    print(f"🚀 Pre-computing scores (Doc Search -> Sent Lookup -> Rerank)...")
    
    for data in tqdm(selected_data, desc="Pre-computing"):
        query = data["question"]
        supporting_facts = extract_supporting_facts(data)
        
        # 1. FAISS 문서 검색 (Document Index)
        query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = gpu_index.search(query_emb, top_k_docs)
        
        top_titles = [doc_metadata[idx]["title"] for idx in I[0]]
        top_paras = [doc_metadata[idx]["paragraph"] for idx in I[0]]
    
        # 2. 문서 Reranking
        para_pairs = [[query, para] for para in top_paras]
        para_scores = doc_reranker.predict(para_pairs)
    
        ranked_indices = sorted(range(len(para_scores)), key=lambda i: para_scores[i], reverse=True)[:3]
        ranked_titles = [top_titles[i] for i in ranked_indices]

        candidates = []
        texts_for_rerank = []
        original_texts = []
        
        # 2. 문서 내 문장 펼치기 (Using Metadata Map)
        for title in ranked_titles:
            # [수정된 부분] Title을 키로 사용하여 Map에서 문장 리스트 가져오기
            sentences = title_to_sents_map.get(title, [])
            
            for sent in sentences:
                # Title: Sentence 포맷 적용
                formatted_input = f"{title} : {sent}"
                texts_for_rerank.append([query, formatted_input])
                original_texts.append(sent)
        
        if not texts_for_rerank:
            precomputed_results.append({
                "query": query,
                "ground_truth": supporting_facts,
                "docs_with_scores": []
            })
            continue

        # 3. Reranking 수행
        scores = reranker.predict(texts_for_rerank, batch_size=128) # 배치 사이즈 조절 가능
        
        docs_with_scores = list(zip(original_texts, scores))
        
        precomputed_results.append({
            "query": query,
            "ground_truth": supporting_facts,
            "docs_with_scores": docs_with_scores
        })
        
    return precomputed_results

# ... (evaluate_on_precomputed, run_grid_search는 이전 코드와 동일하므로 생략) ...
# ==========================================
# 4. 평가 로직 (Evaluate & Grid Search)
# ==========================================
def evaluate_on_precomputed(precomputed_data, method="relative", **kwargs):
    recalls, precisions, f1_scores, num_selected_list = [], [], [], []
    for item in precomputed_data:
        supporting_facts = item['ground_truth']
        docs_with_scores = item['docs_with_scores']
        if not supporting_facts: continue

        if method == "relative": selected = select_by_relative_threshold(docs_with_scores, **kwargs)
        elif method == "adaptive": selected = select_by_adaptive_threshold(docs_with_scores, **kwargs)
        elif method == "gap": selected = select_by_score_gap(docs_with_scores, **kwargs)
        elif method == "absolute": selected = select_by_absolute_threshold(docs_with_scores, **kwargs)
        
        selected_docs = [doc for doc, _ in selected]
        num_selected_list.append(len(selected_docs))
        
        true_positives = sum([1 for doc in selected_docs if doc in supporting_facts])
        recall = true_positives / len(supporting_facts)
        precision = true_positives / len(selected_docs) if len(selected_docs) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        
    return {
        'recall_mean': np.mean(recalls) if recalls else 0,
        'precision_mean': np.mean(precisions) if precisions else 0,
        'f1_mean': np.mean(f1_scores) if f1_scores else 0,
        'avg_selected': np.mean(num_selected_list) if num_selected_list else 0
    }

def run_grid_search(precomputed_data, param_grid):
    results = []
    total_iter = sum(len(list(product(*grid.values()))) for grid in param_grid.values())
    with tqdm(total=total_iter, desc="Grid Search") as pbar:
        for method, grid in param_grid.items():
            keys = list(grid.keys())
            values = list(grid.values())
            for combo in product(*values):
                params = dict(zip(keys, combo))
                eval_res = evaluate_on_precomputed(precomputed_data, method=method, **params)
                result_entry = {'method': method, **params, 'recall': eval_res['recall_mean'], 'precision': eval_res['precision_mean'], 'f1': eval_res['f1_mean'], 'avg_selected': eval_res['avg_selected']}
                results.append(result_entry)
                pbar.update(1)
    return pd.DataFrame(results)

# ==========================================
# 5. 메인 실행
# ==========================================

def main():
    N_SAMPLE = 500
    OUTPUT_DIR = Path(os.path.expanduser("~/faiss"))
    
    print("📂 데이터 로딩 중...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    bridge_data = [d for d in dataset if d["type"] == "bridge"]
    
    # 1. 문서(Paragraph) Metadata 로드
    with open(OUTPUT_DIR / "documents_metadata.pkl", "rb") as f:
        doc_metadata = pickle.load(f)
    
    # 2. 문장(Sentence) Metadata 로드
    with open(OUTPUT_DIR / "sentences_metadata.pkl", "rb") as f: # 파일명 확인
        sent_metadata = pickle.load(f)

    # 3. 문서 FAISS Index 로드
    doc_index = faiss.read_index(str(OUTPUT_DIR / "documents.index")) # 문서용 인덱스
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, doc_index)
    
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedder.cuda()

    print("🔥 Loading Trained Reranker...")
    doc_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', device='cuda')
    trained_reranker = CrossEncoder('hotpotqa_reranker_with_title/final', device='cuda')
    
    # [핵심] Title -> Sentences 매핑 생성
    title_to_sents_map = create_title_to_sentences_map(sent_metadata)
    
    rd.seed(42)
    selected_data = rd.sample(bridge_data, N_SAMPLE)
    
    # ---------------------------------------------------------
    # Step 1: 점수 미리 계산
    # ---------------------------------------------------------
    trained_precomputed = precompute_reranking_scores(
        selected_data, 
        doc_metadata, 
        title_to_sents_map, # Map 전달
        gpu_index, 
        embedder, 
        trained_reranker,
        doc_reranker=doc_reranker,
        top_k_docs=10
    )

    # ---------------------------------------------------------
    # Step 2: Grid Search
    # ---------------------------------------------------------
    print("\n[Phase 2] Running Grid Search...")
    
    # Trained 모델용 파라미터 그리드
    param_grid = {
        'relative': {'ratio': [0.5, 0.6, 0.7, 0.8, 0.9], 'min_docs': [2], 'max_docs': [5]},
        'adaptive': {'std_multiplier': [0.0, 0.5, 1.0], 'min_docs': [2], 'max_docs': [5]},
        'gap': {'gap_threshold': [0.5, 1.0, 2.0, 3.0, 4.0], 'min_docs': [2], 'max_docs': [5]},
        'absolute': {'threshold': [-5, -2, 0, 2, 5], 'min_docs': [2], 'max_docs': [5]}
    }

    results_df = run_grid_search(trained_precomputed, param_grid)
    
    # ---------------------------------------------------------
    # Step 3: 결과 출력
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print("🏆 Optimization Results")
    print(f"{'='*80}")
    
    best_row = results_df.loc[results_df['f1'].idxmax()]
    
    print(f"Best Method: {best_row['method']}")
    print(f"Best F1 Score : {best_row['f1']:.5f}")
    print(f"Precision     : {best_row['precision']:.5f}")
    print(f"Recall        : {best_row['recall']:.5f}")
    print(f"Avg Docs      : {best_row['avg_selected']:.2f}")
    
    params_only = best_row.drop(['method', 'recall', 'precision', 'f1', 'avg_selected']).dropna().to_dict()
    print("Parameters:", params_only)
    
    results_df.to_csv("optimized_params_doc_linked.csv", index=False)
    print(f"\n✅ 결과 저장 완료: optimized_params_doc_linked.csv")

if __name__ == "__main__":
    main()