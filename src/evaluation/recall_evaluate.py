from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import faiss
import numpy as np
import os
import random as rd
import pickle
from pathlib import Path

# ==========================================
# 정규화 함수
# ==========================================

def normalize_title(title: str) -> str:
    return " ".join(title.strip().split())

def normalize_text(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    return text

def extract_supporting_facts(data):
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

def get_sents_than_rerank(
    query,
    sents_metadata,
    gpu_index,
    embedder,
    reranker
):
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    K = 50
    D, I = gpu_index.search(query_emb, K)
    candidate_docs = [sents_metadata[idx]["text"] for idx in I[0]]
    
    pairs = [[query, doc] for doc in candidate_docs]
    scores = reranker.predict(pairs)

    sorted_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    selected_docs = [doc for doc, _ in sorted_results[:5]]
    return selected_docs

def get_docs_than_rerank(
    query,
    doc_metadata,
    sent_metadata,
    gpu_index,
    embedder,
    doc_reranker,
    sents_reranker
):
    """추천 설정으로 문서 검색 및 선택"""
    # 1. 문서 검색 (num_docs=3)
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    K = 10
    D, I = gpu_index.search(query_emb, K)
    
    top_titles = [doc_metadata[idx]["title"] for idx in I[0]]
    top_paras = [doc_metadata[idx]["paragraph"] for idx in I[0]]
    
    # 2. 문서 Reranking
    para_pairs = [[query, para] for para in top_paras]
    para_scores = doc_reranker.predict(para_pairs)
    
    ranked_indices = sorted(range(len(para_scores)), key=lambda i: para_scores[i], reverse=True)[:3]
    ranked_titles = [top_titles[i] for i in ranked_indices]
    
    # 3. 문장 추출
    candidate_docs = []
    for title in ranked_titles:
        matching_sents = [(title,m["text"]) for m in sent_metadata if m["title"] == title] #test
        candidate_docs.extend(matching_sents)
    
    # 4. 문장 Reranking
    pairs = [[query, doc[0]+":"+doc[1]] for doc in candidate_docs]
    
    scores = sents_reranker.predict(pairs)
    
    # 5. Relative threshold 선택 (ratio=0.6)
    sorted_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    top_score = sorted_results[0][1]
    threshold = top_score * 0.5
    
    selected = [
        (doc, score) 
        for doc, score in sorted_results 
        if score >= threshold
    ]
    
    # Min/Max 보장
    if len(selected) < 3:
        selected = sorted_results[:3]
    elif len(selected) > 8:
        selected = selected[:8]
    
    return [sentence for (_,sentence), _ in selected]

def evaluate_with_params(
    selected_data,
    doc_metadata,
    sent_metadata,
    gpu_index,
    embedder,
    base_reranker,
    trained_reranker,
    is_base,
):
    """Precision, Recall, F1 계산"""
    recalls = []
    precisions = []
    f1_scores = []
    num_selected_list = []
    
    for data in selected_data:
        query = data["question"]
        supporting_facts = extract_supporting_facts(data)
        
        if not supporting_facts:
            continue
        
        # 문서 검색
        selected_docs = get_sents_than_rerank(query, sent_metadata, gpu_index, embedder,base_reranker) if is_base \
            else get_docs_than_rerank(query, doc_metadata, sent_metadata, gpu_index, embedder, base_reranker, trained_reranker)
        
        # 평가
        true_positives = sum([1 for doc in selected_docs if doc in supporting_facts])
        recall = true_positives / len(supporting_facts)
        precision = true_positives / len(selected_docs) if len(selected_docs) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        num_selected_list.append(len(selected_docs))

    return {
        'recall_mean': np.mean(recalls) if recalls else 0,
        'precision_mean': np.mean(precisions) if precisions else 0,
        'f1_mean': np.mean(f1_scores) if f1_scores else 0,
        'avg_selected': np.mean(num_selected_list) if num_selected_list else 0,
        'num_samples': len(recalls)
    }

# ==========================================
# 메인 실행
# ==========================================

def main():
    N_SAMPLE = 1000
    OUTPUT_DIR = Path(os.path.expanduser("~/faiss"))
    
    print("📂 데이터 로딩 중...")
    
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    bridge_data = [d for d in dataset if d["type"] == "bridge"]
    
    with open(OUTPUT_DIR / "documents_metadata.pkl", "rb") as f:
        doc_metadata = pickle.load(f)
    
    with open(OUTPUT_DIR / "sentences_metadata.pkl", "rb") as f:
        sent_metadata = pickle.load(f)
    
    doc_index = faiss.read_index(str(OUTPUT_DIR / "documents.index"))
    res = faiss.StandardGpuResources()
    gpu_doc_index = faiss.index_cpu_to_gpu(res, 0, doc_index)
    
    sent_index = faiss.read_index(str(OUTPUT_DIR / "sentences.index"))
    gpu_sent_index = faiss.index_cpu_to_gpu(res, 0, sent_index)

    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedder.cuda()
    base_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', device='cuda')
    #trained_reranker = CrossEncoder('hotpotqa_reranker_full/final', device='cuda')
    trained_reranker = CrossEncoder('hotpotqa_reranker_title_full/final', device='cuda')
    print("✓ 완료\n")
    
    rd.seed(42)
    selected_data = rd.sample(bridge_data, N_SAMPLE)
    
    base_results = evaluate_with_params(selected_data,doc_metadata, sent_metadata, gpu_sent_index,embedder,base_reranker,trained_reranker,True)
    trained_results = evaluate_with_params(selected_data,doc_metadata, sent_metadata, gpu_doc_index,embedder,base_reranker,trained_reranker,False)

    print("\n" + "="*80)
    print("📊 최종 비교")
    print("="*80)
    print(f"\n{'지표':<15} {'Baseline':<15} {'Trained':<15} {'개선':<20}")
    print("-"*80)

    # Recall
    r_imp = trained_results['recall_mean'] - base_results['recall_mean']
    r_pct = (r_imp / base_results['recall_mean']) * 100 if base_results['recall_mean'] > 0 else 0
    print(f"{'Recall':<15} {base_results['recall_mean']:<15.5f} {trained_results['recall_mean']:<15.5f} {r_imp:+.5f} ({r_pct:+.1f}%)")

    # Precision
    p_imp = trained_results['precision_mean'] - base_results['precision_mean']
    p_pct = (p_imp / base_results['precision_mean']) * 100 if base_results['precision_mean'] > 0 else 0
    print(f"{'Precision':<15} {base_results['precision_mean']:<15.5f} {trained_results['precision_mean']:<15.5f} {p_imp:+.5f} ({p_pct:+.1f}%)")

    # F1
    f1_imp = trained_results['f1_mean'] - base_results['f1_mean']
    f1_pct = (f1_imp / base_results['f1_mean']) * 100 if base_results['f1_mean'] > 0 else 0
    print(f"{'F1 Score':<15} {base_results['f1_mean']:<15.5f} {trained_results['f1_mean']:<15.5f} {f1_imp:+.5f} ({f1_pct:+.1f}%)")

    print("-"*80)
    print(f"{'평균 선택':<15} {base_results['avg_selected']:<15.2f} {trained_results['avg_selected']:<15.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
