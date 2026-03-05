import os
import json
import pickle
import random as rd
from pathlib import Path
from tqdm import tqdm

import faiss
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# utils에서 모듈화된 함수 임포트
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.retrieval import get_sents_than_rerank
from utils.generation import get_summary, generate_answer_few_shot

def main():
    N_SAMPLE = 5500
    OUTPUT_DIR = Path(os.path.expanduser("~/faiss"))

    ##### LOAD QUERY #####
    print("데이터 로딩 중...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    bridge_data = [d for d in dataset if d["type"] == "bridge"]

    with open(OUTPUT_DIR / "sentences_metadata.pkl", "rb") as f:
        sent_metadata = pickle.load(f)

    sent_index = faiss.read_index(str(OUTPUT_DIR / "sentences.index"))
    res = faiss.StandardGpuResources()
    gpu_sent_index = faiss.index_cpu_to_gpu(res, 0, sent_index)

    print("✓ 완료\n")

    ##### MODEL LOAD #####
    print("모델 로딩 중...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedder.cuda()
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', device='cuda')

    pipe = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.2-1B-Instruct",
        device_map="cuda",
    )

    ##### SELECT SAMPLES #####
    selected_data = rd.sample(bridge_data, N_SAMPLE)
    queries = [d["question"] for d in selected_data]
    ground_truths = [d["answer"] for d in selected_data]

    results = []
    for query, ground_truth in tqdm(zip(queries, ground_truths), total=len(queries), desc="RAG Baseline Pipeline"):
        selected_docs = get_sents_than_rerank(
            query=query,
            sents_metadata=sent_metadata,
            gpu_index=gpu_sent_index,
            embedder=embedder,
            reranker=reranker,
            top_k_search=50,
            top_k_rerank=5
        )
        context = " ".join(selected_docs)
        summary = get_summary(query, context=context, pipe=pipe)
        answer = generate_answer_few_shot(query, summary=summary, pipe=pipe)

        result_item = {
            "question": query,
            "ground_truth": ground_truth,
            "predicted_answer": answer,
            "summary": summary
        }
        results.append(result_item)

    save_path = "results/rag_baseline_results.json"
    os.makedirs("results", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=4)

    print(f"\n💾 모든 결과가 다음 경로에 저장되었습니다: {save_path}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()