from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import torch
import faiss
import numpy as np
import re
import os
import random as rd 
import pickle
from pathlib import Path
import json
from tqdm import tqdm

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
    threshold = top_score * 0.6
    
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

#####GET_SUMMARY#####
def get_summary(query, context, pipe):
    TOKEN_THRESHOLD = 200
    if len(context) < TOKEN_THRESHOLD:   
        return context
    summary_messages = [
    {
        "role": "system",
        "content": (
            "You are an expert research assistant. "
            "Your task is to extract and synthesize information relevant to the user's question from the provided text snippets."
        )
    },
    {
        "role": "user",
        "content": f"""
### Instruction:
Read the following 'Context' and synthesize the information to help answer the 'Question'.
1. **Focus only on relevant facts**: Ignore information that does not help answer the question.
2. **Be concise**: Do not write a long essay. Just provide the key facts in a coherent paragraph.

### Question:
{query}

### Context:
{context}

### Synthesized Summary:
"""
    }
]
    outputs = pipe(
    summary_messages,
    max_new_tokens=256,
    pad_token_id=pipe.tokenizer.eos_token_id,
    temperature = 0.3
    )
    return outputs[0]["generated_text"][-1]['content']

#####GERATE ANSWER#####
def generate_answer(query, summary, pipe):
    few_shot_examples = """
    """
    answer_messages = [
    {
        "role": "system",
        "content": (
            "You are a precise QA agent. "
            "You must answer the question based EXCLUSIVELY on the provided summary. "
        )
    },
    {
        "role": "user",
        "content": f"""
### Instruction:
Your goal is to answer the question using the provided summary.
Follow these steps:
1. **Analyze**: Read the question and the summary carefully.
2. **Reason**: Think step-by-step about how the facts in the summary connect to the question.
3. **Answer**: Provide the final answer concisely. Only one or two words are allowed.



**Output Format:**
Please strictly follow this format:
---
**Final Answer:** [Write only the final answer here]
---

### Summary:
{summary}

### Question:
{query}
"""
    }
]
    outputs = pipe(
    answer_messages,
    max_new_tokens=128,
    pad_token_id=pipe.tokenizer.eos_token_id,
    temperature = 0.1
    )
    full_output =  outputs[0]["generated_text"][-1]['content']
    match_first = re.search(r"\*\*Final Answer:\*\*\s*(.*)", full_output, re.DOTALL | re.IGNORECASE)
    match_second = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL | re.IGNORECASE)
    if match_first:
        return match_first.group(1).strip()
    if match_second:
        return match_second.group(1).strip()
    return full_output

def generate_answer_few_shot(query, summary, pipe):
    # 1. System Prompt: 역할과 규칙 정의
    system_prompt = (
        "You are a precise QA agent. "
        "Your goal is to answer the user's question based EXCLUSIVELY on the provided context summary. "
    )

    # 2. Few-Shot Examples (대화 턴으로 분리)
    # 모델이 '따라하기' 모드가 되도록 예시를 대화 기록처럼 구성합니다.
    example_dialogues = [
        # [Example 1]
        {"role": "user", "content": """
### Instruction:
Answer the question based on the summary.

### Summary:
The British Army's infantry rifle regiment that Talaiasi Labalaba served in was first created in 1881 by the amalgamation of the 83rd \
         (County of Dublin) Regiment of Foot and the 86th (Royal County Down) Regiment of Foot, becoming the Royal Irish Rifles.

### Question:
The infantry rifle regiment of the British Army that Talaiasi Labalaba served in was first created in what year?
"""},
        {"role": "assistant", "content": """
**Final Answer:** 1881
"""},
        
        # [Example 2]
        {"role": "user", "content": """
### Instruction:
Answer the question based on the summary.

### Summary:
The European Cup was won by Juventus in 1985, but the season was marred by the Heysel Stadium disaster, \
         in which 39 people were killed and 600 injured, primarily due to a violent confrontation between Juventus and Liverpool fans.

### Question:
Who was the winner of the European Cup the year that 39 people were killed?
"""},
        {"role": "assistant", "content": """
**Final Answer:** Juventus
"""}
    ]

    # 3. 실제 질문 구성 (Actual Task)
    # 예시들과 포맷을 똑같이 맞춰줍니다.
    actual_task_message = {
        "role": "user", 
        "content": f"""
### Instruction:
Now, answer the following NEW question based on the NEW summary provided below.
Do not use the previous examples' answers.

### Summary:
{summary}

### Question:
{query}
"""
    }

    # 4. 전체 메시지 리스트 조립
    # System -> Example User -> Example Assistant -> ... -> Real User
    answer_messages = [{"role": "system", "content": system_prompt}] + example_dialogues + [actual_task_message]
    
    # 5. 생성
    outputs = pipe(
        answer_messages,
        max_new_tokens=128,
        pad_token_id=pipe.tokenizer.eos_token_id,
        temperature=0.1 # 창의성이 필요 없으므로 낮게 설정
    )
    
    full_output = outputs[0]["generated_text"][-1]['content']
    match_first = re.search(r"\*\*Final Answer:\*\*\s*(.*)", full_output, re.DOTALL | re.IGNORECASE)
    match_second = re.search(r"Final Answer:\s*(.*)", full_output, re.DOTALL | re.IGNORECASE)
    if match_first:
        return match_first.group(1).strip()
    if match_second:
        return match_second.group(1).strip()
    return full_output
    

def main():
    N_SAMPLE = 5500
    OUTPUT_DIR = Path(os.path.expanduser("~/faiss"))
    #####LOAD QUERY#####
    print("###############OUR MODEL###############")
    print("데이터 로딩 중...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor",split = "validation")
    bridge_data = [d for d in dataset if d["type"] == "bridge"]

    with open(OUTPUT_DIR / "documents_metadata.pkl", "rb") as f:
        doc_metadata = pickle.load(f)
    with open(OUTPUT_DIR / "sentences_metadata.pkl", "rb") as f:
        sent_metadata = pickle.load(f)

    doc_index = faiss.read_index(str(OUTPUT_DIR / "documents.index"))
    res = faiss.StandardGpuResources()
    gpu_doc_index = faiss.index_cpu_to_gpu(res, 0, doc_index)
    
    print("✓ 완료\n")

    #####MODEL LOAD#####
    print("모델 로딩 중...") 
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedder.cuda()
    doc_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', device='cuda')
    sents_reranker = CrossEncoder('hotpotqa_reranker_title_full/final', device='cuda')
    
    pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    device_map="cuda",
    )
    #####SELECT SAMPLES#####
    selected_data = rd.sample(bridge_data, N_SAMPLE)
    queries = [d["question"] for d in selected_data]
    ground_truths = [d["answer"] for d in selected_data]

    results = []
    for query, ground_truth in tqdm(zip(queries, ground_truths), total=len(queries), desc="RAG Pipeline"):
        selected_docs = get_docs_than_rerank(query,doc_metadata=doc_metadata, sent_metadata=sent_metadata, gpu_index=gpu_doc_index, 
                                             embedder=embedder, doc_reranker=doc_reranker,sents_reranker=sents_reranker)
        context = " ".join(selected_docs)
        summary = get_summary(query, context=context, pipe=pipe)
        answer = generate_answer_few_shot(query, summary=summary, pipe=pipe)
        """
        print("----------------------------------------------------------------------")
        print("\n💬 Query:\n", query)
        print("📝 Summary:\n", summary)
        print("\n💬 Final Answer:\n", answer)
        print("\n💬 Grounded truth:\n", ground_truth)
        print("----------------------------------------------------------------------")
        """
        result_item = {
            "question": query,
            "ground_truth": ground_truth,
            "predicted_answer": answer,
            "summary": summary            # (선택사항) 요약 모델 성능 분석용
        }
        results.append(result_item)
    save_path = "title_full_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=4)

    print(f"\n💾 모든 결과가 다음 경로에 저장되었습니다: {save_path}")                
    torch.cuda.empty_cache()
if __name__ == "__main__":
    main()

