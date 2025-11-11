from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
import numpy as np
import os
import random as rd
import json

##### PREPARE #####

##### load query, ground_truth #####
queries = []
#ground_truth = []
#support_docs_titles = [] #For reranker evaluation
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor",split = "validation")
bridge_data = [d for d in dataset if d["type"] == "bridge"]
for data in bridge_data:
    queries.append(data["question"])
#    ground_truth.append(data["answer"])
#    support_docs_titles.append(data["supporting_facts"]["title"])

##### load faiss index  & docs #####
index_path = os.path.expanduser("~/faiss/bridge_val_faiss.index")
json_path = os.path.expanduser("~/faiss/bridge_val_metadata.json")
with open(json_path, "r", encoding="utf-8") as f:
    meta_data = json.load(f)
docs = [p['paragraph'] for p in meta_data]
ids = [p['id'] for p in meta_data]

docs_index = faiss.read_index(index_path)
res = faiss.StandardGpuResources()  # GPU 리소스 생성
gpu_index = faiss.index_cpu_to_gpu(res, 0, docs_index)  # 0번 GPU 사용

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

##### model load #####
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', device = 'cuda')

sum_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", dtype=torch.float16, device_map="auto")
sum_model.to("cuda")

"""
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small", 
    dtype=torch.float16,
    device_map="auto"
)
gen_model.to("cuda")
"""
##### 1. Initial docs search (cosine similarity) #####
selected_queries = rd.sample(queries,20)
truths = []
for query in selected_queries:
    truth = [d["answer"] for d in bridge_data if d["question"] == query][0]
    truths.append(truth)
for t in range(len(selected_queries)):
    query = selected_queries[t]
    query_emb = embedder.encode(
        [query], convert_to_numpy=True
    )
    faiss.normalize_L2(query_emb)
    K=20
    D,I = gpu_index.search(query_emb, K)
    top_docs = [docs[i] for i in I[0]]
##### 2. Reranker #####
#reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
#reranker.model.to("cuda")
#scores = reranker.compute_score(pairs)  
    pairs = [(query, d) for d in top_docs]
    scores = reranker.predict(pairs)
    scores = np.array(scores)
    min_score, max_score = scores.min(), scores.max()
    normalized_scores = (scores - min_score) / (max_score - min_score + 1e-8)

    ranked_docs = sorted(
        [{"text": d, "score": s} for d, s in zip(top_docs, normalized_scores)],
        key=lambda x: x["score"],
        reverse=True
    )
    ranked_docs = ranked_docs[:2]
    weighted_context = ""
    for doc in ranked_docs: 
        weight = doc["score"] / sum(d["score"] for d in ranked_docs)
        if weight > 0.1:
            weighted_context += f"(score: {weight:.2f}) {doc['text']} "
    
    ##### 3. Summary #####
    TOKEN_THRESHOLD = 200
    if len(weighted_context) > TOKEN_THRESHOLD:
        prompt_summary = f"""
    You are an expert assistant who helps summarize information. 
    Your task is to extract relevant and essential informations from the provided context.

    Follow these specific rules:
    1. **Prioritize higher-weighted text.** Focus on the most important information as indicated by 'score'.
    2. **Maintain factual accuracy.** Only include information from the provided context and do not infer or add external knowledge.
    3. **DO NOT CONTAIN 'score' in summarized texts

    ###context
    {weighted_context}

    Return the essential summarized information(TEXT ONLY, NOT '(score:)') in less than 4 sentences.
    """
        with torch.no_grad():
            inputs = sum_tokenizer(prompt_summary, return_tensors="pt").to(sum_model.device)
            summary_ids = sum_model.generate(**inputs, max_length=128)
            summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    else:
        summary = weighted_context
    ##### 4. Answering #####
    gen_prompt = f"""
    Your task is to answer the ###Question based on the given ###Summary.

    Follow these examples to understand the format:

    Example 1:
    Question: Jim Betts ran against the first American to orbit what?
    Summary: Jim Betts is a former member of the Ohio House of Representatives. He ran against incumbent U.S. Senator John Glenn in 1980. He lost and ran in 1982 for Lieutenant Governor. \
        (score: 0.44) John Herschel Glenn Jr. (July 18, 1921 – December 8, 2016) was a United States Marine Corps aviator, engineer, astronaut, and United States Senator from Ohio. In 1962 he became the first American to orbit the Earth, circling it three times.
    Answer: Earth

    Example 2:
    Question: Siegfriedstellung in the First World War was a defensive position of what country's army?
    Summary:  Hindenburg Line ("Siegfriedstellung" or Siegfried Position) was a German defensive position of World War I, built during the winter of 1916–1917 on the Western Front, from Arras to Laffaux, near Soissons on the Aisne.
    Answer: German
    
    Now answer the following ###Question:

    ###Question:
    {query}

    ###Summary:
    {summary}

    ###Answer:
    """
    # 임시설정
    with torch.no_grad():
        inputs = sum_tokenizer(gen_prompt, return_tensors="pt").to(sum_model.device)
        outputs = sum_model.generate(**inputs, max_new_tokens=128, top_p=0.8,do_sample=False)
        answer = sum_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("----------------------------------------------------------------------")
    print("\n💬 Query:\n", query)
    print("📝 Summary:\n", summary)
    print("\n💬 Final Answer:\n", answer)
    print("\n💬 Grounded truth:\n", truths[t])
    print("----------------------------------------------------------------------")
##### 5. Evaluation #####
   