
from FlagEmbedding import FlagReranker
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, re
import numpy as np

#embedder=SentenceTransformer("BAAI/bge-small-en-v1.5")

question = "What allows computers to learn forom data without being explicitly programmed?"

# 1️⃣ 가중치 정렬된 문서 리스트 예시
docs = [
    "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
    "AI systems can be categorized into narrow AI, which is designed for specific tasks, and general AI, which has the potential to perform any intellectual task a human can do.",
    "Machine Learning (ML) is a subset of AI that allows computers to learn from data without being explicitly programmed.",
    "Deep learning, a subset of ML, uses neural networks with many layers to analyze various types of data.",
    "AI technologies are used in various industries such as healthcare, finance, transportation, and entertainment.",
    "Self-driving cars use AI to analyze their surroundings and make decisions in real-time.",
    "Natural Language Processing (NLP) is a branch of AI that enables computers to understand, interpret, and respond to human language.",
    "AI can also be used in robotics to enable machines to perform physical tasks autonomously.",
    "While AI has many benefits, there are concerns about job displacement and ethical considerations in decision-making.",
    "AI is expected to play a major role in solving complex global challenges such as climate change and disease prevention."
]

#reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
#reranker.model.to("cuda")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', device = 'cuda')

pairs = [(question, d) for d in docs]
#scores = reranker.compute_score(pairs)  
scores = reranker.predict(pairs)
scores = np.array(scores)
min_score, max_score = scores.min(), scores.max()
normalized_scores = (scores - min_score) / (max_score - min_score + 1e-8)

ranked_docs = sorted(
    [{"text": d, "score": s} for d, s in zip(docs, normalized_scores)],
    key=lambda x: x["score"],
    reverse=True
)

# 4️⃣ 가중치 기반 요약 입력 만들기
ranked_docs=ranked_docs[:4]
weighted_context = ""
for doc in ranked_docs: 
    weight = doc["score"] / sum(d["score"] for d in ranked_docs)
    if weight > 0.1:
        weighted_context += f"(score: {weight:.2f}) {doc['text']} "

print("📝Initial:\n",weighted_context)

# 2️⃣ T5 summarizer 준비
sum_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", dtype=torch.float16, device_map="auto")
sum_model.to("cuda")

prompt_summary = f"""
You are an expert assistant who helps summarize information. 
Your task is to extract only the most relevant and essential information from the provided context.

Follow these specific rules:
1. **Prioritize higher-weighted text.** Focus on the most important information as indicated by 'score'.
2. **Maintain factual accuracy.** Only include information from the provided context and do not infer or add external knowledge.
3. **DO NOT CONTAIN 'score' in summarized texts

###context
{weighted_context}

Return the essential summarized information(TEXT ONLY, NOT SCORE) in less than 3 sentences.
"""

TOKEN_THRESHOLD = 150

#inputs = sum_tokenizer(prompt_summary, return_tensors="pt").to(sum_model.device)
#summary_ids = sum_model.generate(**inputs, max_length=128)
#summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#print("📝Summary:\n", summary)


# 5️⃣ 답변 생성 (TinyLlama)

# 3️⃣ TinyLlama 생성 모델 준비
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small", 
    dtype=torch.float16,
    device_map="auto"
)
gen_model.to("cuda")

gen_prompt = f"""
Your task is to answer the ###Question based on the given ###Summary. If the ###Summary does not contain enough information, answer with "NO ANSWER".

Follow these examples to understand the format:

Example 1:
Question: Who built the Eiffel Tower?
Summary: The Eiffel Tower was designed and constructed by Gustave Eiffel in 1889.
Answer: Gustave Eiffel

Example 2:
Question: When was the Eiffel Tower completed?
Summary: The Eiffel Tower was completed in 1889 for the World's Fair.
Answer: 1889

Now answer the following ###Question:

###Question:
{question}

###Summary:
{weighted_context}

###Answer:
"""

inputs = gen_tokenizer(gen_prompt, return_tensors="pt").to(gen_model.device)
outputs = gen_model.generate(**inputs, max_new_tokens=64, top_p=1.0,do_sample=False)
answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n💬 Final Answer:\n", answer)
