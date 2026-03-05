from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

OUTPUT_DIR = os.path.expanduser("~/faiss")
FAISS_PATH = os.path.join(OUTPUT_DIR, "bridge_val_faiss.index")
META_PATH = os.path.join(OUTPUT_DIR, "bridge_val_metadata.json")

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor",split = "validation")

# ✅ 2. "bridge" 타입만 필터링
bridge_data = [d for d in dataset if d["type"] == "bridge"]
print(f"Total bridge examples: {len(bridge_data)}")

# ✅ 3. paragraph 단위로 변환 (중복 title 제거)
paragraphs = []
titles = []
ids = []

seen_titles = set()  # 중복 방지용

for item in bridge_data:
    ctx = item["context"]
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        if len(sents) == 0:
            continue
        if title in seen_titles:  # 이미 추가된 제목이면 스킵
            continue

        seen_titles.add(title)
        paragraph = " ".join(sents)
        paragraphs.append(paragraph)
        titles.append(title)
        ids.append(item["id"])  # 소스 질문 ID (optional)

print(f"Unique paragraphs: {len(paragraphs)}")

# ✅ 4. 문서 임베딩 (경량 모델 예시)
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

embeddings = embedder.encode(
    paragraphs, batch_size=64, convert_to_numpy=True, show_progress_bar=True
)

# ✅ 5. FAISS index 구축
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) #Inner product로 유사도 계산 (=cosine similarity)
faiss.normalize_L2(embeddings)
index.add(embeddings)
print(f"FAISS index contains {index.ntotal} unique paragraphs")

# ✅ 6. 메타데이터 저장
faiss.write_index(index, FAISS_PATH)

meta = [
    {"id": pid, "title": t, "paragraph": p}
    for pid, t, p in zip(ids, titles, paragraphs)
]

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("✅ Finished: FAISS index and metadata saved (no duplicate titles).")