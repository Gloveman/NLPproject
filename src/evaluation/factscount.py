
from datasets import load_dataset
import numpy as np

# 1. 데이터셋 로드
print("⏳ Loading HotpotQA validation set...")
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

# 2. Supporting Fact 개수 계산
# supporting_facts 필드는 {'title': [...], 'sent_id': [...]} 형태입니다.
# 리스트의 길이를 세면 됩니다.
counts = [len(x['supporting_facts']['title']) for x in dataset]

# 3. 통계량 계산
max_count = np.max(counts)
min_count = np.min(counts)
avg_count = np.mean(counts)
median_count = np.median(counts)
p95 = np.percentile(counts, 95) # 상위 5% 커트라인
p99 = np.percentile(counts, 99) # 상위 1% 커트라인

print("-" * 30)
print(f"🔍 Supporting Facts Statistics")
print("-" * 30)
print(f"🔹 Max Count (최대값): {max_count}")
print(f"🔹 Min Count (최소값): {min_count}")
print(f"🔹 Average (평균): {avg_count:.2f}")
print(f"🔹 Median (중앙값): {median_count}")
print(f"🔹 95th Percentile (상위 95%): {p95}")
print(f"🔹 99th Percentile (상위 99%): {p99}")
print("-" * 30)
