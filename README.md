# NLPproject
자연어처리 입문 프로젝트 코드

---

### 파일 설명

* rag_baseline.py : baseline 시스템 전체 과정
* buildDB.py : Huggingface에서 dataset을 불러와 vector DB index 및 원본 data 저장 코드(참고용) - vector index에는 embedding만 저장되므로 원본 문서를 별도로 저장해야 함

* summarize_generate.py : 점수 매기기 + 요약 + 답변 생성까지 (참고용)

### 출력 예시(rag_baseline.py)
💬 Query:
&nbsp;What is the real name of the free jazz drummer on the album "Center of the World"?

📝 Summary:
&nbsp;Center of the World is the eponymous album by the free jazz quartet consisting of saxophonist Frank Wrigth, pianist Bobby Few, bassist Alan Silva and drummer Muhammad Ali. (score: 0.42) Muhammad Ali (born Raymond Patterson, 1936) is a free jazz drummer.

💬 Final Answer:
&nbsp;Raymond Patterson

💬 Grounded truth:
&nbsp;Raymond Patterson

