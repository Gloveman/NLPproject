def get_sents_than_rerank(
    query,
    sents_metadata,
    gpu_index,
    embedder,
    reranker,
    top_k_search=50,
    top_k_rerank=5
):
    """
    baseline 방식의 추천 시스템: Sentence들을 검색하고 바로 CrossEncoder로 Rerank
    """
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = gpu_index.search(query_emb, top_k_search)

    candidate_docs = [sents_metadata[idx]["text"] for idx in I[0]]
    pairs = [[query, doc] for doc in candidate_docs]
    scores = reranker.predict(pairs)

    sorted_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    selected_docs = [doc for doc, _ in sorted_results[:top_k_rerank]]
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
    """
    Our 방식의 추천 설정:
    1. 문서(단락) 단위 검색
    2. 단락 Reranking (top 3 문서)
    3. 해당 문서의 모든 문장 추출
    4. 문장 Reranking (CrossEncoder)
    5. 상대적 Threshold (top score * 0.6)에 따라 문장 필터링 (최소 3개, 최대 8개)
    """
    # 1. 문서 검색 (top 10)
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
        matching_sents = [(title, m["text"]) for m in sent_metadata if m["title"] == title]
        candidate_docs.extend(matching_sents)

    # 4. 문장 Reranking
    pairs = [[query, doc[0]+":"+doc[1]] for doc in candidate_docs]
    scores = sents_reranker.predict(pairs)

    # 5. Relative threshold 선택 (ratio=0.6)
    sorted_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)

    if not sorted_results:
        return []

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

    return [sentence for (_, sentence), _ in selected]
