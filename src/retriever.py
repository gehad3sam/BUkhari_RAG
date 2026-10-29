# import os
# from qdrant_client import QdrantClient
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import CrossEncoder
#
#
# embedding_model = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-m3",
#     model_kwargs={'device': 'cuda'},
#     encode_kwargs={'normalize_embeddings': True}
# )
# reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda') # Explicitly use CPU
#
# def search_and_rerank_cpu(user_query,qdrant_url, qdrant_api_key,collection_name="sahih_bukhari", qdrant_top_k=30, final_top_k=5):
#     # 1. Vectorize query using CPU embedding model
#     user_query = normalize(user_query)
#     client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
#     query_vector = embedding_model.embed_query(user_query)
#
#     # 2. Retrieve initial candidates from Qdrant Cloud
#     search_results = client.query_points(
#         collection_name=collection_name,
#         query=query_vector,
#         limit=qdrant_top_k
#     )
#     candidate_chunks = search_results.points
#
#     # 3. Format pairs for the Cross-Encoder
#     pairs = [[user_query, point.payload['original_text']] for point in candidate_chunks]
#
#     # 4. Predict scores on CPU
#     rerank_scores = reranker_model.predict(pairs)
#
#     # 5. Sort and structure results
#     reranked_results = []
#     for score, point in zip(rerank_scores, candidate_chunks):
#         reranked_results.append({
#             "rerank_score": float(score),
#             "qdrant_score": point.score,
#             "hadith_number": point.payload['hadith_number'],
#             "book_title": point.payload['book_title'],
#             "original_text": point.payload['original_text']
#         })
#
#     reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
#
#     # 6. Print Top Results
#     # print(f"\n--- 🎯 TOP {final_top_k} RERANKED RESULTS (CPU MODE) ---")
#     # for i, item in enumerate(reranked_results[:final_top_k], 1):
#     #     print(f"\nRank #{i}")
#     #     print(f"Reranker Score: {item['rerank_score']:.4f}")
#     #     print(f"Hadith #: {item['hadith_number']} | Book: {item['book_title']}")
#     #     print(f"Text Preview: {item['original_text'][:]}")
#     #     print("-" * 50)
#
#     return reranked_results[:final_top_k]
#
#
# import re
#
#
# def normalize(text):
#     if not isinstance(text, str):
#         return ""
#     # 1. Expand honorific symbol (Reassigned properly)
#     text = text.replace("ﷺ", "صلى الله عليه وسلم")
#
#     # 2. Strip Tashkeel (Diacritics) & Tatweel (Kashida '\u0640')
#     text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
#
#     # 3. Collapse duplicate whitespace
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()
#



from typing import List, Dict, Any

COLLECTION_NAME = "sahih_bukhari_with_cohere"
SIMILARITY_THRESHOLD = 0.42
QDRANT_TOP_K = 15
RERANK_TOP_N = 7
MIN_RELEVANCE_SCORE = 0.10


def retrieve_candidates(
    user_query: str,
    embedding_model: Any,
    qdrant_client: Any,
    collection_name: str = COLLECTION_NAME,
    limit: int = QDRANT_TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> List[Any]:
    """Stage 1: Dense Vector Retrieval from Qdrant with similarity threshold filtering."""
    query_vector = embedding_model.embed_query(user_query)
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
    )
    return [
        point for point in search_results.points
        if point.score >= similarity_threshold
    ]


def rerank_candidates(
    user_query: str,
    candidate_chunks: List[Any],
    cohere_client: Any,
    top_n: int = RERANK_TOP_N,
    min_relevance_score: float = MIN_RELEVANCE_SCORE,
) -> List[Dict[str, Any]]:
    """Stage 2: Deduplicate Qdrant chunks and rerank via Cohere Multilingual Reranker v3."""
    unique_candidates = []
    seen_texts = set()
    for point in candidate_chunks:
        text = point.payload["original_text"]
        if text not in seen_texts:
            unique_candidates.append(point)
            seen_texts.add(text)

    docs = [
        f"كتاب: {point.payload['book_title']} | {point.payload['original_text']}"
        for point in unique_candidates
    ]

    response = cohere_client.rerank(
        model="rerank-multilingual-v3.0",
        query=user_query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )

    top_chunks = []
    for result in response.results:
        if result.relevance_score >= min_relevance_score or len(top_chunks) < 3:
            original_point = unique_candidates[result.index]
            top_chunks.append({
                "score": result.relevance_score,
                "hadith_number": original_point.payload["hadith_number"],
                "book_title": original_point.payload["book_title"],
                "text": original_point.payload["original_text"],
            })

    return top_chunks