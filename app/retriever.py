from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

from app.config import MILVUS_URI, MILVUS_COLLECTION
from app.embedding import embed_query

_client = None

# 懒加载
def get_client():
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
    return _client

#混合检索
def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    top_k = max(1, min(int(top_k), 20))
    client = get_client()

    dense_req = AnnSearchRequest(
        data=[embed_query(query)],
        anns_field="dense",
        param={"metric_type": "COSINE"},
        limit=max(top_k * 4, 10)
    )
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse",
        param={"metric_type": "BM25"},
        limit=max(top_k * 4, 10)
    )

    hits = client.hybrid_search(
        collection_name=MILVUS_COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=["text", "source", "chunk_index"]
    )[0]

    return [{"source": h["entity"]["source"],
             "text": h["entity"]["text"],
             "score": round(float(h["distance"]), 6),
             "chunk_index": h["entity"]["chunk_index"]} for h in hits]


# 3.知识库状态
def collection_stats() -> dict:
    stats = get_client().get_collection_stats(MILVUS_COLLECTION)
    return {"collection": MILVUS_COLLECTION,
            "row_count": int(stats.get("row_count", 0) or 0)}
