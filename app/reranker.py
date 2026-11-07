import dashscope
from http import HTTPStatus
from app.config import api_key

MODEL = "qwen3.7-text-rerank"


def rerank(
    query: str,
    rrf_docs: list[dict],
    top_n: int = 5,
) -> list[dict]:
    if not rrf_docs:
        return []

    # 取出每个片段的正文（字段名是 text，不是 search_text）
    documents = []
    for doc in rrf_docs:
        documents.append(doc["text"])

    try:
        resp = dashscope.TextReRank.call(
            model=MODEL,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=False,
            api_key=api_key,
        )
    except Exception as e:
        print("重排请求异常:", type(e).__name__, str(e)[:80])
        return rrf_docs[:top_n]

    if resp.status_code != HTTPStatus.OK:
        print("重排失败:", resp.status_code, resp.message)
        return rrf_docs[:top_n]

    # 按接口返回的顺序取值，把分数贴到原片段上
    result = []
    for item in resp.output.results:
        one = dict(rrf_docs[item.index])
        one["rerank_score"] = float(item.relevance_score)
        result.append(one)

    return result[:top_n]
