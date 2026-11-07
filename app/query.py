# 1.检索子 Agent：检索 → 判断够不够 → 不够就改写问题重检（最多 2 次）
import json
import os

from langchain.messages import HumanMessage
from app.config import MAX_RETRIEVAL_RETRY, RECALL_TOP_K
from app.llm import model_retriever
from app.retriever import hybrid_search
from app.reranker import rerank

# 2.判断"检索够不够"用的提示词：只要 JSON，方便程序解析
JUDGE_PROMPT = """你是医疗检索质量评估员。判断下面检索到的片段是否足以回答用户问题。

用户问题：{query}

检索到的片段：
{snippets}

只输出一个 JSON 对象，不要任何解释：
{{"enough": true 或 false, "rewritten_query": "不够时给一个更容易命中的检索问题；够就填空字符串"}}"""


# 3.让模型判断，返回 (够不够, 改写后的问题)
def _judge(query: str, docs: list[dict]) -> tuple[bool, str]:
    snippets = "\n".join(
        f"[{i}] {d['source']} 第{d['chunk_index']}片：{d['text'][:200]}"
        for i, d in enumerate(docs, 1)
    ) or "（一个片段都没检索到）"

    msg = model_retriever.invoke([HumanMessage(JUDGE_PROMPT.format(query=query, snippets=snippets))])
    raw = str(msg.content).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 模型没按格式输出时，宁可当"够用"，别浪费重试次数
        return True, ""

    return bool(data.get("enough")), (data.get("rewritten_query") or "").strip()


# 4.主流程：检索 → 判断 → 够不够 → 不够就改写重检，最多 MAX_RETRIEVAL_RETRY 次
def retrieve_with_reflection(query: str, top_k: int | None = None) -> dict:
    top_k = top_k or RECALL_TOP_K
    current = query
    candidates: list[dict] = []
    docs: list[dict] = []
    trace = []

    # 开关：.env 里写 USE_RERANK=false 就关掉重排（用于 A/B 对比）
    use_rerank = os.getenv("USE_RERANK", "true").lower() == "true"

    for attempt in range(1, MAX_RETRIEVAL_RETRY + 2):
        # 1) 先多召回：取 top_k 的 4 倍，给重排留挑选空间
        candidates = hybrid_search(current, max(top_k * 4, 10))

        # 2) 再精排：开了开关就用模型重排，否则直接取前 top_k 条
        if use_rerank:
            docs = rerank(current, candidates, top_k)
        else:
            docs = candidates[:top_k]

        # 3) 判断这批片段够不够回答问题
        enough, rewritten = _judge(query, docs)

        trace.append({
            "attempt": attempt,
            "used_query": current,
            "hits": len(candidates),
            "kept": len(docs),
            "rerank": use_rerank,
            "enough": enough,
        })

        if enough or not rewritten:
            return {"docs": docs, "attempts": attempt, "trace": trace}

        current = rewritten   # 用模型改写后的问题再检索一次

    return {"docs": docs, "attempts": len(trace), "trace": trace, "degraded": True}
