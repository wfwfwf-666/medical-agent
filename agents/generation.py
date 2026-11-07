from langchain.messages import HumanMessage, SystemMessage

from app.llm import model_generation, model_retriever

# 2.系统提示词 —— 防幻觉的第一道闸：只允许依据资料回答
SYSTEM_PROMPT = """你是医疗健康问答助手，服务对象是普通患者。你会收到【参考资料】和【用户问题】。

回答规则：
1. 只使用【参考资料】中的信息回答。资料里没写的具体事实（数值、药名、剂量、检查指标）一律不得编造。
2. 可以改写、合并、分点整理资料，让答案通顺易读；但不得改变原意，不得引入资料以外的新事实。
3. 每个关键结论后用括号标注来源，格式如（来源：hypertension_diet.md）。
4. 资料不足以支撑回答时，直接回复："根据现有资料无法确定，建议咨询专业医生。"不要猜测。
5. 若不同资料的说法或数值不一致，并列说明差异，并建议以医生当面指导为准，不要只挑一个。
6. 遇到急症信号（胸痛、呼吸困难、意识改变、大出血、自伤倾向等），优先提示立即就医或拨打 120。这是唯一允许超出参考资料的内容。
7. 不做诊断、不开处方、不给具体用药剂量。
8. 回答分点、简短，先给结论再给依据。结尾固定一行：以上内容仅供参考，不能替代医生面诊。
9. 只回答资料中出现的内容，不要补充背景知识"""


# 3.把检索片段拼成给模型看的上下文
def build_context(docs: list[dict]) -> tuple[str, list[str]]:
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(
            f"[资料{i}] 来源：{d['source']} 第{d['chunk_index']}片\n{d['text']}"
        )
    sources = sorted({d["source"] for d in docs})
    return "\n\n".join(parts), sources


# 4.主流程：主模型 → 失败切备用，并记录是否兜底
def generate_answer(
    question: str,
    docs: list[dict],
    history_symptoms: str = "",
    retrieval_degraded: bool = False,
) -> dict:
    context, sources = build_context(docs)

    extra = ""
    if history_symptoms:
        extra += f"\n【用户历史症状（跨会话记忆，可参考）】\n{history_symptoms}\n"
    if retrieval_degraded:
        extra += "\n【注意】本次检索多轮未命中，回答须更保守，并建议就医。\n"

    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"{extra}\n【资料】\n{context}\n\n【问题】\n{question}"),
    ]

    # 4.1 先试主模型（DeepSeek）
    try:
        resp = model_generation.invoke(messages)
        model_used = "primary"
    except Exception as e:                      # noqa: BLE001
        # 4.2 主模型超时/报错 → 切备用（Qwen）
        print(f"[generate] 主模型失败，切备用: {type(e).__name__}: {str(e)[:80]}")
        resp = model_retriever.invoke(messages)
        model_used = "fallback"

    return {
        "answer": resp.content,
        "sources": sources,
        "model_used": model_used,
        "doc_count": len(docs),
    }