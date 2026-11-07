# 1.全局唯一的模型实例，四个 Agent 都从这里 import，别各自 init 一遍
import os

from langchain.chat_models import init_chat_model

from app.config import LLM_MODEL_DEEPSEEK, LLM_MODEL_QWEN, api_key, base_url, OLLAMA_URL
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI
#检索模型
model_retriever = init_chat_model(
    model = LLM_MODEL_QWEN,
    model_provider="openai",
    api_key=api_key,
    base_url=base_url,
    extra_body={"enable_thinking": False},
)

#回答模型
model_generation = init_chat_model(
    model = LLM_MODEL_DEEPSEEK,
    extra_body={"thinking": {"type": "disabled"}},
)

llm_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

embedding_client = AsyncOpenAI(
    api_key="ollama",
    base_url=OLLAMA_URL + "/v1",
    timeout=120,
    max_retries=0
)

# 文本推理模型
evaluator_llm = llm_factory(
    model="deepseek-chat",
    client=llm_client,
    max_tokens=384000
)

# 向量模型
evaluator_embeddings = embedding_factory(
    "openai",
    model="bge-m3",
    client=embedding_client
)
