"""统一 embedding 入口 —— 对外只暴露 embed_documents / embed_query。
以后想换 OpenAI、本地 sentence-transformers，只改这个文件。

↳ 知识点：LangChain 的 Embeddings 抽象
   任何实现 embed_documents(list[str]) -> list[list[float]]
   和 embed_query(str) -> list[float] 的对象，都能插到 RAG 链里。
"""
from langchain_ollama import OllamaEmbeddings

from app.config import EMBEDDING_MODEL, OLLAMA_URL


#引入模型
_embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_URL,
)
embeddings=_embeddings

def embed_documents(texts: list[str]) -> list[list[float]]:
    """批量向量化（入库用）。bge-m3 输出 1024 维。"""
    return _embeddings.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """单条向量化（查询用）。注意：必须和入库用同一个模型。"""
    return _embeddings.embed_query(text)