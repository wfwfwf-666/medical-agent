# 读 .env、常量

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "medical_docs")
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("GATEWAY_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("GATEWAY_BASE_URL")


EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")
DASHSCOPE_API_KEY=os.getenv("DASHSCOPE_API_KEY")
OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL")

DOCS_DIR = ROOT / "data" / "docs"
LLM_MODEL_DEEPSEEK = (os.getenv("LLM_MODEL_DEEPSEEK")
                      or os.getenv("LLM_MODEL")
                      or "deepseek-v4-flash")
LLM_MODEL_QWEN = os.getenv("LLM_MODEL_QWEN", "qwen3.8-flash")
MAX_CLARIFY_ROUNDS = int(os.getenv("MAX_CLARIFY_ROUNDS", "3"))
MAX_RETRIEVAL_RETRY = int(os.getenv("MAX_RETRIEVAL_RETRY", "2"))
RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", "5"))