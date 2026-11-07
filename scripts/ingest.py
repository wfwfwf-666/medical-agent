"""把 data/docs 里的医疗文档切片、向量化，写入 Milvus（dense + BM25 稀疏）。

↳ 用到的 LangChain 知识点：
    DirectoryLoader / TextLoader   -> 文档加载，产物是 Document
    RecursiveCharacterTextSplitter -> 递归字符切片，产物还是 Document
    OllamaEmbeddings               -> app/embedding.py 里封装好的

↳ 为什么 Milvus 这层不用 langchain-milvus：
    我们要 BM25 Function（服务端自动生成稀疏向量）+ RRF 混合检索，
    langchain-milvus 对这两个特性覆盖不全，所以 schema 和查询自己写。
    面试被问"为什么不用现成 vectorstore"，这就是答案。
"""
import argparse
import sys
import uuid
from collections import defaultdict
from pathlib import Path
# ↳ 让 scripts/ 目录下的脚本能 import 到项目根的 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import (AnnSearchRequest, DataType, Function, FunctionType,
                      MilvusClient, RRFRanker)

from app.config import (DOCS_DIR, EMBEDDING_DIM, MILVUS_COLLECTION,
                        MILVUS_URI)
from app.embedding import embed_documents, embed_query

CHUNK_SIZE = 450      # ↳ 每个切片约 450 字
CHUNK_OVERLAP = 80    # ↳ 相邻切片重叠 80 字，避免答案被切断
INSERT_BATCH = 200


# ---------------------------------------------------------------- 1. 加载
def load_documents():
    """↳ DirectoryLoader 按 glob 批量读文件，每个文件变成一个 Document。"""
    docs = []
    for pattern in ("**/*.md", "**/*.txt"):
        loader = DirectoryLoader(
            str(DOCS_DIR),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        docs.extend(loader.load())

    if not docs:
        raise SystemExit(f"{DOCS_DIR} 里没有 .md/.txt 文档，先放几篇再跑")
    print(f"[load] 读到 {len(docs)} 个文件")
    return docs


# ---------------------------------------------------------------- 2. 切片
def split_documents(docs):
    """↳ RAG 核心：chunk_size / chunk_overlap 的取舍。
    中文按句号问号切，比按固定长度硬切更不容易切坏语义。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        add_start_index=True,   # 在 metadata 里记下起始位置，方便定位
    )
    chunks = splitter.split_documents(docs)
    print(f"[split] 切成 {len(chunks)} 片")
    return chunks


# ---------------------------------------------------------------- 3. 转换
def chunks_to_rows(chunks):
    """把 Document 转成 Milvus 能插入的 dict 列表。"""
    rows = []
    counter = defaultdict(int)          # 每个文件的切片从 0 开始编号
    for doc in chunks:
        source = Path(doc.metadata.get("source", "unknown")).name
        rows.append({
            "id": uuid.uuid4().hex,     # 主键，用 uuid 防止重复
            "text": doc.page_content,   # 切片正文
            "source": source,           # 来自哪个文件（评测引用要用）
            "chunk_index": counter[source],
        })
        counter[source] += 1
    return rows


# ---------------------------------------------------------------- 4. 建表
def build_collection(client: MilvusClient, recreate: bool):
    """建 collection。注意三个关键点，见注释。"""
    if client.has_collection(MILVUS_COLLECTION):
        if not recreate:
            print(f"[milvus] 集合已存在: {MILVUS_COLLECTION}")
            return
        client.drop_collection(MILVUS_COLLECTION)
        print(f"[milvus] 已删除旧集合")
    #创建主键
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False
    )
    schema.add_field(
        "id",
        DataType.VARCHAR,
        is_primary=True,
        max_length=64
    )
    # ↳ 关键点 1：enable_analyzer + chinese 分词器，BM25 才能按中文词切分
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=8192,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    )

    # ↳ 关键点 2：dense 是我们自己算的语义向量（1024 维）稠密向量字段
    schema.add_field(
        "dense",
        DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIM
    )
    # ↳ 关键点 3：sparse 不手动填，由下面的 BM25 Function 自动生成 稀疏向量字段
    schema.add_field(
        "sparse",
        DataType.SPARSE_FLOAT_VECTOR
    )
    schema.add_field(
        "source",
        DataType.VARCHAR,
        max_length=512
    )
    schema.add_field(
        "chunk_index",
        DataType.INT64
    )

    schema.add_function(Function(
        name="text_bm25",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX",
                           metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="AUTOINDEX",
                           metric_type="BM25")

    client.create_collection(MILVUS_COLLECTION, schema=schema,
                             index_params=index_params)
    client.load_collection(MILVUS_COLLECTION)
    print(f"[milvus] 已创建并加载 {MILVUS_COLLECTION}")


# ---------------------------------------------------------------- 5. 自测
def self_test(client: MilvusClient, query: str):
    """↳ 混合检索 = 语义(dense) + 关键词(BM25)，再用 RRF 合并两个排名。
    RRF 不看分数绝对值，只看名次，所以两路分数尺度不同也能融合。
    """
    #创建稠密向量请求和稀疏向量请求
    dense_req = AnnSearchRequest(
        data=[embed_query(query)],
        anns_field="dense",
        param={"metric_type": "COSINE"},
        limit=10,
    )
    # ↳ sparse 这路的 data 直接传"原文"，Milvus 服务端会自己分词+算 BM25
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse",
        param={"metric_type": "BM25"},
        limit=10,
    )
    #混合检索
    hits = client.hybrid_search(
        collection_name=MILVUS_COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=5,
        output_fields=["text", "source"],
    )[0]

    print(f"\n=== 混合检索自测: {query} ===")
    for i, h in enumerate(hits, 1):
        snippet = h["entity"]["text"].replace("\n", " ")[:60]
        print(f"{i}. [{h['distance']:.4f}] {h['entity']['source']} | {snippet}...")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="删掉重建集合")
    ap.add_argument("--query", default="高血压患者饮食要注意什么")
    args = ap.parse_args()

    chunks = split_documents(load_documents())
    rows = chunks_to_rows(chunks)

    client = MilvusClient(uri=MILVUS_URI)
    build_collection(client, args.recreate)

    stats = client.get_collection_stats(MILVUS_COLLECTION)
    if int(stats.get("row_count", 0) or 0) > 0:
        print(f"[milvus] 已有 {stats['row_count']} 行，跳过写入（重灌加 --recreate）")
    else:
        # ↳ 批量向量化：一次请求多条，比一条条调快几十倍
        vectors = embed_documents([r["text"] for r in rows])
        for r, v in zip(rows, vectors):
            r["dense"] = v
        for i in range(0, len(rows), INSERT_BATCH):
            client.insert(MILVUS_COLLECTION, rows[i:i + INSERT_BATCH])
            print(f"[milvus] 已写入 {min(i + INSERT_BATCH, len(rows))}/{len(rows)}")
        client.flush(MILVUS_COLLECTION)

    print("[milvus] 统计:", client.get_collection_stats(MILVUS_COLLECTION))
    self_test(client, args.query)


if __name__ == "__main__":
    main()