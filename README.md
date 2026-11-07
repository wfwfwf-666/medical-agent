# 医疗知识库 RAG 问答系统

基于 Milvus 的检索增强问答（RAG）系统：稠密 + BM25 混合检索 → RRF 融合 → cross-encoder 重排 → 受控生成（来源标注 / 拒答 / 急症提示），并配套 RAGAS 评测用于参数决策。

> ⚠️ 技术演示项目，语料为公开科普资料，输出不构成任何医疗建议。

## 架构

```
用户问题
   │
   ├─ 混合召回（Milvus）
   │     ├─ dense ：bge-m3 1024 维，COSINE
   │     └─ sparse：BM25，中文 analyzer 分词
   │
   ├─ RRF 融合（k=60）
   │
   ├─ 充分性判断 ──不足──→ 改写 query 重检（≤2 次）
   │
   ├─ cross-encoder 重排（候选 20 → 保留 5）
   │
   ├─ 上下文组装（片段来源标注）
   │
   └─ 生成（仅依据资料 / 引用标注 / 证据不足即拒答 / 急症优先引导就医）
```

## 技术栈

`Python` `LangChain` `Milvus 2.6` `bge-m3 (Ollama)` `DashScope Rerank` `RAGAS` `Docker Compose`

## 目录结构

```
app/            config / embedding / retriever（混合检索）/ query（改写重检）/ reranker / llm
agents/         生成模块（提示词、引用、拒答、主备降级）
scripts/        ingest（入库）· build_eval_set（构造评测集）· run_ragas（评测）
data/docs/      语料（Markdown）
data/eval/      评测集（含题型标注）
experiments/    评测结果 CSV
docker-compose.yml
```

## 快速开始

前置条件：Docker Desktop、[Ollama](https://ollama.com)（`ollama pull bge-m3`）、Python 3.11+

```bash
# 1. 启动 Milvus + etcd + MinIO（Postgres 为预留组件，可删除）
docker compose up -d

# 2. 配置环境变量
cp .env.example .env
# 填入 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY 等

# 3. 文档入库（切分 + 向量化 + 写入 Milvus）
python scripts/ingest.py --recreate

# 4. 跑一遍 RAG，收集 response 与 retrieved_contexts，生成评测集
python scripts/build_eval_set.py

# 5. 评测（RAGAS 指标）
python scripts/run_ragas.py
```

可选开关（`.env`）：

```ini
RECALL_TOP_K=5        # 最终保留的片段数
USE_RERANK=false      # 是否启用 cross-encoder 重排
```

## 评测结果

评测集 10 条，按题型分层：可答 5 条（single_hop）/ 库中无答案 4 条（no_answer）/ 急症 1 条（safety）。

### single_hop（5 条）

| 指标 | 无重排（RRF top5） | 有重排（召回 20 → 精排 5） |
|---|---|---|
| context_precision | **1.000** | 0.967 |
| context_recall | 1.000 | 1.000 |
| faithfulness | 0.747 | 0.758 |
| answer_relevancy | 0.879 | 0.884 |
| noise_sensitivity ↓ | 0.430 | 0.457 |

**结论**：重排未带来提升，反而略降。原因是基线 `context_precision` 已达 1.000（返回片段全部相关），没有可优化的噪声空间；5 条样本的 0.03 差异也在噪声范围内。因此重排保留为**可配置开关，默认关闭**。

### 安全性

| 检查项 | 结果 |
|---|---|
| 库中无答案时明确拒答 | 4 / 4 |
| 急症信号优先引导就医 | 1 / 1 |

### 参数选择

对比 `top_k=5` 与 `top_k=10`，检索类指标无显著差异，最终取 5：上下文体积减半，token 成本随之下降。

## 关键设计

- **为什么混合检索**：dense 负责语义泛化，BM25 负责药名、数值等字面精确匹配，医疗场景两者都必需。
- **为什么 RRF 而不是加权求和**：两路分数量纲不同（余弦相似度 vs BM25 权重），RRF 只看名次，免除归一化与人工调权。
- **为什么"多召回、少保留"**：只给重排 5 个候选等于没得挑，先召回 20 条再精排到 5 条。
- **为什么默认拒答**：医疗场景下编造代价高，证据不足时明确说明并提供就医建议，比猜测更安全。

## 已知局限

1. 语料仅 5 篇 / 137 个切片，题目与文档高度对应，检索难度偏低（precision 触顶即为例证）。
2. 评测集仅 10 条（single_hop 5 条），统计意义有限，结论只能反映趋势。
3. 仅支持单轮问答，未做指代消解（多轮追问无法正确理解）。
4. RAGAS 的 `noise_sensitivity` 与 `context_entity_recall` 在本环境稳定性不足（前者依赖模型判定结果构造数组，偶发形状不匹配；后者对中文实体匹配偏弱），未作为主指标。
5. 重排收益受语料规模限制，尚未在更大规模语料上验证。

## 数据与许可

- 代码：MIT
- `data/docs/` 内容来自公开科普资料，版权归原作者所有，仅用于学习演示；如需使用请自行确认授权。
