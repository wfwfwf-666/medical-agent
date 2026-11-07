# 收集回答和检索上下文，构建 ragas Dataset
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragas import Dataset

from agents.generation import generate_answer
from app.query import retrieve_with_reflection

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "data" / "eval" / "questions.jsonl"
EXP_DIR = ROOT / "experiments"
REQUEST_DELAY = 1


def load_questions():
    text = EVAL_FILE.read_text(encoding="utf-8").strip()
    rows = []

    if text.startswith("["):
        rows = json.loads(text)  # 整个数组一次解析
    else:
        for line in text.splitlines():  # 一行一条，分别解析
            line = line.strip()
            rows.append(json.loads(line))
    out = []
    for r in rows:
        q = (r.get("user_input") or "").strip()
        ref = r.get("reference") or ""
        if q:
            out.append({"user_input": q, "reference": ref,
                        "type": r.get("type", "unknown")})
    print(f"加载了 {len(out)} 个评估问题")
    return out

def create_dataset():
    """有本地数据就复用，没有就现跑一遍 RAG 生成"""
    dataset = Dataset(
        name="rag_eval",
        backend="local/csv",
        root_dir=str(EXP_DIR)
    )
    if (EXP_DIR / "datasets" / "rag_eval.csv").exists():
        dataset.reload()
    if len(dataset) > 0:
        print(f"加载了 {len(dataset)} 个样本，跳过生成")
        return dataset

    items = load_questions()
    print(f"开始收集，共 {len(items)} 题，间隔 {REQUEST_DELAY}s，预计 {len(items) * 5}s")

    for i, item in enumerate(items):
        q = item["user_input"]
        r = retrieve_with_reflection(q)
        a = generate_answer(q, r["docs"])

        dataset.append({
            "user_input": q,
            "response": a["answer"],
            "retrieved_contexts": [d["text"] for d in r["docs"]],
            "reference": item["reference"],
            "type": item["type"],
        })
        print(f"[{i + 1}/{len(items)}] {q[:30]} ✓ contexts={len(r['docs'])}")

        if i < len(items) - 1:
            time.sleep(REQUEST_DELAY)

    dataset.save()
    print(f"已保存 {len(items)} 条 -> {EXP_DIR / 'datasets' / 'rag_eval.csv'}")
    return dataset



if __name__ == "__main__":
    ds = create_dataset()
    print("样本数:", len(ds))
    print(ds.__getitem__(3))