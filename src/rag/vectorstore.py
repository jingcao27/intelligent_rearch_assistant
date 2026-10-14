"""Chroma 向量数据库：持久化存储 + 检索年报 chunks。

Chroma 特性：
- 单进程嵌入式，数据存在 SQLite + parquet
- 支持元数据过滤（按公司、年份检索）
- 自带相似度检索（cosine / L2 / ip）
- 与 LangChain 集成成熟（langchain_chroma 包）
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# 持久化目录
PERSIST_DIR = "data/vectorstore"
# Collection 名
COLLECTION_NAME = "annual_reports"
# 单批入库最大条数（qwen3.7-text-embedding 限制 batch ≤ 20）
BATCH_SIZE = 20
# 批次间隔（秒）
BATCH_DELAY = 0.5


def get_vectorstore() -> Chroma:
    """获取 Chroma 向量库实例（持久化模式）。

    若 PERSIST_DIR 已存在则会加载已有索引；否则首次写入时自动创建。
    """
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def build_index(chunks: list[dict[str, Any]]) -> Chroma:
    """把 chunks 批量入库到 Chroma。

    自动分批 + 限速，避免触发 DashScope embedding API 限流。
    每批 embedding + 写入约 2-5 秒。

    Args:
        chunks: 切分好的 chunk 列表，每项含 text + metadata 字段

    Returns:
        Chroma 实例
    """
    if not chunks:
        logger.warning("[build_index] 没有可入库的 chunk")
        return get_vectorstore()

    vs = get_vectorstore()
    texts = [c["text"] for c in chunks]
    metadatas = [
        {k: v for k, v in c.items() if k != "text"} for c in chunks
    ]

    total = len(texts)
    success_count = 0
    for i in range(0, total, BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]
        try:
            vs.add_texts(texts=batch_texts, metadatas=batch_meta)
            success_count += len(batch_texts)
            logger.info(
                "[build_index] 入库进度 %d/%d (%.1f%%)",
                success_count,
                total,
                success_count * 100 / total,
            )
        except Exception as e:
            logger.error("[build_index] 第 %d-%d 批入库失败: %s",
                         i, i + len(batch_texts), e)
        time.sleep(BATCH_DELAY)

    logger.info("[build_index] 完成：成功 %d / 总计 %d", success_count, total)
    return vs


def search(
    query: str,
    k: int = 5,
    filter_dict: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """检索年报内容。

    Args:
        query: 查询文本（中文自然语言）
        k: 返回 top-k 条
        filter_dict: Chroma where 过滤条件
            - {"stock_code": "600519"} 只在茅台年报里搜
            - {"company": "宁德时代"} 只在宁德时代年报里搜
            - {"year": 2024} 只在 2024 年报里搜

    Returns:
        [{"text": ..., "metadata": ..., "score": ...}] 列表
    """
    vs = get_vectorstore()
    # similarity_search_with_score 返回 [(Document, float)]
    results = vs.similarity_search_with_score(query, k=k, filter=filter_dict)
    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in results
    ]


def count() -> int:
    """返回当前向量库的 chunk 数量。"""
    try:
        vs = get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0


def clear() -> None:
    """清空向量库（删除整个 collection）。"""
    import shutil

    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR)
    logger.info("[vectorstore] 已清空 %s", PERSIST_DIR)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        n = count()
        print(f"当前向量库 chunk 数: {n}")
        print("\n用法:")
        print("  python -m src.rag.vectorstore count      # 看条数")
        print("  python -m src.rag.vectorstore search '茅台营收'  # 检索")
        print("  python -m src.rag.vectorstore clear       # 清空")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "count":
        print(f"向量库 chunk 数: {count()}")
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("用法: python -m src.rag.vectorstore search <query>")
            sys.exit(1)
        query = sys.argv[2]
        results = search(query, k=3)
        print(f"=== 检索: {query} ===")
        for i, r in enumerate(results):
            meta = r["metadata"]
            print(f"\n[{i+1}] 公司={meta.get('company')} 页码={meta.get('page')} "
                  f"来源={meta.get('source_file')} score={r['score']:.4f}")
            print(f"    {r['text'][:200]}")
    elif cmd == "clear":
        clear()
        print("向量库已清空")
    else:
        print(f"未知命令: {cmd}")
