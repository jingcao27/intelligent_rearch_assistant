"""一键构建 RAG 索引：解析 PDF → 切分 → 向量化 → 入库。

用法：
    python -m src.rag.build_index                # 全量构建（增量跳过已入库 PDF）
    python -m src.rag.build_index --rebuild       # 清空后重建
    python -m src.rag.build_index --only 600519  # 只构建指定股票
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.rag.chunker import chunks_summary, make_chunks_for_docs
from src.rag.pdf_loader import extract_all_pdfs, parse_meta_from_filename
from src.rag.vectorstore import build_index, clear, count

logger = logging.getLogger(__name__)

REPORTS_DIR = "data/reports"


def build(
    reports_dir: Path = Path(REPORTS_DIR),
    only_stock: str | None = None,
    rebuild: bool = False,
) -> int:
    """构建向量索引。返回入库 chunk 数。"""
    # 1. 重建模式：先清空
    if rebuild:
        logger.info("[build] 清空旧索引...")
        clear()

    # 2. 加载所有 PDF
    logger.info("[build] 加载 PDF: %s", reports_dir)
    docs = extract_all_pdfs(reports_dir)

    # 单股票过滤
    if only_stock:
        docs = [d for d in docs if parse_meta_from_filename(d.file_name)
                and parse_meta_from_filename(d.file_name)[0] == only_stock]
        if not docs:
            logger.error("[build] 未找到股票代码 %s 的 PDF", only_stock)
            return 0

    logger.info("[build] 共 %d 份 PDF 待处理", len(docs))

    # 3. 切分
    logger.info("[build] 切分文本...")
    chunks = make_chunks_for_docs(docs)
    summary = chunks_summary(chunks)
    logger.info(
        "[build] 切分完成: 共 %d chunks, 平均 %d 字符, 总 %d 字符",
        summary["total"], summary["avg_len"], summary["total_chars"],
    )
    for company, n in summary["by_company"].items():
        logger.info("         - %s: %d chunks", company, n)

    # 4. 入库
    logger.info("[build] 开始向量化入库...")
    t0 = time.time()
    build_index(chunks)
    elapsed = time.time() - t0
    logger.info("[build] 入库完成，用时 %.1f 秒", elapsed)

    # 5. 验证
    final_count = count()
    logger.info("[build] 当前向量库总条数: %d", final_count)
    return final_count


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAG 索引")
    parser.add_argument("--rebuild", action="store_true", help="清空后重建")
    parser.add_argument("--only", type=str, default="", help="只构建指定股票代码")
    parser.add_argument("--reports-dir", type=str, default=REPORTS_DIR,
                        help="PDF 目录")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    n = build(
        reports_dir=Path(args.reports_dir),
        only_stock=args.only or None,
        rebuild=args.rebuild,
    )
    print(f"\n=== 构建完成，向量库总条数: {n} ===", file=sys.stderr)


if __name__ == "__main__":
    main()
