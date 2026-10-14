"""年报文本切分：用 RecursiveCharacterTextSplitter 切分成带元数据的 chunk。

年报特点：
- 每份 100-400 页，单页字符数通常 1000-2000
- 章节边界明显（"第一节"、"第二节"）
- 表格内容占比较大（PyMuPDF 提取后表格列名/数值会被打散到不同行）

策略：
- chunk_size=800，overlap=100（年报段落通常 500-1000 字）
- 分隔符优先级：双换行 > 单换行 > 句号 > 分号 > 逗号 > 空格
- 跳过 is_likely_cover_or_toc=True 的低密度页（封面、目录、签字页）
- 每个 chunk 带元数据：stock_code、company、year、page、source_file
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.pdf_loader import PdfDocument, parse_meta_from_filename


# 切分参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 中文友好分隔符（按优先级从高到低）
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]


def make_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """构造文本切分器。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator=False,
        is_separator_regex=False,
    )


def make_chunks(doc: PdfDocument) -> list[dict[str, Any]]:
    """把一份 PdfDocument 切分成带元数据的 chunk 列表。

    Returns:
        [{"text": ..., "stock_code": ..., "company": ..., "year": ..., "page": int, "source_file": ...}]
    """
    # 从文件名解析公司元数据
    meta = parse_meta_from_filename(doc.file_name)
    if meta:
        stock_code, company, year = meta
    else:
        stock_code, company, year = "unknown", "unknown", 0

    splitter = make_splitter()
    chunks: list[dict[str, Any]] = []

    for page in doc.pages:
        # 跳过低密度页（封面、目录、签字页等）
        if page.is_likely_cover_or_toc:
            continue

        # 切分当前页文本
        page_chunks = splitter.split_text(page.text)
        for text in page_chunks:
            # 跳过切完后仍太短的片段（页眉、目录项残留）
            if len(text.strip()) < 50:
                continue
            chunks.append(
                {
                    "text": text.strip(),
                    "stock_code": stock_code,
                    "company": company,
                    "year": year,
                    "page": page.page_num,
                    "source_file": doc.file_name,
                }
            )

    return chunks


def make_chunks_for_docs(docs: list[PdfDocument]) -> list[dict[str, Any]]:
    """批量切分多份 PdfDocument。"""
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        all_chunks.extend(make_chunks(doc))
    return all_chunks


def chunks_summary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """生成切分统计摘要，用于日志输出。"""
    if not chunks:
        return {"total": 0, "avg_len": 0, "by_company": {}}

    by_company: dict[str, int] = {}
    for c in chunks:
        company = c.get("company", "unknown")
        by_company[company] = by_company.get(company, 0) + 1

    total_chars = sum(len(c["text"]) for c in chunks)
    return {
        "total": len(chunks),
        "avg_len": total_chars // len(chunks),
        "by_company": by_company,
        "total_chars": total_chars,
    }


if __name__ == "__main__":
    # 自测：切分茅台年报，打印统计
    import sys
    from pathlib import Path

    from src.rag.pdf_loader import extract_pdf

    if len(sys.argv) < 2:
        print("用法: python -m src.rag.chunker <pdf_path|dir>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        from src.rag.pdf_loader import extract_all_pdfs
        docs = extract_all_pdfs(target)
    else:
        docs = [extract_pdf(target)]

    all_chunks = make_chunks_for_docs(docs)
    summary = chunks_summary(all_chunks)
    print(f"=== 切分统计 ===")
    print(f"PDF 数: {len(docs)}")
    print(f"chunk 总数: {summary['total']:,}")
    print(f"平均 chunk 长度: {summary['avg_len']} 字符")
    print(f"总字符数: {summary['total_chars']:,}")
    print(f"\n按公司分布:")
    for company, count in summary["by_company"].items():
        print(f"  {company}: {count} chunks")

    if all_chunks:
        print(f"\n--- 预览第 5 个 chunk ---")
        print(all_chunks[4]["text"][:400])
        print(f"\n元数据: {all_chunks[4]}")
