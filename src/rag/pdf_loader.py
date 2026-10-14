"""年报 PDF 解析：使用 PyMuPDF 提取文本，保留页面元数据。

年报 PDF 特点：
- 总页数多（100-300 页）
- 大量表格（财务数据、股权结构）
- 含目录、签字页、附注等"低信息密度"内容

本模块提取每页文本 + 元数据，供后续 chunker 使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF 1.24+ 推荐 import pymupdf


@dataclass
class PdfPage:
    """单页 PDF 内容。"""

    page_num: int  # 从 1 开始
    text: str
    is_likely_cover_or_toc: bool  # 是否疑似封面/目录等低密度页


@dataclass
class PdfDocument:
    """整份 PDF 解析结果。"""

    file_path: Path
    file_name: str
    page_count: int
    pages: list[PdfPage]

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


# 简单的"低密度页"识别：正文页字符数 > 300
MIN_CONTENT_CHARS = 300


def _is_low_density_page(text: str) -> bool:
    """简单判断是否为低密度页（封面、目录、签字页）。

    判断标准：去除空白后字符数 < 300，或包含大量"目录"标记。
    """
    clean = re.sub(r"\s+", "", text)
    if len(clean) < MIN_CONTENT_CHARS:
        return True
    # 含明显的目录标记
    if "目  录" in text or re.search(r"\.{5,}\s*\d+\s*$", text, re.MULTILINE):
        return True
    return False


def extract_pdf(pdf_path: Path) -> PdfDocument:
    """提取一份 PDF 的所有页面文本。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        PdfDocument，含每页文本和元数据
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages: list[PdfPage] = []

    for i, page in enumerate(doc):
        # text 模式提取纯文本（不含图片）
        text = page.get_text("text")
        text = text.strip()
        if not text:
            continue  # 跳过纯图片页

        pages.append(
            PdfPage(
                page_num=i + 1,
                text=text,
                is_likely_cover_or_toc=_is_low_density_page(text),
            )
        )

    doc.close()
    return PdfDocument(
        file_path=pdf_path,
        file_name=pdf_path.name,
        page_count=len(pages),
        pages=pages,
    )


def extract_all_pdfs(reports_dir: Path) -> list[PdfDocument]:
    """批量解析目录下所有 PDF。

    Args:
        reports_dir: PDF 目录（如 data/reports/）

    Returns:
        PdfDocument 列表，按文件名排序
    """
    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        raise FileNotFoundError(f"目录不存在: {reports_dir}")

    pdf_files = sorted(reports_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"目录下没有 PDF: {reports_dir}")

    docs: list[PdfDocument] = []
    for pdf_path in pdf_files:
        try:
            doc = extract_pdf(pdf_path)
            docs.append(doc)
        except Exception as e:
            print(f"[error] 解析失败 {pdf_path.name}: {e}")
    return docs


# 便利函数：从文件名解析公司代码和公司名（约定格式 {code}_{name}_{year}.pdf）
_FILE_NAME_PATTERN = re.compile(r"^(\d{6})_(.+?)_(\d{4})\.pdf$")


def parse_meta_from_filename(file_name: str) -> tuple[str, str, int] | None:
    """从文件名解析 (stock_code, company, year)。"""
    m = _FILE_NAME_PATTERN.match(file_name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


if __name__ == "__main__":
    # 自测：
    #   python -m src.rag.pdf_loader <pdf_path>          # 单文件解析
    #   python -m src.rag.pdf_loader <dir>                # 批量统计
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m src.rag.pdf_loader <pdf_path|dir>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        # 批量统计
        docs = extract_all_pdfs(target)
        total_pages = sum(d.page_count for d in docs)
        total_chars = sum(d.total_chars for d in docs)
        total_low = sum(
            1 for d in docs for p in d.pages if p.is_likely_cover_or_toc
        )
        print(f"=== 批量解析 {len(docs)} 份 PDF ===")
        print(f"{'文件名':<40} {'页数':>6} {'字符':>10} {'低密度':>7}")
        print("-" * 70)
        for d in docs:
            low = sum(1 for p in d.pages if p.is_likely_cover_or_toc)
            print(f"{d.file_name:<40} {d.page_count:>6} {d.total_chars:>10,} {low:>7}")
        print("-" * 70)
        print(f"{'合计':<40} {total_pages:>6} {total_chars:>10,} {total_low:>7}")
        print(f"\n预估 chunk 数（800字/段，重叠100）: ~{total_chars // 700:,}")
    else:
        # 单文件解析预览
        doc = extract_pdf(target)
        print(f"文件: {doc.file_name}")
        print(f"总页数: {doc.page_count}")
        print(f"总字符数: {doc.total_chars:,}")
        low = sum(1 for p in doc.pages if p.is_likely_cover_or_toc)
        print(f"低密度页（封面/目录等）: {low} 页")
        if len(doc.pages) >= 10:
            print(f"\n第 10 页预览（前 300 字符）:")
            print(doc.pages[9].text[:300])
