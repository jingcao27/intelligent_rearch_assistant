"""从巨潮资讯网（cninfo.com.cn）下载 A 股上市公司年度报告 PDF。

巨潮是深交所官方信息披露平台，所有 A 股年报都在这里。
本模块通过其公开的 disclosure API 查询年报并下载 PDF。

用法：
    python -m src.rag.download             # 下载 10 家公司 2024 年报
    python -m src.rag.download --year 2023 # 下载 2023 年报
    python -m src.rag.download --stock 600519  # 只下载指定股票
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# 巨潮 API
URL_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
URL_PDF_BASE = "http://static.cninfo.com.cn/"

# 10 家代表股（覆盖白酒/电池/新能源/保险/银行/石油/石化/医药/白酒/中药）
# 用户可自行修改或扩展
COMPANIES: list[tuple[str, str]] = [
    ("600519", "贵州茅台"),
    ("300750", "宁德时代"),
    ("002594", "比亚迪"),
    ("601318", "中国平安"),
    ("600036", "招商银行"),
    ("601398", "工商银行"),
    ("601857", "中国石油"),
    ("600028", "中国石化"),
    ("000858", "五粮液"),
    ("600276", "恒瑞医药"),
]

# 简洁请求头，模拟浏览器避免反爬
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 下载超时（秒）
DOWNLOAD_TIMEOUT = 120.0
# 每次请求间隔（秒），避免触发反爬
REQUEST_DELAY = 1.5


def _get_exchange(stock_code: str) -> str:
    """根据股票代码前缀判断交易所。

    - 6 开头：上交所（sse）
    - 0、3 开头：深交所（szse）
    """
    if stock_code.startswith("6"):
        return "sse"
    return "szse"


@dataclass
class Announcement:
    """巨潮公告条目。"""

    title: str
    url: str
    sec_name: str
    announcement_time: str


def query_annual_report(stock_code: str, company: str, year: int) -> list[Announcement]:
    """查询某公司某年度的年度报告。

    年报披露在次年 1-4 月，所以查询年份 = year + 1。

    巨潮的 secName 可能是汉字间带空格的格式（如"五 粮 液"），
    所以查不到时自动 fallback 到每字加空格的版本重试。
    """
    exchange = _get_exchange(stock_code)
    # 年报披露窗口：次年 1 月 1 日 - 4 月 30 日（年报披露截止日）
    start_date = f"{year + 1}-01-01"
    end_date = f"{year + 1}-12-31"

    def _do_query(searchkey: str) -> list[Announcement]:
        data = {
            "searchkey": searchkey,
            "tabName": "fulltext",
            "pageSize": "30",
            "pageNum": "1",
            "category": "category_ndbg_szsh",  # 年度报告类别
            "seDate": f"{start_date}~{end_date}",
        }
        resp = httpx.post(URL_QUERY, data=data, headers=HEADERS, timeout=20.0)
        resp.raise_for_status()
        result = resp.json()

        announcements: list[Announcement] = []
        for ann in result.get("announcements") or []:
            title = ann.get("announcementTitle", "")
            # 只要"年度报告"，跳过"摘要"、"修订版"、"英文版"等
            if "年度报告" not in title:
                continue
            if any(skip in title for skip in ("摘要", "修订", "英文", "取消", "补充")):
                continue
            adjunct_url = ann.get("adjunctUrl", "")
            if not adjunct_url:
                continue
            announcements.append(
                Announcement(
                    title=title,
                    url=URL_PDF_BASE + adjunct_url,
                    sec_name=ann.get("secName", ""),
                    announcement_time=ann.get("announcementTime", ""),
                )
            )
        return announcements

    # 第一次：原样查询
    announcements = _do_query(company)
    if announcements:
        return announcements

    # fallback：每字加空格（针对 secName 是 "五 粮 液" 的情况）
    spaced_name = " ".join(company)
    if spaced_name != company:
        announcements = _do_query(spaced_name)
        if announcements:
            logger.debug("[fallback] %s 用空格分隔名 %r 查到年报", stock_code, spaced_name)
    return announcements


def download_pdf(url: str, save_path: Path) -> int:
    """下载 PDF 到 save_path，返回字节数。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": HEADERS["User-Agent"]}
    total = 0
    with httpx.stream(
        "GET", url, headers=headers, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
    ) as resp:
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)
    return total


def download_all(reports_dir: Path, year: int = 2024) -> dict[str, str]:
    """下载 10 家公司的年报到 reports_dir。

    Returns:
        成功下载的映射：{stock_code: pdf_path}
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    success: dict[str, str] = {}

    for stock_code, company in COMPANIES:
        save_name = f"{stock_code}_{company}_{year}.pdf"
        save_path = reports_dir / save_name

        # 已下载则跳过（断点续传）
        if save_path.exists() and save_path.stat().st_size > 100_000:
            logger.info("[skip] %s %s 已存在 (%d KB)", stock_code, company,
                        save_path.stat().st_size // 1024)
            success[stock_code] = str(save_path)
            continue

        try:
            anns = query_annual_report(stock_code, company, year)
            if not anns:
                logger.warning("[fail] 未找到 %s(%s) %d 年年度报告",
                               stock_code, company, year)
                continue

            # 按公告时间倒序，取最新一份（巨潮默认就是时间倒序）
            ann = anns[0]
            logger.info("[download] %s %s ← %s", stock_code, company, ann.title)
            bytes_written = download_pdf(ann.url, save_path)
            logger.info("  └─ 保存到 %s (%d KB)", save_path.name, bytes_written // 1024)
            success[stock_code] = str(save_path)

            # 限速
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.error("[error] %s %s: %s", stock_code, company, e)

    logger.info("=== 完成：成功 %d / 总计 %d ===", len(success), len(COMPANIES))
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 A 股年度报告")
    parser.add_argument("--year", type=int, default=2024, help="年报年份（默认 2024）")
    parser.add_argument("--stock", type=str, default="", help="只下载指定股票代码")
    parser.add_argument("--out", type=str, default="data/reports", help="输出目录")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out)
    if args.stock:
        # 单股票模式（用于测试）
        global COMPANIES
        COMPANIES = [(args.stock, args.stock)]
    download_all(out_dir, year=args.year)


if __name__ == "__main__":
    main()
