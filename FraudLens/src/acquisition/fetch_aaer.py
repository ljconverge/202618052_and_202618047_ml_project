"""Step 2: download every page of the SEC AAER index (raw HTML) and extract the table rows.

The index is paginated with a ?page=N parameter. The site also serves the same pages with
extra sort options (&order=...&sort=...). Those are duplicates, so every URL is reduced to
its page number before it is visited. Pages are numbered in the order they were found.

Run from the project root:
    python src/acquisition/fetch_aaer.py

Outputs:
    data/raw/aaer/aaer_page_<n>.html    each page exactly as downloaded
    data/raw/aaer/aaer_rows_raw.csv     every unique table row, with the page it came from
"""
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT, load_config  # noqa: E402

CFG = load_config()
START_URL = "https://www.sec.gov/enforcement-litigation/accounting-auditing-enforcement-releases"
OUT = ROOT / "data" / "raw" / "aaer"
MAX_PAGES = 100          # safety limit so a bad link cannot loop forever


def canonical(url):
    """Keep only the scheme, host, path and the page number. Drop sort/order options."""
    parts = urlparse(url)
    page = parse_qs(parts.query).get("page", ["0"])[0]
    query = urlencode({"page": page}) if page != "0" else ""
    return urlunparse((parts.scheme, parts.netloc, parts.path, "", query, ""))


def extract_rows(html, page_no):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for t_idx, table in enumerate(soup.find_all("table")):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            link = tr.find("a", href=True)
            rows.append({
                "page_no": page_no,
                "table_no": t_idx,
                "cells": " | ".join(c.get_text(" ", strip=True) for c in cells),
                "link": urljoin(START_URL, link["href"]) if link else "",
            })
    return rows


def find_page_links(html, base_url):
    base_path = urlparse(START_URL).path
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"])
        parts = urlparse(full)
        if parts.path == base_path and "page" in parse_qs(parts.query):
            links.add(canonical(full))
    return links


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": CFG["sec"]["user_agent"]}
    queue = [canonical(START_URL)]
    visited, all_rows = [], []

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        visited.append(url)
        page_no = len(visited)
        (OUT / f"aaer_page_{page_no}.html").write_text(resp.text, encoding="utf-8")
        all_rows += extract_rows(resp.text, page_no)
        for link in sorted(find_page_links(resp.text, url)):
            if link not in visited and link not in queue:
                queue.append(link)
        print(f"page {page_no}: {url}")
        time.sleep(1)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["cells", "link"])
    df.to_csv(OUT / "aaer_rows_raw.csv", index=False)
    print(f"Pages downloaded: {len(visited)} | unique table rows saved: {len(df)}")
    if len(visited) >= MAX_PAGES:
        print("WARNING: stopped at the page limit. Check the page links before trusting the result.")
    if df.empty:
        print("No table rows found. Open data\\raw\\aaer\\aaer_page_1.html in a browser and report the layout.")


if __name__ == "__main__":
    main()