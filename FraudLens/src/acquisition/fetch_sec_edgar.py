"""
Step 1: download SEC company list and XBRL company facts (raw JSON, unchanged).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT, load_config  # noqa: E402

CFG = load_config()
HEADERS = {"User-Agent": CFG["sec"]["user_agent"]}
OUT = ROOT / "data" / "raw" / "sec_edgar"
FACTS = OUT / "facts"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def get_with_retry(url, retries=3):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException as err:
            print(f"  attempt {attempt}/{retries} failed: {err}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FACTS.mkdir(parents=True, exist_ok=True)

    tickers = get_with_retry(TICKERS_URL).json()
    companies = pd.DataFrame(
        [{"cik": int(v["cik_str"]), "ticker": v["ticker"], "company_name": v["title"]}
         for v in tickers.values()]
    )
    companies.to_csv(OUT / "company_list_all.csv", index=False)
    sample = companies.head(CFG["sec"]["companies_to_sample"])
    sample.to_csv(OUT / "company_list_sample.csv", index=False)
    print(f"SEC registrants: {len(companies)} | sampled: {len(sample)}")

    for i, row in enumerate(sample.itertuples(), start=1):
        path = FACTS / f"CIK{row.cik:010d}.json"
        if path.exists():           # resumable: skip files already downloaded
            continue
        try:
            resp = get_with_retry(FACTS_URL.format(cik=row.cik))
            path.write_text(resp.text, encoding="utf-8")
            print(f"[{i}/{len(sample)}] saved {row.ticker}")
        except requests.RequestException as err:
            print(f"[{i}/{len(sample)}] FAILED {row.ticker}: {err}")
        time.sleep(0.2)             # stay well under SEC's rate limit

    saved = len(list(FACTS.glob("CIK*.json")))
    print(f"Done. XBRL files on disk: {saved}")


if __name__ == "__main__":
    main()
