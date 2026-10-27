"""Download raw XBRL company facts for every CIK in a CSV file (resumable).

Run from the project root:
    python src/acquisition/fetch_facts_for_list.py data/raw/sec_edgar/company_list_final.csv

Output: data/raw/sec_edgar/facts/CIK##########.json (skips files already present)
Missing companies are listed at the end so you can decide what to do with them.
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
FACTS = ROOT / "data" / "raw" / "sec_edgar" / "facts"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def download(cik, retries=3):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(FACTS_URL.format(cik=cik), headers=HEADERS, timeout=60)
            if resp.status_code == 404:
                return None                      # company has no XBRL facts
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as err:
            print(f"  attempt {attempt}/{retries} failed for CIK {cik}: {err}")
            time.sleep(3 * attempt)
    return None


def main():
    csv_path = Path(sys.argv[1])
    companies = pd.read_csv(csv_path)
    FACTS.mkdir(parents=True, exist_ok=True)
    missing = []
    for i, row in enumerate(companies.itertuples(), start=1):
        path = FACTS / f"CIK{int(row.cik):010d}.json"
        if path.exists():
            continue
        text = download(int(row.cik))
        if text is None:
            missing.append(row.ticker)
            print(f"[{i}/{len(companies)}] no facts: {row.ticker}")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"[{i}/{len(companies)}] saved {row.ticker}")
        time.sleep(0.2)
    print(f"Done. Companies without XBRL facts: {len(missing)} {missing}")


if __name__ == "__main__":
    main()
