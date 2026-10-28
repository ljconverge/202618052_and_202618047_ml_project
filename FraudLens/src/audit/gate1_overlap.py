"""Day 1 Gate 1: how many of our sampled companies appear in the SEC AAER enforcement list?

Run from the project root (after fetch_sec_edgar.py and fetch_aaer.py).
Searches ALL SEC registrants, not only the sample.
    python src/audit/gate1_overlap.py

Output:
    reports/tables/gate1_overlap.csv   one row per sampled company that matched an AAER row
Printed: the matched count. Fraud labels need this overlap to be large enough to learn from.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

SAMPLE = ROOT / "data" / "raw" / "sec_edgar" / "company_list_all.csv"   # search all SEC registrants
AAER = ROOT / "data" / "raw" / "aaer" / "aaer_rows_raw.csv"
OUT = ROOT / "reports" / "tables" / "gate1_overlap.csv"
MIN_NAME_LEN = 8   # very short normalised names match too easily, so skip them


def normalise(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def main():
    sample = pd.read_csv(SAMPLE)
    aaer = pd.read_csv(AAER)
    aaer_text = [normalise(t) for t in aaer["cells"].fillna("")]

    matches = []
    for row in sample.itertuples():
        key = normalise(row.company_name)
        if len(key) < MIN_NAME_LEN:
            continue
        hits = [t for t in aaer_text if key in t]
        if hits:
            matches.append({"cik": row.cik, "ticker": row.ticker, "company_name": row.company_name,
                            "aaer_rows_matched": len(hits)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(matches)
    result.to_csv(OUT, index=False)
    print(f"Sampled companies: {len(sample)}")
    print(f"AAER rows read:    {len(aaer)}")
    print(f"Sampled companies matched to AAER: {len(result)}")
    print("Name matching is approximate: check every match by hand before using it as a label.")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()