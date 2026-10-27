"""Day 1 review sheet: list the AAER rows behind each Gate 1 match, for a hand check.

Run from the project root, after gate1_overlap.py:
    python src/audit/show_matches.py

Output: reports/tables/gate1_review.csv
    One row per (company, AAER row) match, with an empty 'decision' column.
    Fill decision with keep or drop for each row. Keep the same columns.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

OVERLAP = ROOT / "reports" / "tables" / "gate1_overlap.csv"
AAER = ROOT / "data" / "raw" / "aaer" / "aaer_rows_raw.csv"
OUT = ROOT / "reports" / "tables" / "gate1_review.csv"


def normalise(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def main():
    overlap = pd.read_csv(OVERLAP)
    aaer = pd.read_csv(AAER)
    aaer["norm"] = aaer["cells"].fillna("").map(normalise)

    rows = []
    for c in overlap.itertuples():
        key = normalise(c.company_name)
        for a in aaer[aaer["norm"].str.contains(key, regex=False)].itertuples():
            rows.append({
                "cik": c.cik,
                "ticker": c.ticker,
                "company_name": c.company_name,
                "aaer_text": a.cells,
                "aaer_link": a.link if isinstance(a.link, str) else "",
                "decision": "",
            })
    review = pd.DataFrame(rows)
    review.to_csv(OUT, index=False)
    print(f"Review rows: {len(review)} covering {review['cik'].nunique() if len(review) else 0} companies")
    print(f"Saved -> {OUT}")
    print("Fill the decision column with keep or drop, then save the file.")


if __name__ == "__main__":
    main()
