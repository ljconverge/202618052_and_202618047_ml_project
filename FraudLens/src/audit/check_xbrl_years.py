"""Check which fraud candidates have XBRL facts for the years around their enforcement case.

Every keep row is tested, because one company can have several cases. A company is usable
if at least one of its cases is covered by its XBRL data: the file must reach the year
before the case and the year of the case.

Run from the project root, after fetch_facts_for_list.py:
    python src/audit/check_xbrl_years.py

Input:   reports/tables/gate1_review.csv   (rows with decision = keep)
         data/raw/sec_edgar/facts/CIK##########.json
Output:  reports/tables/xbrl_coverage.csv          one row per case
         reports/tables/xbrl_coverage_companies.csv one row per company
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

REVIEW = ROOT / "reports" / "tables" / "gate1_review.csv"
FACTS = ROOT / "data" / "raw" / "sec_edgar" / "facts"
OUT_CASES = ROOT / "reports" / "tables" / "xbrl_coverage.csv"
OUT_COMPANIES = ROOT / "reports" / "tables" / "xbrl_coverage_companies.csv"


def fiscal_years(path):
    """Return the sorted set of fiscal years that appear in any reported fact."""
    data = json.loads(path.read_text(encoding="utf-8"))
    years = set()
    for taxonomy in data.get("facts", {}).values():
        for concept in taxonomy.values():
            for unit_rows in concept.get("units", {}).values():
                for row in unit_rows:
                    if isinstance(row.get("fy"), int):
                        years.add(row["fy"])
    return sorted(years)


def enforcement_year(text):
    match = re.search(r"(19|20)\d{2}", str(text))
    return int(match.group(0)) if match else None


def main():
    review = pd.read_csv(REVIEW, dtype={"decision": str})
    kept = review[review["decision"].fillna("").str.lower() == "keep"].copy()
    if kept.empty:
        raise SystemExit("No rows marked keep in gate1_review.csv. Fill or merge the decisions first "
                         "(merge_decisions.py, or review the rows by hand).")
    kept["enforcement_year"] = kept["aaer_text"].map(enforcement_year)
    kept = kept.drop_duplicates(subset=["cik", "aaer_text"])

    cache = {}
    rows = []
    for c in kept.itertuples():
        cik = int(c.cik)
        if cik not in cache:
            path = FACTS / f"CIK{cik:010d}.json"
            cache[cik] = fiscal_years(path) if path.exists() else []
        years, ey = cache[cik], c.enforcement_year
        usable = bool(years) and ey is not None and min(years) <= ey - 1 and max(years) >= ey
        rows.append({"cik": cik, "ticker": c.ticker, "company_name": c.company_name,
                     "case": c.aaer_text, "enforcement_year": ey,
                     "xbrl_years": f"{min(years)}-{max(years)}" if years else "no file",
                     "usable_case": usable})

    cases = pd.DataFrame(rows)
    companies = (cases.groupby(["cik", "ticker", "company_name"], as_index=False)
                 .agg(cases=("case", "count"), usable_cases=("usable_case", "sum"),
                      xbrl_years=("xbrl_years", "first")))
    companies["usable"] = companies["usable_cases"] > 0

    OUT_CASES.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUT_CASES, index=False)
    companies.to_csv(OUT_COMPANIES, index=False)
    print(companies.to_string(index=False))
    print(f"\nUsable companies: {int(companies['usable'].sum())} of {len(companies)}")
    print(f"Saved -> {OUT_CASES}")
    print(f"Saved -> {OUT_COMPANIES}")


if __name__ == "__main__":
    main()