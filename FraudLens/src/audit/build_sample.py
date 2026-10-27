"""Build the final modelling sample.

Fraud group:   companies marked usable in xbrl_coverage_companies.csv (label_group = fraud).
Control group: random companies from the 250-company sample that are not an AAER keep company
               (label_group = control). Their XBRL files already exist.

Run from the project root, after check_xbrl_years.py:
    python src/audit/build_sample.py
Then download the XBRL facts for the final list:
    python src/acquisition/fetch_facts_for_list.py data/raw/sec_edgar/company_list_final.csv

Output: data/raw/sec_edgar/company_list_final.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

SAMPLE = ROOT / "data" / "raw" / "sec_edgar" / "company_list_sample.csv"
REVIEW = ROOT / "reports" / "tables" / "gate1_review.csv"
COVERAGE = ROOT / "reports" / "tables" / "xbrl_coverage_companies.csv"
OUT = ROOT / "data" / "raw" / "sec_edgar" / "company_list_final.csv"
CONTROLS_PER_FRAUD = 3
SEED = 42


def main():
    sample = pd.read_csv(SAMPLE)
    review = pd.read_csv(REVIEW, dtype={"decision": str})
    coverage = pd.read_csv(COVERAGE)

    usable = coverage[coverage["usable"]][["cik", "ticker", "company_name"]].copy()
    if usable.empty:
        raise SystemExit("No usable companies in xbrl_coverage_companies.csv. Run check_xbrl_years.py first.")
    usable["label_group"] = "fraud"

    keep_ciks = set(review.loc[review["decision"].astype(str).str.lower() == "keep", "cik"])
    pool = sample[~sample["cik"].isin(keep_ciks)]
    n_controls = min(len(pool), CONTROLS_PER_FRAUD * len(usable))
    controls = pool.sample(n=n_controls, random_state=SEED).copy()
    controls["label_group"] = "control"

    final = pd.concat([usable, controls], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT, index=False)
    print(f"Fraud companies: {len(usable)} | controls: {len(controls)} | total: {len(final)}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()