"""Build the final modelling sample.

Fraud group:   companies marked usable in xbrl_coverage_companies.csv (label_group = fraud).
Control group: random companies from the sample pool that are NOT AAER keep companies and
               that have an XBRL facts file (label_group = control).
Number of controls per fraud company comes from config/config.yaml (sample.controls_per_fraud).

Run from the project root, after check_xbrl_years.py:
    python src/audit/build_sample.py
Output: data/raw/sec_edgar/company_list_final.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT, load_config  # noqa: E402

CFG = load_config()["sample"]
SAMPLE = ROOT / "data" / "raw" / "sec_edgar" / "company_list_sample.csv"
FACTS = ROOT / "data" / "raw" / "sec_edgar" / "facts"
REVIEW = ROOT / "reports" / "tables" / "gate1_review.csv"
COVERAGE = ROOT / "reports" / "tables" / "xbrl_coverage_companies.csv"
OUT = ROOT / "data" / "raw" / "sec_edgar" / "company_list_final.csv"


def has_facts(cik):
    return (FACTS / f"CIK{int(cik):010d}.json").exists()


def main():
    for path in (SAMPLE, REVIEW, COVERAGE):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run the earlier steps first.")

    sample = pd.read_csv(SAMPLE)
    review = pd.read_csv(REVIEW, dtype={"decision": str})
    coverage = pd.read_csv(COVERAGE)

    usable = coverage[coverage["usable"]][["cik", "ticker", "company_name"]].copy()
    if usable.empty:
        raise SystemExit("No usable fraud companies. Check xbrl_coverage_companies.csv before sampling.")
    usable["label_group"] = "fraud"

    keep_ciks = set(review.loc[review["decision"].fillna("").str.lower() == "keep", "cik"])
    pool = sample[~sample["cik"].isin(keep_ciks) & sample["cik"].apply(has_facts)]
    n_controls = min(len(pool), CFG["controls_per_fraud"] * len(usable))
    controls = pool.sample(n=n_controls, random_state=CFG["seed"]).copy()
    controls["label_group"] = "control"

    final = pd.concat([usable, controls], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT, index=False)
    print(f"Pool with XBRL data: {len(pool)} | fraud: {len(usable)} | controls: {len(controls)} | total: {len(final)}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()