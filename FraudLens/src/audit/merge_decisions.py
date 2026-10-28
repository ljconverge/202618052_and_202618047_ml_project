"""Carry manual keep/drop decisions from an older review file into the current gate1_review.csv.

Matching key: cik + aaer_text. Only rows that already have a decision are copied.
Rows without a decision are left blank for you to review.

Run from the project root:
    python src/audit/merge_decisions.py                              (uses reports/gate1_review_backup.csv)
    python src/audit/merge_decisions.py path/to/old_review.csv       (any other older file)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

NEW = ROOT / "reports" / "tables" / "gate1_review.csv"
DEFAULT_OLD = ROOT / "reports" / "gate1_review_backup.csv"


def main():
    old_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OLD
    if not old_path.exists():
        raise SystemExit(f"Old review file not found: {old_path}. Pass its path as an argument.")
    new = pd.read_csv(NEW, dtype=str).fillna("")
    old = pd.read_csv(old_path, dtype=str).fillna("")
    columns = list(new.columns)

    decided = old[old["decision"].str.strip() != ""][["cik", "aaer_text", "decision"]]
    merged = new.drop(columns=["decision"]).merge(decided, on=["cik", "aaer_text"], how="left")
    merged["decision"] = merged["decision"].fillna("")
    merged = merged[columns]
    merged.to_csv(NEW, index=False)

    carried = int((merged["decision"] != "").sum())
    print(f"Old decisions found: {len(decided)} | carried over: {carried} | still blank: {len(merged) - carried}")
    print(f"Saved -> {NEW}")
    if carried < len(decided):
        print("Some old decisions did not match. Their aaer_text changed, so review those rows by hand.")


if __name__ == "__main__":
    main()