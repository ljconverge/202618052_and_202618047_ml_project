"""Report news coverage for fraud and control companies, using crawl outputs only."""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT  # noqa: E402

NEWS = ROOT / "data" / "raw" / "news"
FINAL = ROOT / "data" / "raw" / "sec_edgar" / "company_list_final.csv"
OUT = ROOT / "reports" / "tables" / "news_coverage.csv"


def load_outputs():
    paths = list((NEWS / "batches").glob("batch_*_out.csv")) + [NEWS / "news_headlines_raw.csv"]
    frames = []
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if "headline" in df.columns:
            frames.append(df[df["headline"].notna()])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["company_name"])


def main():
    news = load_outputs()
    final = pd.read_csv(FINAL)
    counts = news["company_name"].value_counts()
    final["articles"] = final["company_name"].map(counts).fillna(0).astype(int)
    final["has_news"] = final["articles"] > 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT, index=False)
    print(final.groupby("label_group")["has_news"].agg(["sum", "count"]))
    missing = final[~final["has_news"]]
    print(f"\nCompanies with no headlines: {len(missing)}")
    print(missing[["label_group", "ticker", "company_name"]].to_string(index=False))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()