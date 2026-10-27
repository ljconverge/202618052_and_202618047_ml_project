"""
Day 1 audit: scan every CSV under data/ and write a markdown report.
"""
import os
import pandas as pd

DATA_DIR = "data"
REPORT = os.path.join("reports", "tables", "day1_data_audit.md")
ID_CANDIDATES = ["cik", "ticker", "company_name", "company"]   # the "entity"
LABEL_CANDIDATES = ["fraud_label", "label", "is_fraud"]


def find_csvs(root):
    paths = []
    for folder, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".csv"):
                paths.append(os.path.join(folder, name))
    return sorted(paths)


def load_or_none(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:      # the empty-file bug you hit with Scrapy
        return None


def label_lines(df):
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    id_col = next((c for c in ID_CANDIDATES if c in df.columns), None)
    if label_col is None:
        return []
    labels = df[label_col]
    if pd.api.types.is_numeric_dtype(labels):
        positive = labels == 1
    else:                                  # text labels such as "Fraudulent (Minority)"
        s = labels.astype(str).str.lower()
        positive = s.str.contains("fraud") & ~s.str.contains("not|non|no ")
    out = [f"- label `{label_col}`: {int(positive.sum())} positive of {len(df)} rows "
           f"({positive.mean() * 100:.4f}%)"]
    if id_col:
        out.append(f"- distinct `{id_col}`: {df[id_col].nunique()} overall, "
                   f"**{df.loc[positive, id_col].nunique()} among positive rows**  <- GATE 1")
    return out


def column_table(df, max_cols=25):
    rows = ["| column | dtype | % missing | distinct |", "|---|---|---|---|"]
    for col in list(df.columns)[:max_cols]:
        rows.append(f"| {col} | {df[col].dtype} | {df[col].isna().mean() * 100:.1f} "
                    f"| {df[col].nunique()} |")
    return rows


def main():
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    lines = ["# Day 1 data audit", "",
             "| file | rows | cols | size KB | status |", "|---|---|---|---|---|"]
    details = []
    for path in find_csvs(DATA_DIR):
        df = load_or_none(path)
        size_kb = os.path.getsize(path) / 1024
        name = path.replace("\\", "/")
        if df is None:
            lines.append(f"| {name} | 0 | 0 | {size_kb:.1f} | EMPTY FILE |")
            continue
        lines.append(f"| {name} | {len(df)} | {df.shape[1]} | {size_kb:.1f} | ok |")
        details += ["", f"## {name}", ""] + label_lines(df) + [""] + column_table(df)
    text = "\n".join(lines + details)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved -> {REPORT}")


if __name__ == "__main__":
    main()
