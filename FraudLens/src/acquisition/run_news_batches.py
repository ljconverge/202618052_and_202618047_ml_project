"""Crawl news for every company in company_list_final.csv that has no headlines yet.

  1. A company counts as done if any saved news file has at least one headline for it.
  2. Remaining companies run in batches (news.batch_size), one request at a time.
  3. After each batch, if few companies returned headlines (GDELT rate-limiting), the script
     waits longer before the next batch.
  4. data/raw/news/news_headlines_combined.csv is rebuilt after every batch.

Resumable: re-running only crawls companies still missing headlines.

Run from the project root:
    python src/acquisition/run_news_batches.py
"""
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT, load_config  # noqa: E402

CFG = load_config()["news"]
COMPANY_LIST = ROOT / "data" / "raw" / "sec_edgar" / "company_list_final.csv"
NEWS_DIR = ROOT / "data" / "raw" / "news"
BATCH_DIR = NEWS_DIR / "batches"
COMBINED = NEWS_DIR / "news_headlines_combined.csv"
SCRAPY_DIR = ROOT / "src" / "acquisition" / "news_scraper"


def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def saved_news_files():
    """Only crawl outputs: batch_*_out.csv and the raw news file. Never the company lists."""
    files = list(BATCH_DIR.glob("batch_*_out.csv")) if BATCH_DIR.exists() else []
    raw = NEWS_DIR / "news_headlines_raw.csv"
    if raw.exists():
        files.append(raw)
    return files


def completed_companies():
    """Companies the spider answered successfully, including those with zero articles."""
    names = set()
    status_dir = NEWS_DIR / "status"
    if status_dir.exists():
        for path in status_dir.glob("*_status.csv"):
            df = read_csv_safe(path)
            if not df.empty and {"company_name", "status"}.issubset(df.columns):
                names.update(df.loc[df["status"] == "ok", "company_name"])
    return names


def companies_with_news():
    names = set()
    for path in saved_news_files():
        df = read_csv_safe(path)
        if not df.empty and {"company_name", "headline"}.issubset(df.columns):
            names.update(df.loc[df["headline"].notna(), "company_name"].dropna().unique())
    return names


def remaining_companies():
    if not COMPANY_LIST.exists():
        raise SystemExit(f"Missing {COMPANY_LIST}. Run build_sample.py first.")
    companies = read_csv_safe(COMPANY_LIST)
    if "company_name" not in companies.columns:
        raise SystemExit(f"{COMPANY_LIST} is empty or has no company_name column. Re-run build_sample.py.")
    done = companies_with_news() | completed_companies()
    return companies[~companies["company_name"].isin(done)].reset_index(drop=True)


def run_batch(batch_no, batch):
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    list_file = BATCH_DIR / f"batch_{batch_no:02d}_companies.csv"
    out_file = BATCH_DIR / f"batch_{batch_no:02d}_out.csv"
    batch.to_csv(list_file, index=False)
    cmd = [sys.executable, "-m", "scrapy", "crawl", "news_spider",
           "-a", f"companies={list_file}", "-o", str(out_file)]
    code = subprocess.run(cmd, cwd=SCRAPY_DIR).returncode
    out = read_csv_safe(out_file)
    got = out["company_name"].nunique() if not out.empty else 0
    return code, got


def rebuild_combined():
    frames = [read_csv_safe(p) for p in saved_news_files()]
    frames = [f for f in frames if not f.empty and "headline" in f.columns]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["headline"].notna()]
    combined = combined.drop_duplicates(subset=["company_name", "link"])
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED, index=False)
    return combined


def main():
    todo = remaining_companies()
    if todo.empty:
        rebuild_combined()
        print("Every company already has news. Nothing to crawl.")
        return
    size = CFG["batch_size"]
    batches = [todo.iloc[i:i + size] for i in range(0, len(todo), size)]
    est_min = len(todo) * CFG["request_delay_seconds"] / 60
    print(f"{len(todo)} companies need news, in {len(batches)} batches of up to {size}")
    print(f"Estimated crawl time without rate limits: about {est_min:.0f} minutes")

    for batch_no, batch in enumerate(batches, start=1):
        print(f"Batch {batch_no}/{len(batches)}: {len(batch)} companies")
        code, got = run_batch(batch_no, batch)
        combined = rebuild_combined()
        share = got / len(batch)
        print(f"  headlines returned for {got}/{len(batch)} companies | total headlines: {len(combined)}")
        if code != 0:
            print(f"  scrapy exited with code {code}. Re-run this script to continue.")
            break
        if batch_no < len(batches):
            limited = share < CFG["min_batch_yield"]
            pause = CFG["pause_rate_limited_seconds"] if limited else CFG["pause_normal_seconds"]
            print(f"  pausing {pause // 60} min" + (" (GDELT rate-limiting: few results)" if limited else ""))
            time.sleep(pause)

    combined = rebuild_combined()
    print(f"Finished. Headlines: {len(combined)} | companies with news: {combined['company_name'].nunique()}")


if __name__ == "__main__":
    main()