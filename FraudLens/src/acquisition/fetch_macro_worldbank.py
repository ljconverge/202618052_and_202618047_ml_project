"""Step 3: download World Bank indicators for the configured country.

Run from the project root:
    python src/acquisition/fetch_macro_worldbank.py

Output:
    data/raw/macro/worldbank_macro_raw.csv   one row per year, one column per indicator
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import ROOT, load_config  # noqa: E402

CFG = load_config()
OUT = ROOT / "data" / "raw" / "macro"
API = "https://api.worldbank.org/v2/country/{country}/indicator/{code}"


def fetch_indicator(country, code, start, end, retries=3):
    params = {"format": "json", "date": f"{start}:{end}", "per_page": 1000}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(API.format(country=country, code=code), params=params, timeout=90)
            resp.raise_for_status()
            payload = resp.json()          # payload[0] = metadata, payload[1] = records
            if len(payload) < 2 or not payload[1]:
                return pd.DataFrame(columns=["year", "value"])
            return pd.DataFrame([{"year": int(r["date"]), "value": r["value"]} for r in payload[1]])
        except requests.RequestException as err:
            print(f"  attempt {attempt}/{retries} failed for {code}: {err}")
            time.sleep(3 * attempt)
    return pd.DataFrame(columns=["year", "value"])


def main():
    wb = CFG["world_bank"]
    OUT.mkdir(parents=True, exist_ok=True)
    wide = None
    for code, name in wb["indicators"].items():
        print(f"Fetching {name} ({code}) for {wb['country']}")
        df = fetch_indicator(wb["country"], code, wb["start_year"], wb["end_year"])
        df = df.rename(columns={"value": name})
        wide = df if wide is None else wide.merge(df, on="year", how="outer")
    wide.insert(0, "country", wb["country"])
    wide = wide.sort_values("year").reset_index(drop=True)
    out = OUT / "worldbank_macro_raw.csv"
    wide.to_csv(out, index=False)
    print(f"Saved {len(wide)} rows -> {out}")


if __name__ == "__main__":
    main()