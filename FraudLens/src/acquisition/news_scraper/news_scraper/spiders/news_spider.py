"""Raw news headlines per company from the GDELT DOC 2.0 API (uncleaned).

One request per company, one request at a time. Each company's outcome is recorded:
  ok      = GDELT answered (headlines may be zero, which is a valid "no matching news" result)
  failed  = no usable answer (HTTP 429 after retries, timeout, network error)
Outcomes are written to data/raw/news/status/<input list name>_status.csv.

Run from src/acquisition/news_scraper (normally called by run_news_batches.py):
    scrapy crawl news_spider -a companies=<csv> -o <output.csv>
"""
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import scrapy
import yaml

from news_scraper.items import NewsHeadlineItem

# spiders -> news_scraper -> news_scraper -> acquisition -> src -> project root
ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LIST = ROOT / "data" / "raw" / "sec_edgar" / "company_list_final.csv"
STATUS_DIR = ROOT / "data" / "raw" / "news" / "status"
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
    NEWS_CFG = yaml.safe_load(f)["news"]


class NewsSpider(scrapy.Spider):
    name = "news_spider"
    handle_httpstatus_list = [429, 500, 502, 503, 504]   # let parse() see these and record them
    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "DOWNLOAD_DELAY": NEWS_CFG["request_delay_seconds"],
        "DOWNLOAD_DELAY_JITTER": 0.3,
        "AUTOTHROTTLE_ENABLED": False,
        "DOWNLOAD_TIMEOUT": 60,
        "RETRY_TIMES": 4,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504],
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, companies=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.company_file = Path(companies) if companies else DEFAULT_LIST
        self.status = {}

    async def start(self):
        companies = pd.read_csv(self.company_file)
        self.logger.info(f"Requesting news for {len(companies)} companies from {self.company_file}")
        for row in companies.itertuples():
            name = row.company_name
            query = quote(f'"{name}" (fraud OR investigation OR lawsuit OR "SEC probe")')
            url = (f"{GDELT}?query={query}&mode=artlist"
                   f"&maxrecords={NEWS_CFG['max_articles_per_company']}&format=json&sort=hybridrel")
            yield scrapy.Request(
                url,
                callback=self.parse,
                errback=self.on_error,
                meta={"company_name": name, "cik": int(row.cik), "label_group": row.label_group},
            )

    def on_error(self, failure):
        name = failure.request.meta["company_name"]
        self.status[name] = "failed"
        self.logger.warning(f"[{name}] request failed: {failure.value}")

    def parse(self, response):
        name = response.meta["company_name"]
        if response.status != 200:
            self.status[name] = "failed"
            self.logger.warning(f"[{name}] HTTP {response.status}, will be retried later")
            return
        try:
            articles = json.loads(response.text).get("articles", [])
        except json.JSONDecodeError:
            self.status[name] = "failed"
            self.logger.warning(f"[{name}] response was not JSON")
            return
        self.status[name] = "ok"
        if not articles:
            self.logger.info(f"[{name}] no matching articles (valid result)")
        for art in articles:
            item = NewsHeadlineItem()
            item["company_name"] = name
            item["headline"] = art.get("title")
            item["published_date"] = art.get("seendate")
            item["source"] = art.get("domain")
            item["link"] = art.get("url")
            yield item

    def closed(self, reason):
        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        out = STATUS_DIR / f"{self.company_file.stem}_status.csv"
        pd.DataFrame([{"company_name": k, "status": v} for k, v in self.status.items()]).to_csv(out, index=False)
        self.logger.info(f"Status saved -> {out}")