"""Raw news headlines per company from the GDELT DOC 2.0 API (uncleaned).

Run from src/acquisition/news_scraper:
    scrapy crawl news_spider -o ../../../data/raw/news/news_headlines_raw.csv
To retry a subset, pass a company file:
    scrapy crawl news_spider -a companies=../../../data/raw/sec_edgar/company_list_fraud_missing.csv -o ...

Default input: data/raw/sec_edgar/company_list_final.csv (created by build_sample.py).
GDELT only searches roughly the last three months, so this is a recent-coverage snapshot.
"""
import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import scrapy

from news_scraper.items import NewsHeadlineItem

# spiders -> news_scraper -> news_scraper -> acquisition -> src -> project root
ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LIST = ROOT / "data" / "raw" / "sec_edgar" / "company_list_final.csv"
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"


class NewsSpider(scrapy.Spider):
    name = "news_spider"
    custom_settings = {
        "ROBOTSTXT_OBEY": False,
        "DOWNLOAD_DELAY": 5,                   # GDELT answers HTTP 429 when requests come too fast
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "DOWNLOAD_DELAY_JITTER": 0.5,
        "AUTOTHROTTLE_ENABLED": False,
        "RETRY_TIMES": 5,
    }

    def __init__(self, companies=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.company_file = Path(companies) if companies else DEFAULT_LIST

    async def start(self):
        companies = pd.read_csv(self.company_file)
        self.logger.info(f"Requesting news for {len(companies)} companies from {self.company_file}")
        for row in companies.itertuples():
            name = row.company_name
            query = quote(f'"{name}" (fraud OR investigation OR lawsuit OR "SEC probe")')
            url = f"{GDELT}?query={query}&mode=artlist&maxrecords=8&format=json&sort=hybridrel"
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta={"company_name": name, "cik": int(row.cik), "label_group": row.label_group},
            )

    def parse(self, response):
        name = response.meta["company_name"]
        try:
            articles = json.loads(response.text).get("articles", [])
        except json.JSONDecodeError:
            self.logger.warning(f"[{name}] response was not JSON (status {response.status})")
            return
        if not articles:
            self.logger.info(f"[{name}] no articles returned")
        for art in articles:
            item = NewsHeadlineItem()
            item["company_name"] = name
            item["headline"] = art.get("title")
            item["published_date"] = art.get("seendate")
            item["source"] = art.get("domain")
            item["link"] = art.get("url")
            yield item