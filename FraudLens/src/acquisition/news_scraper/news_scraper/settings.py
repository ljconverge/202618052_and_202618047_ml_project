"""Project-wide Scrapy settings. The spider's custom_settings override these."""
BOT_NAME = "news_scraper"
SPIDER_MODULES = ["news_scraper.spiders"]
NEWSPIDER_MODULE = "news_scraper.spiders"
ROBOTSTXT_OBEY = False
FEED_EXPORT_ENCODING = "utf-8"
LOG_LEVEL = "INFO"