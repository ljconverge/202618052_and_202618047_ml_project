"""Fields for one scraped headline. Scrapy writes these to the CSV output."""
import scrapy


class NewsHeadlineItem(scrapy.Item):
    company_name = scrapy.Field()
    headline = scrapy.Field()
    published_date = scrapy.Field()
    source = scrapy.Field()
    link = scrapy.Field()