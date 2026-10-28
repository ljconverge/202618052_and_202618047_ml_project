"""Confirm that every package the project needs can be imported."""
import pandas
import numpy
import sklearn
import imblearn
import matplotlib
import seaborn
import scrapy
import yaml
import requests
import bs4
import vaderSentiment

print("pandas", pandas.__version__, "| numpy", numpy.__version__, "| sklearn", sklearn.__version__)
print("all imports OK")