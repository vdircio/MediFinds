import numpy as np
import math
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import requests
import os, os.path, csv
import queue
from urllib.parse import urljoin
import json
from googleapiclient.discovery import build


# # iterates through all the links and appends them to list
# def crawl(url):
#   links = []
#   response = requests.get(url)
#   if response.status_code == 200:
#     soup = BeautifulSoup(response.text, 'html.parser')
#     main_content_section = soup.find(id='main-content')
#     if main_content_section:
#       for link in main_content_section.find_all('a', href=True):
#         href = link['href']
#         full_url = urljoin(url, href)
#         if not full_url.endswith(".pdf"):
#           links.append(full_url)
#         if len(links) == 50:
#           break
#     return links[7:]

def save_results_to_json(results, filename):
    with open(filename, "w") as json_file:
        json.dump(results, json_file, indent=4)

# scrapes web pages to get title and description
def scrape(url):
    result = requests.get(url).text
    doc = BeautifulSoup(result, "html.parser")
    title = doc.title.text.strip() if doc.title else "Title Not Found"
    article_sections = doc.find_all(class_="c-article-section")
    article_texts = []
    for section in article_sections:
        section_text = section.get_text(separator='\n').strip()
        article_texts.append(section_text)

    return title, article_texts[0]

# main function prints titles and descriptions
def crawler(url, query):
  api_key = "AIzaSyApuu62NJc8mYB27-Dz8v4MN8MghW-DwVw"
  resource = build("customsearch", "v1", developerKey=api_key).cse()
  # create and execute request
  result = resource.list(q="python", cx='7312e2f7473b445d3').execute()
  links = []
  for item in result['items']:
      links.append(item['link'])

  titles_descriptions = {}
  for link in links:
      title, abstract = scrape(link)
      titles_descriptions[title] = abstract

  save_results_to_json(titles_descriptions, 'searc_results.json')
  
  return titles_descriptions