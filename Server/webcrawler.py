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

# iterates through all the links and appends them to list
def crawl(url):
  links = []
  response = requests.get(url)
  if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    main_content_section = soup.find(id='main-content')
    if main_content_section:
      for link in main_content_section.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(url, href)
        if not full_url.endswith(".pdf"):
          links.append(full_url)
        if len(links) == 50:
          break
    return links[7:]

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
def crawler(url):
  links = crawl(url)
  titles_descriptions = {}
  for link in links:
      title, abstract = scrape(link)
      titles_descriptions[title] = abstract
  return titles_descriptions