import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import requests
import os, os.path, csv
from urllib.parse import urljoin
import json
from googleapiclient.discovery import build
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)


# main function prints titles and descriptions
@app.route('/search', methods=['POST', 'OPTIONS'])
@cross_origin()
def search():
    query = request.json.get('query')
    if not query:
       return jsonify({'error': 'No query provided'}), 400

    results = crawler(query)
    if not results:
        return jsonify({'error': 'No search results found'}), 404
        
    
    return jsonify(results)


def crawler(query):
  api_key = "AIzaSyApuu62NJc8mYB27-Dz8v4MN8MghW-DwVw"
  resource = build("customsearch", "v1", developerKey=api_key).cse()
  # create and execute request
  result = resource.list(q=query, cx='7312e2f7473b445d3').execute()
  links = []
  for item in result['items']:
      links.append(item['link'])

  titles_descriptions = {}
  for link in links:
      title, abstract = scrape(link)
      titles_descriptions[title] = abstract
#   save_results_to_json(titles_descriptions, 'search_results.json')
  return titles_descriptions


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

    return title, article_texts[0] if article_texts else ''

if __name__ == '__main__':  
   app.run(debug=True)