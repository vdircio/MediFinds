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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
import re
import torch


app = Flask(__name__)
cors = CORS(app)

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

# main function 
# returns title and abstract to search function which sends results to frontend
def crawler(query):
    # Google search api retrieves links
    api_key = "AIzaSyApuu62NJc8mYB27-Dz8v4MN8MghW-DwVw"
    resource = build("customsearch", "v1", developerKey=api_key).cse()
    result = resource.list(q=query, cx='7312e2f7473b445d3').execute()
    links = [item['link'] for item in result['items']]

    # dictionary holding titles and abstracts
    titles_descriptions = {}
    for link in links:
        title, abstract = scrape(link)
        titles_descriptions[title] = abstract

    # clean and preprocess text
    cleaned_abstracts = {title: remove_headers(abstract) for title, abstract in titles_descriptions.items()}

    # Create a language model
    CM, BOW = collection_LM(list(cleaned_abstracts.values()))

    # Query likelihood and summarize
    summaries = {title: query_likelihood(query, abstract, BOW, CM) for title, abstract in cleaned_abstracts.items()} 
    final_summaries = {title: summarize(summary) for title, summary in summaries.items()} 

    return final_summaries


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


def summarize(summaries):
    final_sums = []
    for summary in summaries:
        text = ""
        for j in i:
            text += j.capitalize()
        inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=60, min_length=20, length_penalty=5., num_beams=2)
        summary = tokenizer.decode(summary_ids[0])
        final_sums.append(summary)
    
    final_sums.pop()

    for sen in final_sums:
       sen = remove_tags(sen)

    return final_sums

def remove_tags(sen):
    sen = re.sub('<pad>', '', sen)
    sen = re.sub('</s>', '', sen)
    sen = re.sub('<', '', sen)
    sen = sen.strip()
    return sen

def remove_headers(text):
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sentence in sentences:
        pos_newline = sentence.find('\n')
        if pos_newline != -1:
            sentence = sentence[pos_newline + 1:]
        sentence = sentence.replace('\n', ' ')
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence).lower()
        if sentence.strip():
            cleaned_sentences.append(sentence.strip())
    return cleaned_sentences


def clean_words(words):
    # Cleans list of words by removing punctuation and digits
    cleaned = [re.sub(r'[^\w\s]', '', word) for word in words]
    cleaned = [re.sub(r'\d+', '', word) for word in cleaned]
    return cleaned

def collection_LM(list_of_abs):
    nltk.download('punkt')
    stemmer = PorterStemmer()
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    dictionary = {}
    for abstract in list_of_abs:
        for sentence in abstract:
            words = clean_words(sentence.split())
            for word in words:
                stemmed_word = stemmer.stem(word)
                if stemmed_word not in stop_words and len(stemmed_word) > 2:
                    dictionary[stemmed_word] = dictionary.get(stemmed_word, 0) + 1

    # Calculate probabilities instead of frequencies
    total_freq = sum(dictionary.values())
    for word in dictionary:
        dictionary[word] /= total_freq

    # Sort dictionary by frequency in descending order and create a list of the most frequent words
    sorted_dict = dict(sorted(dictionary.items(), key=lambda x: x[1], reverse=True))
    bag_of_words = list(sorted_dict)[:50]  # Limit to top 50 words

    return sorted_dict, bag_of_words


def query_likelihood(query, abstract_sentences, bagofwords, CM):
  # Cleaner Function For Abstracts!
  def cleaner(document):
    # Replace all numbers with a space!
    nonums = re.sub(r'[0-9]', ' ', document)
    # Replace all punctuation with space!
    nopunc = re.sub(r'[^\w\s]', ' ', nonums)
    # Lowercase all remaining words!
    lower = nopunc.lower()
    # Get Rid of all stopwords and join remaining through a space!
    stop_words = (stopwords.words('english'))
    words = word_tokenize(lower)
    filter = [w for w in words if not w.lower() in stop_words]
    # Now Remove words of length 2 or below as they are either stopwords not
    # included in NLTK (such as hi or us) or are gibberish (gt)!
    filter = [w for w in filter if len(w) > 2]
    filter = ' '.join(filter)
    return filter

  # Query is a string of words
  # file_name is the name of the file that will be used!
  # Vocab is a list containing all vocab words

  # First Split Query into list of words
  # This time we want to stem since vocab is stemmed!
  words = query.split()
  ps = PorterStemmer()
  for i in range(len(words)):
    words[i] = ps.stem(words[i])

  # First we want to find doc/sentence length normalization n!
  n = 0
  for i in range(len(abstract_sentences)):
      filtered_doc = cleaner(abstract_sentences[i])
      length = len(filtered_doc.split())
      n += length
  n = n / len(abstract_sentences)

  # Our third step is to find the count of vocab words in the query
  c_w_q = np.zeros(len(bagofwords))
  for i in range(len(bagofwords)):
    for j in range(len(words)):
      if (words[j] == bagofwords[i]):
        c_w_q[i] += 1
  # We will utilize Jelinek Mercer Smoothing to help!
  # Our lambda will be between 0 and 1
  lambd = 0.4
  # Now we can define our ranking list
  ranking = []
  for i in range(len(abstract_sentences)):
    c_w_d = np.zeros(len(bagofwords))
    p_w_C = np.zeros(len(bagofwords))

    filtered_doc = cleaner(abstract_sentences[i])
    d = len(filtered_doc.split())
    for j in range(len(bagofwords)):
      c_w_d[j] += filtered_doc.count(bagofwords[j])
      p_w_C[j] += CM[bagofwords[j]]
    ranking.append((np.sum(c_w_q * np.log(1 + ((1 - lambd)/lambd) * (c_w_d/(d * p_w_C)))), i))

  ranking = sorted(ranking, key = lambda k: k[0], reverse = False)
  return abstract_sentences[ranking[-1][1]], abstract_sentences[ranking[-2][1]]

if __name__ == '__main__':  
   app.run(debug=True)