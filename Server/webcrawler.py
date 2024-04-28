<<<<<<< HEAD:webcrawler.py
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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
import re


app = Flask(__name__)z
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

    results = {} # list to hold link, title, and summaries
    cleaned_abstracts = []
    titles = []

    for link in links:
        title, abstract = scrape(link)
        cleaned_abstract = remove_headers(abstract)
        cleaned_abstracts.append(cleaned_abstract)
        titles.append(title)

    # Create a language model
    CM, BOW = collection_LM(list(cleaned_abstracts))

    summaries = []
    for i in cleaned_abstracts:
        summaries.append(query_likelihood(query, i, BOW, CM))

    final_summaries = []
    for summary in summaries:
        text = ""
        for sentence in summary:
            text += sentence.capitalize()
            inputs = tokenizer.encode("summarize: " + text,
                                      return_tensors='pt',
                                      max_length=512,
                                      truncation=True)
            summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5., num_beams=2)
            final_summary = tokenizer.decode(summary_ids[0])
            final_summary = final_summary.replace("<pad>", "").replace("</s>", "")
            final_summary = capitalize_sentences(final_summary)
            final_summary = remove_unfinished(final_summary)
            final_summaries.append(final_summary)
    
    # for i in range(len(final_summaries)):
    #     final_summaries[i] = remove_unfinished

    for link, title, final_summary in zip(links, titles, final_summaries):
        results[link] = {'link': link, 'title': title, 'summary': final_summary}

    results.pop(links[-1])
    return results

def remove_unfinished(sentence):
    seq = sentence.split()
    final_index = 0
    for i in range(len(seq)):
      if "." in seq[i]:
        final_index = i
    return " ".join(seq[: final_index + 1])

def capitalize_sentences(text):
    seq = text.split()
    seq[0] = seq[0].capitalize()
    for i in range(len(seq)):
      if "." in seq[i] and i < len(seq) - 1:
        seq[i + 1] = seq[i + 1].capitalize()
    return " ".join(seq)

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



def remove_headers(text):
    headers = ["Abstract", "Background", "Conclusion", "Methods"]
    for header in headers:
        text = text.replace(header, '')
    sentences = nltk.sent_tokenize(text.replace('\n', ' '))
    
    return sentences

def clean_words(words):
    # Cleans list of words by removing punctuation and digits
    cleaned = [re.sub(r'[^\w\s]', '', word) for word in words]
    cleaned = [re.sub(r'\d+', '', word) for word in cleaned]
    return cleaned

def collection_LM(list_of_abs):
    # nltk.download('punkt')
    stemmer = PorterStemmer()
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    # stemmer = PorterStemmer()
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
    bag_of_words = list(sorted_dict.keys())[:50]  # Limit to top 50 words

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
    n = n / len(abstract_sentences) if len(abstract_sentences) != 0 else 0

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
    for sentence in abstract_sentences:
        filtered_doc = cleaner(sentence)
        doc_words = filtered_doc.split()
        d = len(doc_words)
        if d == 0:
            continue  # Skip empty documents to avoid division by zero
        
        c_w_d = np.array([doc_words.count(word) for word in bagofwords])
        p_w_C = np.array([CM.get(word, 0) for word in bagofwords])

        # Safe divide and safe log calculation to avoid runtime warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            term_freq = c_w_d / (d * p_w_C)
            log_terms = np.log(1 + ((1 - lambd) / lambd) * term_freq)
            log_terms[np.isnan(log_terms)] = 0  # Replace NaNs with 0

        ranking.append((np.sum(c_w_q * log_terms), sentence))

    ranking.sort(key=lambda x: x[0], reverse=True)

    # Return the top two sentences or abstracts, as per the original intent
    return [x[1] for x in ranking[:2]]

def remove_tags(sen):
    sen = re.sub('<pad>', '', sen)
    sen = re.sub('</s>', '', sen)
    sen = re.sub('<', '', sen)
    sen = sen.strip()
    return sen

if __name__ == '__main__':  
      app.run(host='0.0.0.0', port=5000)

=======
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
from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)
import re


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

    results = [] # list to hold link, title, and summary description
    cleaned_abstracts = []

    for link in links:
        title, abstract = scrape(link)
        cleaned_abstract = remove_headers(abstract)
        cleaned_abstracts.append(cleaned_abstract)
        results.append({'link': link,'title': title})

    # Create a language model
    CM, BOW = collection_LM(list(cleaned_abstracts))

    # Query likelihood and summarize
    for result, cleaned_abstract in zip(results, cleaned_abstracts):
        summary = summarize(query_likelihood(query, cleaned_abstract, BOW, CM))
        result['summary'] = summary  # Add summary to dictionary

    return results


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
    # nltk.download('punkt')
    stemmer = PorterStemmer()
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    # stemmer = PorterStemmer()
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
    bag_of_words = list(sorted_dict.keys())[:50]  # Limit to top 50 words

    return sorted_dict, bag_of_words


def summarize(summaries):
    final_sums = []
    for summary in summaries:
        text = ""
        for j in summary:
            text += j.capitalize()
        inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=60, min_length=20, length_penalty=5., num_beams=2)
        summary = tokenizer.decode(summary_ids[0])
        final_sums.append(summary)
    
    # final_sums.pop()

    for i in range(len(final_sums)):
        final_sums[i] = remove_tags(final_sums[i])

    return final_sums

def query_likelihood(query, abstract_sentences, bagofwords, CM):
    # Cleaner Function For Abstracts!
    def cleaner(document):
        nonums = re.sub(r'[0-9]', ' ', document)
        nopunc = re.sub(r'[^\w\s]', ' ', nonums)
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

def remove_tags(sen):
    sen = re.sub('<pad>', '', sen)
    sen = re.sub('</s>', '', sen)
    sen = re.sub('<', '', sen)
    sen = sen.strip()
    return sen

if __name__ == '__main__':  
      app.run(host='0.0.0.0', port=5000)

>>>>>>> 43580f29fb80d0af24b1dfc8d82af461bf5eb0cb:Server/webcrawler.py
