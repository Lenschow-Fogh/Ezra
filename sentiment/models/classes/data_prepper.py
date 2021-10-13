import json
import requests
import math
from bs4 import BeautifulSoup
from collections import OrderedDict
import pandas as pd
import numpy as np
import re


class DataPrepper:
    def __init__(self):
        print("constructor of DataPrepper")

    def load_json(self, path):
        with open(path) as json_file:
            return json.load(json_file)

    def write_json(self, path, data):
        with open(path, "w") as out_file:
            json.dump(data, out_file)

    def filter_articles(self, categories, articles):
        filt_articles = []
        for article in articles:
            if article['category'] in categories and article['link'].find("https://www.huffingtonpost.comhttp") == -1:
                filt_articles.append(article)
        return filt_articles

    def scrape_url(self, url, textlessUrls):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        text = ""

        if soup.find('div', attrs={'class': 'content-list-component yr-content-list-text text'}) is not None:
            for paragraph_wrapper in soup.findAll('div', attrs={'class': 'content-list-component yr-content-list-text text'}):
                if paragraph_wrapper.find('p') is not None:
                    p = paragraph_wrapper.find('p').text
                    # t.replace("\u", " ")
                    p = p.replace("\u201c", ' ')
                    p = p.replace('\u201d', ' ')
                    p = p.replace('\u00a0', ' ')
                    text += p + " "
        if soup.find('div', attrs={'class': 'primary-cli cli cli-text'}) is not None:
            for paragraph_wrapper in soup.findAll('div', attrs={'class': 'primary-cli cli cli-text'}):
                if paragraph_wrapper.find('p') is not None:
                    p = paragraph_wrapper.find('p').text
                    # t.replace("\u", " ")
                    p = p.replace("\u201c", ' ')
                    p = p.replace('\u201d', ' ')
                    p = p.replace('\u00a0', ' ')
                    text += p + " "
        if soup.find('div', attrs={'class': 'primary-cli cli cli-text '}) is not None:
            for paragraph_wrapper in soup.findAll('div', attrs={'class': 'primary-cli cli cli-text '}):
                if paragraph_wrapper.find('p') is not None:
                    p = paragraph_wrapper.find('p').text
                    # t.replace("\u", " ")
                    p = p.replace("\u201c", ' ')
                    p = p.replace('\u201d', ' ')
                    p = p.replace('\u00a0', ' ')
                    text += p + " "
        if text == "":
            textlessUrls.append(url)
            print("-----------------------------------------------------------------\nError no text found on url:",
                  url, "\ntotal textless urls so far:", len(textlessUrls), "\nAll textless urls:", textlessUrls)
        return text

    def order_dict(self, d, order):
        if order == 'desc':
            return OrderedDict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))

    def get_weights(self, data, nlp, ignore_terms, ignore_ents):
        N = len(data)
        m_tf = []
        m_df = {}
        w_tf = []
        w_df = {}

        for article in data:
            doc = nlp(article['text'])
            tf_d = {}
            if article['gender'] == 'M':
                for token in doc:
                    if not token.is_stop and token.lemma_ not in ignore_terms and token.ent_type_ not in ignore_ents:
                        word = token.lemma_
                        if word in tf_d:
                            tf_d[word] += 1
                        else:
                            tf_d[word] = 1
                m_tf.append(tf_d)
                for key in tf_d:
                    if key in m_df:
                        m_df[key] += 1
                    else:
                        m_df[key] = 1
            if article['gender'] == 'W':
                for token in doc:
                    if not token.is_stop and token.lemma_ not in ignore_terms and token.ent_type_ not in ignore_ents:
                        word = token.lemma_
                        if word in tf_d:
                            tf_d[word] += 1
                        else:
                            tf_d[word] = 1
                w_tf.append(tf_d)
                for key in tf_d:
                    if key in w_df:
                        w_df[key] += 1
                    else:
                        w_df[key] = 1

        m_tf_idfs = {}
        for tf_dict in m_tf:
            tf_idf = {}
            for k, v in tf_dict.items():
                tf_idf = v*(math.log(N/m_df[k]))
                if k in m_tf_idfs:
                    m_tf_idfs[k].append(tf_idf)
                else:
                    m_tf_idfs[k] = [tf_idf]

        w_tf_idfs = {}
        for tf_dict in w_tf:
            tf_idf = {}
            for k, v in tf_dict.items():
                tf_idf = v*(math.log(N/w_df[k]))
                if k in w_tf_idfs:
                    w_tf_idfs[k].append(tf_idf)
                else:
                    w_tf_idfs[k] = [tf_idf]

        for k, v in m_tf_idfs.items():
            m_tf_idfs[k] = sum(v)/N

        for k, v in w_tf_idfs.items():
            w_tf_idfs[k] = sum(v)/N

        return m_tf_idfs, w_tf_idfs

    # Borrowed from https://stats.stackexchange.com/a/70807
    def normalize_dict(self, d):
        v_min = min(d.values())
        v_max = max(d.values())
        for k in d:
            d[k] = (d[k]-v_min)/(v_max-v_min)

    def get_polarity(self, d1, d2):
        pol_dict = {key: d1[key] - d2.get(key, 0) for key in d1}
        for k, v in d2.items():
            if k not in pol_dict:
                pol_dict[k] = -v
        return pol_dict

    def remove_tags(self, text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)

    # Borrowed from: https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91 
    # and https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
    def preprocess_text(self, sen):
        # Removing html tags
        sentence = self.remove_tags(sen)

        #Removing URLs with a regular expression
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        sentence = url_pattern.sub(r'', sentence)

        # Remove Emails
        sentence = re.sub('\S*@\S*\s?', '', sentence)
        
        # Remove new line characters
        sentence = re.sub('\s+', ' ', sentence)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Remove distracting single quotes
        sentence = re.sub("\'", "", sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence