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
        if order == 'asc':
            return OrderedDict(sorted(d.items(), key=lambda kv: kv[1], reverse=False))

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
            if article['gender'] == 'F':
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

    from nltk.corpus import stopwords
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

    def plot_weights(self, weights, name):
        from collections import Counter

        weights = list(weights.values())
        vals = []
        for v in weights: 
            vals.append(round(v, 2))

        c = Counter(vals)

        total_values = len(vals)

        vals_90_pct = total_values / 100 * 90

        boundary = 0
        counter = 0
        for i in c:
            if(counter + c[i] < int(vals_90_pct)):
                counter = counter + c[i]
                boundary = i
            else:
                break

        import matplotlib.pyplot as plt
        import numpy as np

        my_cmap = plt.get_cmap("viridis")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

        plt.figure(figsize=[10,6])
        bars = plt.bar(list(c.keys()), list(c.values()), color=my_cmap(rescale(list(c.values()))), width=0.01, alpha=0.7, align='center')

        plt.legend(loc="best")
        plt.ylim([0, max(list(c.values()))+10])
        ax2 = plt.gca()

        ymin, ymax = ax2.get_ylim()
        plt.vlines(boundary, ymin=ymin, ymax=ymax, colors='r', label='90% of values')

        plt.ylabel('Frequency of weight', fontdict={'fontsize':13, 'fontweight': 'bold'})
        plt.xlabel('# of words with same weight', fontdict={'fontsize':13, 'fontweight': 'bold'})
        plt.title("Distribution of " + name + " lengths", fontdict={'fontsize':14, 'fontweight': 'bold'})
        plt.legend()
        plt.show()
        print("Boundary is:", boundary)
        print("Counter is:", counter)
        return counter

    def plot_weight_and_polarity(self, m_weights, w_weights, pols, name):
        import numpy as np 
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import normalize

        m_weights = list(m_weights.values())
        w_weights = list(w_weights.values())
        pols = list(pols.values())
        # m_weights = np.array(m_weights)
        pol_to_rep = {
        -1.0: 0,
        -0.9: 1,
        -0.8: 2,
        -0.7: 3,
        -0.6: 4,
        -0.5: 5,
        -0.4: 6,
        -0.3: 7,
        -0.2: 8,
        -0.1: 9,
        -0.0: 10,
        0.0: 10,
        0.1: 11,
        0.2: 12,
        0.3: 13,
        0.4: 14,
        0.5: 15,
        0.6: 16,
        0.7: 17,
        0.8: 18,
        0.9: 19,
        1.0: 20,
        }

        m_weights_rounded = []
        for weight in m_weights:
            m_weights_rounded.append(round(float(weight), 2))
        x, y = np.unique(m_weights_rounded, return_counts=True)


        plt.figure(figsize=[10,6])
        plt.plot(x, y, color='b', alpha=0.7)


        w_weights_rounded = []
        for weight in w_weights:
            w_weights_rounded.append(round(float(weight), 2))
        x, y = np.unique(w_weights_rounded, return_counts=True)

        plt.plot(x, y, color='r', alpha=0.7)

        pols_rounded = []
        for weight in pols:
            pols_rounded.append(round(float(weight), 2))
        x, y = np.unique(pols_rounded, return_counts=True)

        plt.plot(x, y, color='g', alpha=0.7)

        plt.ylabel('Frequency of words', fontdict={'fontsize':13, 'fontweight': 'bold'})
        plt.xlabel('Polarity', fontdict={'fontsize':13, 'fontweight': 'bold'})
        plt.title("Distribution of normalized weights and polarities", fontdict={'fontsize':14, 'fontweight': 'bold'})
        plt.legend()
        plt.show()