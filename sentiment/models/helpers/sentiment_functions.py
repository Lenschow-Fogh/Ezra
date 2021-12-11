import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from collections import Counter
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras import models
import os
import time
import tensorflow as tf
from sklearn.metrics import classification_report
from keras import callbacks


def gender_seq_to_single(seqs):
        genders = []
        for seq in seqs:
            genders.append(seq[0])
        return genders

def equal_genders(data_):
    M = data_["Gender"].value_counts()['M']
    F = data_["Gender"].value_counts()['F']
    
    if M>F:
        diff = M-F
        return data_.drop(data_.loc[data_['Gender'] == 'M'].index[:diff], axis=0)
    elif F>M:
        diff = F-M
        return data_.drop(data_.loc[data_['Gender'] == 'F'].index[:diff], axis=0)

def encode_feature(train_data_, test_data_):
    tokenizer = Tokenizer()
    # ONLY FIT ON TRAIN DATA
    tokenizer.fit_on_texts(train_data_)
    return tokenizer.texts_to_sequences(train_data_), tokenizer.texts_to_sequences(test_data_), len(tokenizer.word_index)

def plot_confusion_matrix(cm_true, cm_pred, title, xlabel, ylabel):
    # Plotting pred confusion matrix
    cm = confusion_matrix(cm_true, cm_pred, normalize='pred')

    fig = plt.figure( figsize=[18.5,10.5])
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    ax.set_xlabel(xlabel, fontsize = 30, labelpad=15.0)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(ylabel, fontsize = 30, labelpad=15.0)
    ax.set_title(title+' (pred)',fontweight="bold", size=30, pad=70.0)

    cm_axis_vals = []

    for x in np.unique(np.array(cm_pred)):
        cm_axis_vals.append(x)

    cb = fig.colorbar(cax)
    cb.set_label(label='',size='xx-large', weight='bold')
    cb.ax.tick_params(labelsize='xx-large')
    plt.xticks(range(2), cm_axis_vals, rotation=90, fontsize=25)
    plt.yticks(range(2), cm_axis_vals, fontsize=25)
    plt.show()

    # Plotting true confusion matrix
    cm = confusion_matrix(cm_true, cm_pred, normalize='true')

    fig = plt.figure( figsize=[18.5,10.5])
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    ax.set_xlabel(xlabel, fontsize = 30, labelpad=15.0)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(ylabel, fontsize = 30, labelpad=15.0)
    ax.set_title(title+' (true)',fontweight="bold", size=30, pad=70.0)

    cm_axis_vals = []

    for x in np.unique(np.array(cm_pred)):
        cm_axis_vals.append(x)

    cb = fig.colorbar(cax)
    cb.set_label(label='',size='xx-large', weight='bold')
    cb.ax.tick_params(labelsize='xx-large')
    plt.xticks(range(2), cm_axis_vals, rotation=90, fontsize=25)
    plt.yticks(range(2), cm_axis_vals, fontsize=25)
    plt.show()

def get_metrics(y_test, y_pred, vocab, vocab_name):
    report = classification_report(y_test, y_pred, output_dict=True)
    df_perf = pd.DataFrame.from_dict(report).transpose()
    df_perf_2 = df_perf[:len(vocab)]
    df_perf_2.insert(loc=0, column=vocab_name, value=vocab)
    df_perf_2.precision = df_perf_2.precision.round(2)
    df_perf_2.recall = df_perf_2.recall.round(2)
    df_perf_2['f1-score'] = df_perf_2['f1-score'].round(2)
    df_perf_2.support = df_perf_2.support.round()
    df_perf_2.reset_index(drop=True, inplace=True)


    return df_perf_2

def xai_binary(predictions, input_data, N_greatest_polarities):
    # Borrowed from: https://www.geeksforgeeks.org/python-program-to-find-n-largest-elements-from-a-list/
    # Function returns N largest elements
    def Nmaxelements(list, N):
        return list[:N]

    def find_complete_sentence(sentence_number, sentences):
        index = sentences.index[sentences['Sentence #'] == sentence_number].tolist()[0]
        return sentences['Text'][index]
    
    complete_sentences = pd.read_json('./../datasets/7_sentences.json')

    for pred, row in zip(predictions, input_data.iterrows()):
        data = row[1]

        print("\n---------------------------------------------------------------------------------------")
        print("The sentence:", '"'+find_complete_sentence(data['Sentence #'], complete_sentences)+'".\n')
        print("Was predicted to be", 'masculine' if pred == 0 else 'feminine', "("+str(pred)+").\n")

        print("Most", 'masculine' if pred == 0 else 'feminine', "words in sentence are:\n")
        greatest_pols = []

        if (pred == 0):
            greatest_pols = Nmaxelements(sorted(data['Polarity']), N_greatest_polarities)
        else:
            greatest_pols = Nmaxelements(sorted(data['Polarity'], reverse=True), N_greatest_polarities)

        for pol in greatest_pols:
            i = data['Polarity'].index(pol)
            print("'"+data["Word"][i]+"'", "with a polarity of", round(pol,2))

def plot_sentence_lengths(data_):
    sentence_plot = data_["Word"].values
    sentence_plot_sorted = list(sorted(sentence_plot, key=len))
    c = Counter(map(len, sentence_plot_sorted))

    total_sentences = 0
    total_words = 0
    for i in c:
        total_sentences = total_sentences + c[i]
        total_words = total_words + c[i]*i

    sentences_80_pct = total_sentences / 100 * 90
    words_80_pct = total_words / 100 * 90

    boundary_sen = 0
    counter_sen = 0

    for i in c:
        if(counter_sen + c[i] < int(sentences_80_pct)):
            counter_sen = counter_sen + c[i]
            boundary_sen = i
        else:
            break

    boundary_word = 0
    counter_word = 0

    for i in c:
        if(counter_word + c[i] * i < int(words_80_pct)):
            counter_word = counter_word + c[i] * i
            boundary_word = i
        else:
            break


    my_cmap = plt.get_cmap("viridis")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))


    plt.figure(figsize=[10,6])
    bars = plt.bar(list(c.keys()), list(c.values()), color=my_cmap(rescale(list(c.values()))), width=0.8, alpha=0.7, align='center')

    # for r in bars.get_children():
    #     if(r.get_x() > boundary_sen):
    #         r.set_alpha(0.2)

    plt.legend(loc="best")
    plt.ylim([0, max(list(c.values()))+10])
    ax2 = plt.gca()

    ymin, ymax = ax2.get_ylim()
    plt.vlines(boundary_sen, ymin=ymin, ymax=ymax, colors='r', label='80% of sentences')
    # plt.vlines(boundary_word, ymin=ymin, ymax=ymax, colors='black', label="80% of words")

    plt.ylabel('Frequency of sentence', fontdict={'fontsize':13, 'fontweight': 'bold'})
    plt.xlabel('# words in sentence', fontdict={'fontsize':13, 'fontweight': 'bold'})
    plt.title("Distribution of sentence lengths", fontdict={'fontsize':14, 'fontweight': 'bold'})
    plt.legend()
    plt.show()
    return boundary_sen
