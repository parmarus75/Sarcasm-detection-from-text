# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:02:36 2019

@author: parma
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
seed=100
pip install spacy && python -m spacy download en


#Importing dataset

dataset = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
dataset.head()

#creating column source by using lambda function
import re
dataset['source'] = dataset['article_link'].apply(lambda x: re.findall(r'\w+', x)[2])
dataset.head()

#data preprocessing
dataset['num_words'] = dataset['headline'].apply(lambda x: len(str(x).split()))
#dataset1=dataset['headline'].apply(lambda x: len(str(x)))

print('Maximum number of word: ', dataset['num_words'].max())
print('\nSentence:\n',dataset[dataset['num_words'] == 39]['headline'].values)
text = dataset[dataset['num_words'] == 39]['headline'].values


f = open('music.txt', 'r')
sent = [word.lower().split() for word in f]

token = [text.split() for word in text]




