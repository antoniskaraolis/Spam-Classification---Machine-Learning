#!/usr/bin/env python
# coding: utf-8

# main.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '', text)
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

def train_test(train_file, test_file):
    train_data = pd.read_csv('train.csv')
    train_data['processed_email'] = train_data['email'].apply(preprocess_text)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data['label'])
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data['processed_email'])

    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    test_data = pd.read_csv('train.csv')
    test_data['processed_email'] = test_data['email'].apply(preprocess_text)

    X_test = vectorizer.transform(test_data['processed_email'])

    predictions = model.predict(X_test)

    with open('predictions.txt', 'w') as f:
        for prediction in predictions:
            f.write(f'{prediction}\n')

    print("Predictions have been written to predictions.txt")

#Call the train_test function
#train_test('train.csv', 'test.csv')


#print the predictions
with open('predictions.txt', 'r') as f:
    print(f.read())

