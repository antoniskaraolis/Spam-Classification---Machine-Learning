#!/usr/bin/env python
# coding: utf-8

# In[1]:


# experiments.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

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

data = pd.read_csv('train.csv')
data['processed_email'] = data['email'].apply(preprocess_text)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['label'])
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['processed_email'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_val)

print("SVM Model Evaluation:")
print(metrics.classification_report(y_val, svm_predictions))
print(metrics.confusion_matrix(y_val, svm_predictions))

mnb_model = MultinomialNB()
mnb_model.fit(X_train, y_train)

mnb_predictions = mnb_model.predict(X_val)

print("Multinomial Naive Bayes Model Evaluation:")
print(metrics.classification_report(y_val, mnb_predictions))
print(metrics.confusion_matrix(y_val, mnb_predictions))

