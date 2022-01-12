import nltk
import pandas as pd
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS 
from wordcloud import WordCloud 
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py

from sklearn.linear_model import LogisticRegression
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



df = pd.read_csv('/Users/teeradolimamnuaysup/Desktop/The Great Room Merchant Hotel.csv')



df = df[df['Stars'] != 3]
df['sentiment'] = df['Stars'].apply(lambda rating : +1 if rating > 3 else -1)

    
positive = df[df['sentiment'] == 1]
negative = df[df['sentiment'] == -1]

def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"','*'))
    return final
df['Comments'] = df['Comments'].apply(remove_punctuation)
df = df.dropna(subset=['Title'])
df['Title'] = df['Title'].apply(remove_punctuation)

dfNew = df[['Title','sentiment']]


index = df.index
df['random_number'] = np.random.randn(len(index))
train = df[df['random_number'] <= 0.8]
test = df[df['random_number'] > 0.8]
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['Title'])
test_matrix = vectorizer.transform(test['Title'])

lr = LogisticRegression()


X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']
lr.fit(X_train,y_train)
predictions =  lr.predict(X_test)
print(predictions)
from sklearn.metrics import confusion_matrix,classification_report
new = np.asarray(y_test)
print(confusion_matrix(predictions,y_test))
print(classification_report(predictions,y_test))