from textblob import TextBlob
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df_file = pd.read_csv('/Users/teeradolimamnuaysup/Desktop/The Great Room Merchant Hotel.csv')
df_file_title = df_file["Comments"]



def wordcloud(text,type):
    df_file = text.str.lower()
    df_file_title2 = df_file.tolist()
    stop_words = (nltk.corpus.stopwords.words('english'))
    stop_words2 = ["restaurant","everything","anything","nothing","thing","need",
                            "good","great","excellent","perfect","much","even","really",'wonderful','fabulous','awesome','outstanding']
    stop_words.extend(stop_words2)
    stop_words = set(stop_words)
    pattern = r'\b[a-zA-Z]{3,}\b'

    positive = ""
    negative = ""









    for i in df_file_title2:
        i.lower()
        blob_i = TextBlob(i)
        sentimenttest = blob_i.sentiment.polarity
        if sentimenttest > 0:
            positive += i + ' '
        else:
            negative += i + ' '

   
    tokenspos = nltk.regexp_tokenize(positive, pattern) 
    tokensneg = nltk.regexp_tokenize(negative, pattern) 
    posnew = ''
    negnew = ''
    for i in tokenspos:
        if i not in stop_words:
            posnew += i + " "
    for i in tokensneg:
        if i not in stop_words:
            negnew += i + " "
    

    if type == 'positive':
        cloud_test2 = WordCloud(background_color="white").generate(posnew)
        plt.imshow(cloud_test2, interpolation='bilinear') 
        plt.axis("off")
        plt.show()

    if type == 'negative':
        cloud_test = WordCloud(background_color="white").generate(negnew)
        plt.imshow(cloud_test, interpolation='bilinear') 
        plt.axis("off")
        plt.show()



