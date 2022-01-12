import nltk
import pandas as pd
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS 
from wordcloud import WordCloud 
from nltk.corpus import stopwords,wordnet 
from typing import List
from nltk.stem import WordNetLemmatizer
import string 
nltk.download([
     "names",
     "stopwords",
     "state_union",
     "twitter_samples",
     "movie_reviews",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt",
     "shakespeare",
     "wordnet"
 ])

def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):
        return wordnet.ADJ 

    elif pos_tag.startswith('V'):
        return wordnet.VERB 

    elif pos_tag.startswith('N'):
        return wordnet.NOUN

    elif pos_tag.startswith('R'):
        return wordnet.ADV
    
    else:
        return wordnet.NOUN



df_file = pd.read_csv('/Users/teeradolimamnuaysup/Desktop/The Great Room Merchant Hotel.csv')
df_file_title = df_file["Comments"]


def relevanttopic(df):
    df_file_title2 = df.str.lower()
    df_file_title3 = df_file_title2.tolist()
    stop_words = (nltk.corpus.stopwords.words('english'))
    stop_words2 = ["restaurant","everything","anything","nothing","thing","need",
                        "good","great","excellent","perfect","much","even","really",'wonderful','fabulous','awesome','outstanding']
    stop_words.extend(stop_words2)
    stop_words = set(stop_words)
    pattern = r'\b[a-zA-Z]{3,}\b'
    tagged_tokens =[]
    lemmatized = ''
    for i in df_file_title3:
        tokens = nltk.regexp_tokenize(i, pattern) 
        tagged_tokens= (nltk.pos_tag(tokens))
        
        wordnet_lemmatizer = WordNetLemmatizer()
        # get lemmatized tokens                             #call function "get_wordnet_pos"
        lemmatized_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
                  # tagged_tokens is a list of tuples (word, tag)
            for (word, tag) in tagged_tokens \
                  # remove stop words
            if word not in stop_words and \
                  # remove punctuations
            word not in string.punctuation and tag == 'NN']
        for w in lemmatized_words:
            lemmatized += w + ' '
    



    
    words : list[str] = nltk.word_tokenize(lemmatized)

    
   
    fd = nltk.FreqDist(words)
    print(fd.most_common(10))


