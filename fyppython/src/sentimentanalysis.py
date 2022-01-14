from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd




sia = SentimentIntensityAnalyzer()

df_file = pd.read_csv('/Users/teeradolimamnuaysup/Desktop/Oliver Restaurant.csv')
df_file_title = df_file["Title"]

def sentimentanalize(text):

    df_file_title2 = ['food is good','service is bad']
    #text.tolist()

    for i in df_file_title2:
        print(i)
        print(sia.polarity_scores(i))




sentimentanalize(df_file_title)