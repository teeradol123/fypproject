import pandas as pd
from relevanttopic import relevanttopic
from wordclout import wordcloud
from sentimentanalysis import sentimentanalize
from test4 import multidimensional
df_file = pd.read_csv('/Users/teeradolimamnuaysup/Desktop/The Great Room Merchant Hotel.csv')
df_file_title = df_file[1:5]["Comments"]

print("Please Select the options below for analysis\n" + "1.Most Relevant Topic\n"  +"2.Word Cloud Positive\n" +"3.Word Cloud Negative \n" + "4.Sentiment Analysis\n" +"5.Verification\n" + "6.Multidimensional Reivews\n" + "7.Exit")

while True:
    try:
        userinput = input("Input your options:")

        if userinput == '1':
            relevanttopic(df_file_title)
        if userinput == '2':
            wordcloud(df_file_title,'positive')
        if userinput == '3':
            wordcloud(df_file_title,'negative')
        if userinput == '4':
            sentimentanalize(df_file_title)
        if userinput == '5':
            continue
        if userinput == '6':
            multidimensional(df_file_title)
        if userinput == '7':
            break

    except Exception as e:
        print(e)