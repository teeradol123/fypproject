
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")


df_file = pd.read_csv('/Users/teeradolimamnuaysup/Desktop/The Great Room Merchant Hotel.csv')
df_file_title = df_file["Comments"]

def multidimensional(text):
  sentences = text.tolist()
  topics = []
  for sentence in sentences:
    doc = nlp(sentence)
    
    descriptive_term = ''
    target = ''
    for token in doc:
      
      
      if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
        target = token.text
      if token.dep_ == 'amod' and token.head.pos_ == 'NOUN':
          target = token.head.text
      if token.pos_ == 'ADJ' and token.head.pos_ == 'AUX':
        prepend = ''
        for child in token.children:
          if child.pos_ != 'ADV':
            continue
          prepend += child.text + ' '
        descriptive_term = prepend + token.text
        topics.append({'topic': target,
        'description': descriptive_term})
      if token.dep_ == 'amod' and token.pos_ == 'ADJ':
          prepend = ''
          for child in token.children:
              if child.pos_ != 'ADV':
                  continue
              prepend += child.text + ' '
          descriptive_term = prepend + token.text
          topics.append({'topic': target,
          'description': descriptive_term})
      
  print(topics)

 

