
import pandas as pd
import spacy
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import numpy as np
nlp = spacy.load("en_core_web_sm")



df_file = pd.read_csv('/Users/teeradolimamnuaysup/Desktop/test1.csv')
df_file.Comments = df_file.Comments.str.lower()

aspect_terms = []
for review in nlp.pipe(df_file.Comments):
    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
    aspect_terms.append(' '.join(chunks))
df_file['aspect_terms'] = aspect_terms



aspect_categories_model = Sequential()
aspect_categories_model.add(Dense(512, input_shape=(6000,), activation='relu'))
aspect_categories_model.add(Dense(6, activation='softmax'))
aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
vocab_size = 6000 # We set a maximum size for the vocabulary
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df_file.Comments)
aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df_file.aspect_terms))
label_encoder = LabelEncoder()
integer_category = label_encoder.fit_transform(df_file.Category)
dummy_category = to_categorical(integer_category)

aspect_categories_model.fit(aspect_tokenized, dummy_category, epochs=5, verbose=1)

new_review = "the staff were not welcoming"

chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']
new_review_aspect_terms = ' '.join(chunks)
new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])
predict_x = aspect_categories_model.predict(new_review_aspect_tokenized)
class_x= np.argmax(predict_x,axis=1)
new_review_category = label_encoder.inverse_transform(class_x)
print(new_review_category)


sentiment_terms = []
for review in nlp.pipe(df_file['Comments']):
        if review.is_parsed:
            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
        else:
            sentiment_terms.append('')  
df_file['sentiment_terms'] = sentiment_terms
print(df_file)