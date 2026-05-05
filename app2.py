import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import load_model
import re
import streamlit as st 

words_index = imdb.get_word_index()

def preprocesssing(text:str):
    text = re.sub(pattern='[^A-Za-z0-9]',repl=' ',string=text)
    words = text.lower().split()
    one_hot_list = [words_index[word] for word in words]
    return one_hot_list

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

text = st.text_area('Movie Review')

if st.button('clickme'):
    if text == '':
        st.write('please enter your review')
    else:
        one_hot_list = preprocesssing(text)

        model = load_model('simplernn.h5')

        predicted_list = np.array(one_hot_list).reshape(1,-1)
        predicted_value = model.predict(predicted_list)

        review = ''
        if predicted_value[0][0] < 0.5:
            review = 'negative' 
        else:
            review = 'positive'

        st.write(review)

