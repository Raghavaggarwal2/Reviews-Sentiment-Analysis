import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load pretrained model
model = load_model('Simple_Rnn_imdb.h5') 

def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
  words = text.lower().split()
  encoded_review = []
  for word in words:
    idx = word_index.get(word, 2) + 3
    # Only allow indices in [0, 9999], else use 2 (unknown)
    if idx < 10000:
      encoded_review.append(idx)
    else:
      encoded_review.append(2)
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

# Prediction function
def predict_sentiment(review):
  preprocessed_input = preprocess_text(review)
  prediction = model.predict(preprocessed_input)

  sentiment = 'Positive' if prediction > 0.5 else 'Negative'
  return sentiment, prediction[0][0]

import streamlit as st

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
  if not user_input.strip():
    st.warning('Please enter a movie review before classifying.')
  else:
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
  st.write('Please enter a movie review.')
  st.write('Please enter a movie review.')
