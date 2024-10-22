import nltk
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import re
import string

from tensorflow import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.utils import pad_sequences

model = tf.keras.models.load_model('models/model_0.84.keras')
with open('models/tokenizer.pickle', 'rb') as handle:
  tokenizer = pickle.load(handle)

def load_stopwords():
    try:
        with open('stopwords.txt', 'r') as f:
            return set(word.strip() for word in f.readlines())
    except FileNotFoundError:
        # If the file doesn't exist, download the stopwords and save them
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        with open('stopwords.txt', 'w') as f:
            for word in stop_words:
                f.write(f"{word}\n")
        return stop_words

# Load stopwords
stop_words = load_stopwords()

# Streamlit UI
st.title("Emotion Detection based on Text")
st.markdown("This App focuses on detecting emotions from textual data using a Bidirectional Long Short-Term Memory (LSTM) neural network. The goal is to classify emotions expressed in text into categories such as joyfulness, sadness, anger, and more. By leveraging the power of LSTM networks, which are particularly effective for sequence prediction tasks, this project aims to provide accurate and efficient emotion classification.")
st.markdown('[Visit my Github Profile](https://github.com/bintangyosua)')
user_input = st.text_area('Please enter your text in English')

# Preprocessing function
def text_preprocess(text):
    # Remove HTML
    text = re.sub(pattern, '', text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', punctuation))
    # Tokenization and stemming, excluding stopwords
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Predefined patterns and stemmer
pattern = re.compile('<.*?>')  # Pattern for removing HTML tags
punctuation = string.punctuation  # Extracting punctuation
ps = PorterStemmer()  # Creating a PorterStemmer object

if st.button("Emotion Prediction"):
    if user_input:
        processed_texts = text_preprocess(user_input)
        sequence = tokenizer.texts_to_sequences([processed_texts])
        padded_sequence = pad_sequences(sequence, maxlen=50, padding='post')
        prediction = model.predict(padded_sequence)
        predicted_label = np.argmax(prediction, axis=1)
        emotion_dict = {
          0: 'Sadness',
          1: 'Joy',
          2: 'Love',
          3: 'Anger',
          4: 'Fear',
          5: 'Surprise'
        }
        st.success(f"Detected Emotion: {emotion_dict[predicted_label[0]]}")
    else:
        st.error("Please enter your text first.")