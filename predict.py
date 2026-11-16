import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the saved model
model_path = 'intent_model.h5'
model = load_model(model_path)

# Load the tokenizer and label encoder
tokenizer_path = 'tokenizer.pkl'
encoder_path = 'label_encoder.pkl'
tokenizer = joblib.load(tokenizer_path)
encoder = joblib.load(encoder_path)

MAX_SEQ_LEN = 100  

# Function to preprocess and vectorize text
def preprocess_text(text, tokenizer, max_seq_len):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_seq_len, padding='post')
    return padded_seq

# Function to predict intent
def predict_intent(text, model, tokenizer, encoder, max_seq_len):
    seq = preprocess_text(text, tokenizer, max_seq_len)
    pred = model.predict(seq)
    intent_index = np.argmax(pred, axis=-1)
    intent = encoder.inverse_transform(intent_index)
    return intent[0]

# Example usage
if __name__ == "__main__":
    user_input = input("Enter your text: ")
    intent = predict_intent(user_input, model, tokenizer, encoder, MAX_SEQ_LEN)
    print(f"Predicted intent: {intent}")
