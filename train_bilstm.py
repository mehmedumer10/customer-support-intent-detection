import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load the intent dataset
intent_dataset_path = 'intent_dataset.csv'
intent_df = pd.read_csv(intent_dataset_path, delimiter=';')

# Extract the necessary columns
texts = intent_df['instruction'].values
categories = intent_df['category'].values
intents = intent_df['intent'].values

# Parameters
vocab_size = 4636
embedding_dim = 300
max_length = 100
num_classes = len(set(intents))

# Tokenize texts
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(intents)
labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Save tokenizer and label encoder
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def create_model(vocab_size, embedding_dim, max_length, num_classes):
    inputs = Input(shape=(max_length,))
    embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    global_pooling = GlobalAveragePooling1D()(lstm)
    dense = Dense(128, activation='relu')(global_pooling)
    dropout = Dropout(0.5)(dense)
    outputs = Dense(num_classes, activation='softmax')(dropout)
    
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = create_model(vocab_size, embedding_dim, max_length, num_classes)
model.summary()

# Training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Save the model
model.save('intent_model.h5')