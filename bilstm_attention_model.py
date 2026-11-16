import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Lambda, Dropout
import tensorflow.keras.backend as K

def attention_layer(inputs):
    attention = Dense(1, activation='tanh')(inputs)
    attention_weights = Lambda(lambda x: K.softmax(x, axis=1))(attention)
    context_vector = Lambda(lambda x: K.sum(x[0] * x[1], axis=1))([inputs, attention_weights])
    return context_vector

def intent_model(vocab_size, embedding_dim, max_length, num_classes):
    inputs = Input(shape=(max_length,))
    embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
    
    lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding)
    
    attention_vector = attention_layer(lstm)
    
    dense1 = Dense(64, activation='relu')(attention_vector)
    dropout = Dropout(0.5)(dense1)
    output = Dense(num_classes, activation='softmax')(dropout)
    
    model = Model(inputs=inputs, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
