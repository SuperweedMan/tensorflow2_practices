#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import  sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, tf.keras:
    print(module.__name__, module.__version__)

#%%
imdb = tf.keras.datasets.imdb
vocab_size = 10000
index_from = 3
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=vocab_size, index_from=index_from)

#%%
print(train_data[0], train_labels[0])
print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)
print(len(train_data[0]), len(train_data[1]))
#%%
word_index = imdb.get_word_index()
print(len(word_index))
print(word_index)
word_index = {k:(v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<END>'] = 3

reverse_word_index = dict(
    [(value, key) for key, value in word_index.items()]
)


def decode_review(test_ids):
    return ' '.join([reverse_word_index.get(word_id, "<UNK>") for word_id in test_ids])


#%%
max_length = 500

train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_data,
    value = word_index['<PAD>'],
    padding = 'post',
    maxlen = max_length
)
test_data = tf.keras.preprocessing.sequence.pad_sequences(
    test_data,
    value = word_index['<PAD>'],
    padding = 'post',
    maxlen = max_length
)

#%%
embedding_dim = 16
batch_size = 128

model = tf.keras.models.Sequential([
    # 1.define matrix: [vocab_size, embedding_dim]
    # 2.[1,2,3,4..], max_length * embedding_dim
    # 3.batch_size * max_length * embedding_dim
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                             input_length=max_length),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.SimpleRNN(units=64, return_sequences=True)),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.SimpleRNN(units=64, return_sequences=False)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

#%%
history = model.fit(
    train_data, train_labels,
    epochs=30,
    batch_size=batch_size,
    validation_split=0.2
)

#%%
model.evaluate(
    test_data, test_labels,
    batch_size=batch_size
)