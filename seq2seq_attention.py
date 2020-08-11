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
import unicodedata
import re

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, tf.keras:
    print(module.__name__, module.__version__)

#%%
# 定义文件路径
spa_en_file_path = './source/spa-eng/spa.txt'


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


en_sentence = 'Then what?'
sp_sentence =  '¿Entonces qué?'

print(unicode_to_ascii(en_sentence))
print(unicode_to_ascii(sp_sentence))


def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())
    # 标点符号前后加空格
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    # 多余的空格变成一个空格
    s = re.sub(r'[" "]+', " ", s)
    # 除了标点符号与字母，都为空格
    s = re.sub(r'[^a-zA-Z?.!,¿]', " ", s)
    # 去掉前后空格
    s = s.rstrip().strip()
    s = '<start> ' + s + ' <end>'
    return s


print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence))


#%%
def parse_data(filename):
    lines = open(filename, encoding='UTF-8').read().strip().split('\n')
    sentence_pairs = [line.split('\t') for line in lines]
    preprocessed_sentence_pairs = [
        (preprocess_sentence(en), preprocess_sentence(sp)) for en, sp, *_ in sentence_pairs]
    return zip(*preprocessed_sentence_pairs)


en_dataset, sp_dataset = parse_data(spa_en_file_path)
print(en_dataset[-1])
print(sp_dataset[-1])


#%%
def tokenizer(lang):
    lang_tokenlizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=None, filters='', split=' '
    )
    lang_tokenlizer.fit_on_texts(lang)
    tensor = lang_tokenlizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenlizer


input_tensor, input_tokenizer = tokenizer(sp_dataset[0:30000])
output_tensor, output_tokenzier = tokenizer(en_dataset[0:30000])


def max_length(tensor):
    return max(len(t) for t in tensor)


print(max_length(output_tensor))
print(max_length(input_tensor))

#%%
from sklearn.model_selection import train_test_split
input_train, input_eval, output_train, output_eval = train_test_split(
    input_tensor, output_tensor, test_size=0.2)

print(len(input_train), len(input_eval))


#%%
def convert(example, tokenizer):
    for t in example:
        if t != 0:
            print('%d --> %s' % (t, tokenizer.index_word[t]))


convert(input_train[0], input_tokenizer)
print()
convert(output_train[0], output_tokenzier)


#%%
def make_dateset(input_tensor, output_tensor, batch_size, epochs, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
    if shuffle:
        dataset = dataset.shuffle(30000)
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset


batch_size = 64
epochs = 20

train_dataset = make_dateset(input_train, output_train, batch_size, epochs, True)
eval_dataset = make_dateset(input_eval, output_eval, batch_size, 1, False)

for x, y in train_dataset.take(1):
    print(x, y)

#%%
embedding_units = 256
units = 1024
input_vocab_size = len(input_tokenizer.word_index) + 1
ouptut_vocab_size = len(output_tokenzier.word_index) + 1
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_units, encoding_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoding_units = encoding_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_units)
        self.gru = tf.keras.layers.GRU(self.encoding_units, return_sequences=True,
                                       return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoding_units))


encoder = Encoder(input_vocab_size, embedding_units, units, batch_size)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(x, sample_hidden)

print(sample_output.shape)
print(sample_hidden.shape)
