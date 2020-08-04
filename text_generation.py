#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
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
input_fileppath = './source/shakespeare.txt'
text = open(input_fileppath, 'r').read()

print(len(text))
print(text[0:100])

#%%
vocab = sorted(set(text))
print(len(vocab))
print(vocab)

char2idx = {char: idx for idx, char in enumerate(vocab)}
print(char2idx)

idx2char = np.array(vocab)
print(idx2char)

text_as_int = np.array([char2idx[c] for c in text])
print(text_as_int[0:10])
print(text[0:10])


#%%
def split_input_target(id_text):
    """abcde -> abcd, bcde"""
    return id_text[0:-1], id_text[1:]


char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
seq_length = 100
seq_dataset = char_dataset.batch(seq_length + 1, drop_remainder = True)

for ch_id in char_dataset.take(2):
    print(ch_id, idx2char[ch_id.numpy()])

for seq_id in seq_dataset.take(2):
    print(seq_id)
    print(repr(''.join(idx2char[seq_id.numpy()])))


#%%
seq_dataset = seq_dataset.map(split_input_target)
for item_input, item_output in seq_dataset.take(2):
    print(item_input.numpy(), item_output.numpy())
