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
import unicodedata
import re
import pickle

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, tf.keras:
    print(module.__name__, module.__version__)


#%%
def reshape_2_img(img_data, label, label_index):
    img_data = tf.reshape(img_data, [32, 32, 3])
    return img_data, label, label_index


def make_dataset(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    train_value = tf.cast(tf.convert_to_tensor(data_dict[b'data']), dtype=tf.float32) / 255.0
    train_labels = tf.one_hot(tf.convert_to_tensor(data_dict[b'labels']), 10)
    train_labels_index = tf.convert_to_tensor(data_dict[b'labels'])
    return tf.data.Dataset.from_tensor_slices(
        (train_value, train_labels, train_labels_index)).map(reshape_2_img)


file_num = 2


def data_file_loop(path):
    if os.path.exists(path):
        for index in range(file_num):
            yield make_dataset(os.path.join(path, 'data_batch_{}'.format(index + 1)))
    else:
        raise ValueError


#%%
if __name__ == '__main__':
#%%
    for dataset in data_file_loop('../source/cifar-10-batches-py'):
        dataset = dataset.batch(10)
        for train_value, train_label in dataset.take(1):
            print(train_value, train_label)
