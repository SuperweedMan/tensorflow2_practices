#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import pandas as pd
import os
import sys
import datetime
import itertools
import time
import io
import tensorflow as tf
import unicodedata
import re
import classical_models.cifar_10_Dataset as cifar_10_DS
import classical_models.ResNet as ResNet

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, tf.keras:
    print(module.__name__, module.__version__)

#%%
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

model = ResNet.ResNet(ResNet.BasicBlock, [3, 4, 6, 3])
x = tf.constant(np.ones([1, 32, 32, 3]))

tf.summary.trace_on(graph=True, profiler=True)
y = model(x)
with writer.as_default():
    tf.summary.trace_export(
        name='test_1',
        step=0,
        profiler_outdir=logdir,
    )

#%%
logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# model = ResNet.ResNet(ResNet.BasicBlock, [3, 4, 6, 3])
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0


model.fit(
    train_images,
    train_labels,
    batch_size=64,
    epochs=5,
    callbacks=[tensorboard_callback])
