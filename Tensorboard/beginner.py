#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import datetime
import time
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
model = ResNet.ResNet(ResNet.BasicBlock, [3, 4, 6, 3])

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

epochs = 2
batch_size = 100
optimizer = tf.optimizers.Adam()
cce = tf.keras.losses.CategoricalCrossentropy()
# cce = tf.keras.losses.()
step = 0
for epoch in range(epochs):
    for dataset in cifar_10_DS.data_file_loop('../source/cifar-10-batches-py'):
        dataset = dataset.batch(batch_size)
        for train_values, train_labels, label_index in dataset:
            # print(train_values.shape, train_labels)
            with tf.GradientTape() as tape:
                pred_output = model(train_values)
                current_loss = cce(train_labels, pred_output)
            grads = tape.gradient(current_loss, model.trainable_variables)
            grads_and_variables = zip(grads, model.trainable_variables)
            optimizer.apply_gradients(grads_and_variables)

            train_loss(current_loss)
            train_accuracy(label_index, pred_output)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result(),
                                  test_loss.result(),
                                  test_accuracy.result()))
            with train_summary_writer.as_default():
                step += 1
                tf.summary.scalar('loss', train_loss.result(), step=step)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=step)
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()