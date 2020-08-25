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
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#%%
print("Shape: ", train_imgs[0].shape)
print("Label: ", train_labels[0], "->", class_names[train_labels[0]])

# 显示单个图像
# tf.summary.image()包含(batch_size, height, width, channels)四个维度
img = np.reshape(train_imgs[0], (-1, 28, 28, 1))
logdir = 'logs/train_data/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
file_writer = tf.summary.create_file_writer(logdir)
with file_writer.as_default():
    tf.summary.image('Training data', img, step=0)

#%%
# 显示多个图像
imgs = np.reshape(train_imgs[0:25], (-1, 28, 28, 1))
file_writer = tf.summary.create_file_writer(logdir)
with file_writer.as_default():
    tf.summary.image('25 images', imgs, max_outputs=15, step=0)

#%%
# 整体的应用

# 创建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 创建DataSet
labels_onehot = tf.one_hot(train_labels, 10)
DS = tf.data.Dataset.from_tensor_slices((train_imgs, labels_onehot, train_labels))
# 超参
epochs = 10
batch_size = 100
optimizer = tf.optimizers.Adam()
cce = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
DS = DS.batch(batch_size)


# 画混淆矩阵
def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


# matplotlib图像转为PNG图像张量
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
file_writer = tf.summary.create_file_writer(logdir)
#%%
for epoch in range(epochs):
    for imgs, labels_onehot, labels in DS:
        with tf.GradientTape() as tape:
            pred_output = model(imgs)
            current_loss = cce(labels, pred_output)
        grads = tape.gradient(current_loss, model.trainable_variables)
        grads_and_variables = zip(grads, model.trainable_variables)
        optimizer.apply_gradients(grads_and_variables)
    print('epoch 1')
    test_pred_raw = model.predict(test_imgs)
    test_pred = np.argmax(test_pred_raw, axis=1)
    cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
    print('confusion_matrix')
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)