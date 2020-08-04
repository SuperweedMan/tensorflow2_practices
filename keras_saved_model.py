#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import preprocessing
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
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) =  fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

scaler = preprocessing.StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

#%%
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
#%%
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy']
              )

#%%
# log_dir = './record/graph_def_and_weights'
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
# output_model_file = os.path.join(log_dir, "fashion _mnist_weights.h5")

# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True, save_weights_only=True),
#     tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
# ]

history = model.fit(
    x_train_scaled, y_train,
    epochs=10,
    validation_data=(x_valid_scaled, y_valid)
)

model.evaluate(x_test_scaled, y_test)

#%%
tf.saved_model.save(model, './record/keras_saved_graph')

#%%
loaded_saved_model = tf.saved_model.load('./record/keras_saved_graph')
print(list(loaded_saved_model.signatures.keys()))
inference = loaded_saved_model.signatures['serving_default']
print(inference)
print(inference.structured_outputs)

#%%
results = inference(tf.constant(x_test_scaled[0:1]))
print(results)
