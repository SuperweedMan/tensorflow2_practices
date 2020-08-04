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
@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)


#%%
cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None], tf.int32)
)
print(cube_func_int32)
print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5], tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1, 2, 3])))
print(cube_func_int32.graph)

#%%
to_export = tf.Module()
to_export.cube = cube
tf.saved_model.save(to_export, './record/signature_to_saved_model')

#%%
imported = tf.saved_model.load('./record/signature_to_saved_model')
print(imported.cube(tf.constant([2])))
