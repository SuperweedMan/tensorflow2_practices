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
import classical_models.cifar_10_Dataset as cifar_10_Dataset

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, tf.keras:
    print(module.__name__, module.__version__)


#%%
class BasicBlock(tf.keras.Model):
    expansion = 1
    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        # ResNet会使用strides=2做下采样，因此Conv_1使用可变参数strides
        self.Conv_1 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=strides,
                                            padding='same', use_bias=False)
        self.BN_1 = tf.keras.layers.BatchNormalization()
        # 在一个BasicBlock里面，第二次卷积不需要改变特征图大小，因此stride固定为1
        self.Conv_2 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                             padding='same', use_bias=False)
        self.BN_2 = tf.keras.layers.BatchNormalization()
        # 直连的实现
        # 当下采样时，block的输出输出特征图大小不同，使用1*1卷积
        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.expansion * out_channels, kernel_size=1,
                                       strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            # 若输入shape等于输出shape，则直连
            self.shortcut = lambda x, _:x

    @tf.function
    def call(self, x, training=True):
        out = tf.nn.relu(self.BN_1(self.Conv_1(x), training=training))
        out = self.BN_2(self.Conv_2(out), training=training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


#%%
class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.Conv_1 = tf.keras.layers.Conv2D(64, 3, 1, padding='same', use_bias=False)
        self.BN_1 = tf.keras.layers.BatchNormalization()

        self.blocks_1 = self._make_blocks(block, 64, num_blocks[0], stride=1)
        self.blocks_2 = self._make_blocks(block, 128, num_blocks[0], stride=2)
        self.blocks_3 = self._make_blocks(block, 256, num_blocks[0], stride=2)
        self.blocks_4 = self._make_blocks(block, 512, num_blocks[0], stride=2)

        self.avg_pool2D = tf.keras.layers.AveragePooling2D(4)
        self.linear = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def _make_blocks(self, block, out_channels, num_blocks, stride):
        # 每一个block集合，都由多个BasicBlock组成，且只有第一个步长为2做下采样
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # 下一层的特征图通道数与上一层要相同
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    @tf.function
    def call(self, x, training=True):
        out = tf.nn.relu(self.BN_1(self.Conv_1(x), training))
        out = self.blocks_1(out, training=training)
        out = self.blocks_2(out, training=training)
        out = self.blocks_3(out, training=training)
        out = self.blocks_4(out, training=training)

        out = self.avg_pool2D(out)
        # out = tf.keras.layers.Flatten()
        out = tf.reshape(out, (out.shape[0], -1))
        out = self.linear(out)
        return out


#%%
if __name__ == '__main__':
    #%%
    model = ResNet(BasicBlock, [3, 4, 6, 3])  # ResNet34
    # model.compile(optimizer='Adam', loss='CategoricalCrossentropy', metrics=['accuarcy'])
    #%%
    epochs = 20
    batch_size = 100
    optimizer = tf.optimizers.Adam()
    cce = tf.keras.losses.CategoricalCrossentropy()
    for dataset in cifar_10_Dataset.data_file_loop('../source/cifar-10-batches-py'):
        dataset = dataset.batch(batch_size)
        for train_values, train_labels in dataset:
            # print(train_values.shape, train_labels)
            with tf.GradientTape() as tape:
                pred_output = model(train_values)
                current_loss = cce(train_labels, pred_output)
            grads = tape.gradient(current_loss, model.trainable_variables)
            grads_and_variables = zip(grads, model.trainable_variables)
            optimizer.apply_gradients(grads_and_variables)
            print(current_loss)