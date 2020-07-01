# this file takes the weights for the model created with transformer.py and
# uses adversarial fine-tuning to improve it

import tensorflow as tf
from tensorflow.keras import layers

import time
import numpy as np
import matplotlib.pyplot as plt

def make_discriminator_model():
  model = tf.keras.Sequential()

  model.add(layers.LSTM(400))
  model.add(layers.Dropout(0.3))

  model.add(layers.LSTM(400))
  model.add(layers.Dropout(0.3))

  # binary classification: actual or generated?
  model.add(layers.Dense(1))

  return model