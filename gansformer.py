# this file takes the weights for the model created with transformer.py and
# uses adversarial fine-tuning to improve it

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

import time
import numpy as np
import matplotlib.pyplot as plt

def load_pretrained_transformer():
  pass

def make_lstm_discriminator_model():
  model = tf.keras.Sequential()

  model.add(layers.LSTM(400))
  model.add(layers.Dropout(0.3))

  model.add(layers.LSTM(400))
  model.add(layers.Dropout(0.3))

  # binary classification: actual or generated?
  model.add(layers.Dense(1))

  return model

# define the combined generator and discriminator model, for updating the generator
def make_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = tf.keras.Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = tf.optimizers.Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load and prepare real examples from the training set
def load_real_samples():
	# load cifar10 dataset
	(trainX, _), (_, _) = load_data()
	# convert from unsigned ints to floats
	X = trainX.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X