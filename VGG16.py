#https://www.kaggle.com/xhlulu/vgg-16-on-cifar10-with-keras-beginner
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical

(train_imgs, train_labs), (test_imgs, test_labs) = tf.keras.datasets.cifar10.load_data()

x_train = preprocess_input(train_imgs)
y_train = to_categorical(train_labs)

x_test = preprocess_input(test_imgs)
y_test = to_categorical(test_labs)

model = tf.keras.applications.vgg16.VGG16(
    weights=None, 
    include_top=True, 
    classes=10,
    input_shape=(32,32,3)
)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     'VGG16.h5', 
#     monitor='val_acc', 
#     verbose=0, 
#     save_best_only=True, 
#     save_weights_only=False,
#     mode='auto'
# )
#
# Train the model
# history = model.fit(
#     x=x_train,
#     y=y_train,
#     validation_split=0.1,
#     batch_size=256,
#     epochs=30,
#     callbacks=[checkpoint],
#     verbose=1
# )

# test the model on 16 random images from the test set
example_imgs_indexes = np.random.choice(range(len(test_imgs)), size=16, replace=False)
example_imgs = test_imgs[example_imgs_indexes]
example_imgs_preprocessed = x_test[example_imgs_indexes]
example_imgs_labels = y_test[example_imgs_indexes]
predictions = model(example_imgs_preprocessed, training=False)

cifar10_categories = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
    ]

fig = plt.figure(figsize=(4,4))
for i in range(predictions.shape[0]):
      predicted_category = cifar10_categories[np.argmax(predictions[i])]
      actual_category = cifar10_categories[np.argmax(example_imgs_labels[i])]
      label = 'pred: ' + predicted_category + ', actual: ' + actual_category

      plt.subplot(4, 4, i+1)
      plt.imshow(example_imgs[i])
      plt.title(label)
      plt.axis('off')

plt.show()

# with open('history.json', 'w') as f:
#     json.dump(history.history, f)

# history_df = pd.DataFrame(history.history)
# history_df[['loss', 'val_loss']].plot()
# history_df[['acc', 'val_acc']].plot()

# model.load_weights('VGG16.h5')
# train_loss, train_score = model.evaluate(x_train, y_train)
# test_loss, test_score = model.evaluate(y_test, y_test)
# print("Train Loss:", train_loss)
# print("Test Loss:", test_loss)
# print("Train F1 Score:", train_score)
# print("Test F1 Score:", test_score)