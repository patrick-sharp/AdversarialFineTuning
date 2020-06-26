import h5py
import tensorflow as tf

model = tf.keras.applications.vgg16.VGG16(
    weights=None, 
    include_top=True, 
    classes=10,
    input_shape=(32,32,3)
)

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

model.load_weights('../VGG16.h5')

