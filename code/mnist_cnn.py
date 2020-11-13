from __future__ import print_function
import keras
from mnist_datasets import load_data
from keras import models
from keras.models import Sequential
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from uaitrain.arch.tensorflow import uflag

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags

flags.DEFINE_integer("epochs", 5, "Number of epochs")

batch_size = 64
num_classes = 10
epochs = FLAGS.epochs

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data(FLAGS.data_dir)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

conv_base=VGG16(weights='imagenet', include_top=False, input_shape=(48,48,3))
conv_base.trainable=False
model=models.Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
# layer 14
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
# layer 15
model.add(Dense(10, activation="softmax"))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="rmsprop",
              metrics=['accuracy'])
tbCallBack = keras.callbacks.TensorBoard(log_dir=FLAGS.log_dir, histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tbCallBack])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_path = FLAGS.output_dir + '/mnist_model.h5'
model.save(model_path)
