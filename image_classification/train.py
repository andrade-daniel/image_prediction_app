import tensorflow as tf
import os
import cv2
import pickle
import random
import numpy as np
from  numpy import array
from time import time
from imutils import paths
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import initializers
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import TensorBoard
import argparse

# parser argument definition

parser = argparse.ArgumentParser()

# parser.add_argument("-i", "--input_enq_path", help = "Path to images folder", required = False,
# default="/images")

parser.add_argument("-o", "--output_path", help = "Path to output data", required = True)

parser.add_argument("-mname", "--model_arch", help = "Model architecture used for training", required = False,
default = "cnn_first_model_enq")

parser.add_argument("-e", "--n_epochs", help = "Number of epochs to train the model", required = False,
default = 100)

parser.add_argument("-c", "--use_callbacks",
help = "To use callbacks/early stopping in training: USE_EARLY_STOPPING/NO_EARLY_STOPPING", required = False,
default = "NO_EARLY_STOPPING")

args = parser.parse_args()

# input_enq_path = args.input_enq_path
output_path = args.output_path
model_arch = args.model_arch
n_epochs = int(args.n_epochs)
use_callbacks = args.use_callbacks

# # to get number of classes from the folders
# dir = input_enq_path
#
# _, dirs, _ = next(os.walk(dir))
# dirs = sorted(dirs)

# loading processed data and labels

data = pickle.load(
    open(
        output_path + "/data.pickle",
        "rb",
    )
)

labels = pickle.load(
    open(
        output_path + "/labels.pickle",
        "rb",
    )
)

# Training parameters
batch_size = 256
n_epochs = n_epochs
n_classes = len(np.unique(labels))
init_lr = 1e-4


# CNN model

def cnn_first_model_enq(n_classes):

    model = Sequential()

    # Convolution Layers
    model.add(Conv2D(16, 3, activation='relu', input_shape=(64, 64, 3), padding='same')) # changing weights initializer to He normal? (instead of Glorot)
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(Dropout(0.30))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(Dropout(0.35))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(Dropout(0.45))

    # Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.7))
    # softmax classifier
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())

    return model

def train_evaluate(model_arch_sel, n_epochs, init_lr, batch_size, data, labels):

    (train_images, test_images, train_labels, test_labels) = train_test_split(data,labels, test_size=0.25, random_state=42)

    with open(output_path + '/train_images.pickle', "wb") as f:
        pickle.dump(train_images, f)

    with open(output_path + '/train_labels.pickle', "wb") as f:
        pickle.dump(train_labels, f)

    with open(output_path + '/test_images.pickle', "wb") as f:
        pickle.dump(test_images, f)

    with open(output_path + '/test_labels.pickle', "wb") as f:
        pickle.dump(test_labels, f)

    # convert the labels from integers to vectors
    train_labels = to_categorical(train_labels,num_classes=n_classes)
    test_labels = to_categorical(test_labels,num_classes=n_classes)

    # model
    model = model_arch_sel(n_classes)

    # Optimizer
    optimizer = Adam(lr=init_lr, decay=init_lr / n_epochs)

    tensorboard = TensorBoard(log_dir="/log/{}".format(time()))

    # Compile
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['binary_accuracy', 'categorical_accuracy'])

    # train the network
    print("[INFO] training network...")

    if use_callbacks == "NO_EARLY_STOPPING":

        # os.mkdir(output_path + "/log")

        Historic = model.fit(train_images,
                    train_labels,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  validation_data=(test_images, test_labels)
                  # , callbacks=[tensorboard]
                  )
    else:

        callbacks = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

        Historic = model.fit(train_images,
                    train_labels,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  validation_data=(test_images, test_labels),
                  callbacks=[callbacks]
                  )

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(output_path + "/model_new_train.h5")

    with open(output_path + '/historico.pickle', "wb") as f:
        pickle.dump(Historic, f)

    # evaluate

    train_acc_eval = model.evaluate(train_images, train_labels, verbose=0)

    with open(output_path + '/train_acc_eval.pickle', "wb") as f:
        pickle.dump(train_acc_eval, f)

    test_acc_eval = model.evaluate(test_images, test_labels, verbose=0)

    with open(output_path + '/test_acc_eval.pickle', "wb") as f:
        pickle.dump(test_acc_eval, f)

    print('Train eval: %.3f, Test eval: %.3f' % (train_acc_eval[2], test_acc_eval[2]))


# Models dictionary (only one used here)

model_dict = {"cnn_first_model_enq": cnn_first_model_enq}

model_arch_sel = model_dict[model_arch]


# train and evaluate the model

train_evaluate(model_arch_sel=model_arch_sel,
n_epochs=n_epochs,
init_lr=init_lr,
batch_size=batch_size,
data=data,
labels=labels)
