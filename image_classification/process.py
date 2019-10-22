import skimage
import os
import cv2
import random
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from numpy import array
from keras.preprocessing.image import img_to_array
from app.car_app_utils import imread
import argparse

# parser argument definition

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_enq_path", help = "Path to Enquadramento images folder", required = True)

parser.add_argument("-o", "--output_path", help = "Path to output data", required = True)

args = parser.parse_args()
input_enq_path = args.input_enq_path
output_path = args.output_path

dir = input_enq_path
fld = input_enq_path.split("/")[-1]

args = {fld: dir}

path, dirs, files = next(os.walk(dir))
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args[fld])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    #image = cv2.imread(imagePath)
    image = imread(imagePath)

    if((image.shape)[-1])==3:
        image = cv2.resize(image, (64, 64))
        image=img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# integer encode
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

with open(output_path + '/data.pickle', "wb") as f:
    pickle.dump(data, f)

with open(output_path + '/labels.pickle', "wb") as f:
    pickle.dump(labels, f)
