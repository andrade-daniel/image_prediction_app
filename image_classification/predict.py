import numpy as np
import imutils
import pickle
import cv2
import matplotlib.pyplot as plt
import os
import PIL.Image
import skimage
import random
import os
from imutils import paths
from collections import defaultdict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    accuracy_score,
    precision_score,
    f1_score
)
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from app.car_app_utils import imread
import argparse

random.seed(42)

# parser argument definition

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode",
default = 'prediction',
help = "Modes to use: test/prediction",
required = False)

parser.add_argument("-p", "--to_predict", help = "Predict: one/many", required = False)

parser.add_argument("-i", "--input_path", help = "Path to input data",
required = False) # TO BE CHANGED TO "required = True", if test dataset is not pickled previously

parser.add_argument("-o", "--output_path", help = "Path to output data", required = True)

parser.add_argument("-mdl", "--model", help = "Model path to load", required = False,
# default = "/datadrive/tensorflow_models/daniel_enquadramento/first_model/model_carro_100.h5")
# default = "C:/Users/daniel.andrade/Documents/DANIEL/fid/folders/model_carro.h5")
default = "C:/Users/daniel.andrade/Documents/DANIEL/fid/output/model_new_train.h5")

args = parser.parse_args()

print("mode: {} \nto_predict: {} \ninput_path: {} \noutput_path: {} \nmodel: {}".format(
        args.mode,
        args.to_predict,
        args.input_path,
        args.output_path,
        args.model
        ))

mode = args.mode
to_predict = args.to_predict
input_path = args.input_path
output_path = args.output_path
model_path = args.model

assert mode in ["test", "prediction"]

def process_images(imgs):
    image = imread(imgs)
    image = cv2.resize(image, (64, 64))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_enquadramento(imgs):
    probabs = model.predict(imgs)[0]
    # idxs = np.argsort(probabs)[::-1][:2] # two highest
    idxs = np.argsort(probabs)[::-1][0] # higher prob
    return probabs, idxs


# get folder names

# DIR = "data/folders"
# _, DIRS, _ = next(os.walk(DIR))
# DIRS = sorted(DIRS)

DIRS =  ['Frontal', 'Frontal right', 'Frontal left', 'Lateral', 'Rear right', 'Rear left', 'Rear'] # folders as the labels

# load the model
model = load_model(model_path)

# load images

if to_predict == 'one':

    image = process_images(input_path)

    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]

    classes = dict(zip(DIRS, range(len(DIRS)+1)))
    decode_classes = {v: k for k, v in classes.items()}

    pred_classes = [decode_classes[idx] for idx in idxs]

    one_label_predicted = "{}: {:.4f}%".format(pred_classes[0], proba[0])
    print(one_label_predicted)

    with open(output_path + '/one_label_predicted.pickle', "wb") as f:
        pickle.dump(one_label_predicted, f)

else: # to predict 'many' images

    if mode == 'prediction':

        # grab the image paths
        imagePaths = sorted(list(paths.list_images(input_path)))

        images_all = [process_images(k) for k in imagePaths]

        probabs_all, idxs_all = map(list, zip(*[predict_enquadramento(k) for k in images_all]))

        with open(output_path + '/probabs_all.pickle', "wb") as f:
            pickle.dump(probabs_all, f)

        with open(output_path + '/idxs_all.pickle', "wb") as f:
            pickle.dump(idxs_all, f)

        classes = dict(zip(DIRS, range(len(DIRS)+1)))
        decode_classes = {v: k for k, v in classes.items()}

        pred_classes = [decode_classes[idx] for idx in idxs_all]

        with open(output_path + '/pred_classes.pickle', "wb") as f:
            pickle.dump(pred_classes, f)

    else: # mode is 'test'

        '''
        bulk of data here is the test dataset created in training
        '''

        test_images = pickle.load(
            open(
                output_path + "/test_images.pickle",
                "rb",
            )
        )

        test_labels = pickle.load(
            open(
                output_path + "/test_labels.pickle",
                "rb",
            )
        )

        y_pred = model.predict(test_images, batch_size=64, verbose=1)

        y_pred_bool = np.argmax(y_pred, axis=1)
        classif_report = classification_report(test_labels, y_pred_bool, target_names=sorted(DIRS), output_dict=True)

        classif_report_df = pd.DataFrame(classif_report).transpose()

        with open(output_path + '/classif_report_df.pickle', "wb") as f:
            pickle.dump(classif_report_df, f)

        f1_score = f1_score(test_labels, y_pred_bool, average='micro')

        with open(output_path + '/f1_score.pickle', "wb") as f:
            pickle.dump(f1_score, f)

        print("Classification report: \n", (classif_report_df))
        print("f1-score micro avg:",(f1_score))
