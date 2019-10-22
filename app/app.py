import os
import streamlit as st
import cv2
import time
import pandas as pd
import numpy as np
import urllib.request
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from car_app_utils import *

# DIR = "data/folders"
# _, DIRS, _ = next(os.walk(DIR))
# DIRS = sorted(DIRS)

DIRS =  ['Frontal', 'Frontal right', 'Frontal left', 'Lateral', 'Rear right', 'Rear left', 'Rear'] # folders as the labels

st.header('Predict a car position view!')

# load the model
model = load_model(r'\output\model_new_train.h5')

url_example = 'https://c8.alamy.com/comp/T6GCJA/new-york-ny-usa-30th-apr-2019-delorean-the-car-at-arrivals-for-framing-john-delorean-premiere-at-the-tribeca-film-festival-crosby-street-hotel-new-york-ny-april-30-2019-credit-eli-winstoneverett-collectionalamy-live-news-T6GCJA.jpg'
url = st.text_input('Paste pic link below...(or leave the example)', url_example)

# load image from url
img = urlToImage(url)

# show the image
st.image(img, use_column_width=False, width=250)

# process the image
image = process_images(url)

# run prediction
with st.spinner('Wait for it...'):
    # time.sleep(45)
    def predict_enquadramento(imgs):
        proba = model.predict(imgs)[0]
        # idxs = np.argsort(proba)[::-1][:2] # two highest
        idxs = np.argsort(proba)[::-1][0] # higher prob
        return proba[idxs], idxs

    proba, idxs = predict_enquadramento(image)

    st.subheader('Result:')
    st.write(f'{DIRS[idxs]} view with {round(proba*100, 2)}% probability.')

    st.balloons()
    st.success('Done!')

