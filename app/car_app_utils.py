import numpy as np
import skimage
import cv2
import urllib.request
from keras.preprocessing.image import img_to_array

def imread(photo_path):
    image = skimage.io.imread(photo_path)
    if len(image.shape) in [1,4]:
        return image[0]         # images with MPO format
    elif len(image.shape) == 2:
        return image.reshape(image.shape + (1,))[:,:,[0,0,0]] # grey images = (height, width) ---> adding dimension
    else:
        return image[:,:,0:3] # removing transparency

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

def urlToImage(url):
    # download image,convert to a NumPy array,and read it into opencv
    resp = urllib.request.urlopen(url)
    img = np.asarray(bytearray(resp.read()),dtype="uint8")
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)

    return img
