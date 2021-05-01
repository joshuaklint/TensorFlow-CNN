import tensorflow as tf
from tensorflow import keras
import cv2, time
from tensorflow.keras.preprocessing import image
import numpy as np
print(np.__version__)

model = keras.models.load_model('Weather_pred.H5')

import requests

import urllib.request

url = 'https://192.168.43.1:8080/shot.jpg'


img_res = urllib.request.urlopen(url)
img_arr = np.array(bytearray(img_res.read()), dtype=np.uint8)
img = cv2.imdecode(img_arr,-1)

cv2.imshow('joshua', img)
cv2.imwrite('weather.jpg',img)
time.sleep(3)
cv2.waitKey(20000)
cv2.destroyAllWindows()

img = image.load_img('weather.jpg', target_size=(150,150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
weather = ['cloudy', 'rainy', 'shine','sunrise']
print(weather[np.argmax(pred)])

