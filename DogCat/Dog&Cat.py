import tensorflow as tf
from tensorflow import keras
import cv2, time
from tensorflow.keras.preprocessing import image
import numpy as np

model = keras.models.load_model('Dog&Cat.h5')

import requests
import numpy as np
import urllib.request

url = 'http://192.168.137.187:38080/shot.jpg'


img_res = urllib.request.urlopen(url)
img_arr = np.array(bytearray(img_res.read()), dtype=np.uint8)
img = cv2.imdecode(img_arr,-1)

cv2.imshow('Animal', img)
cv2.imwrite('Animal1.jpg',img)
time.sleep(3)
cv2.waitKey(20000)
cv2.destroyAllWindows()

img = image.load_img('Animal1.jpg', target_size=(150,150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

Imgpred = model.predict(img)
if Imgpred[0][0] == 1:
    print('Dog')
else:
    print('Cat')

