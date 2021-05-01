import tensorflow as tf
from tensorflow import keras
import cv2, time
from tensorflow.keras.preprocessing import image
import numpy as np

model = keras.models.load_model('Weather_pred.H5')

cap =  cv2.VideoCapture(0)

check , frame = cap.read()
print(check)
print(frame)


cv2.imwrite('joshua.jpg', frame)


time.sleep(5)
cv2.waitKey(2000)
cap.release()
cv2.destroyAllWindows()

img = image.load_img('joshua.jpg', target_size=(150,150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
weather = ['cloudy', 'rainy', 'shine','sunrise']
print(weather[np.argmax(pred)])

tf.tr