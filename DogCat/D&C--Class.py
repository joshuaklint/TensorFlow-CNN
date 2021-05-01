import tensorflow as tf
from tensorflow import keras
import PIL
from PIL import  Image
import  matplotlib.image as img
import matplotlib.pyplot as plt
import os
from  tensorflow.keras.preprocessing import image
from  keras.regularizers import  l1, l2
import  pandas as pd
import numpy as np


basedir = 'C:/Users/lenovo/PycharmProjects/Tensorflow'
traindir = os.path.join(basedir,'train')
testdir = os.path.join(basedir,'test')
validir = os.path.join(basedir, 'validation')


preddir = os.path.join(basedir,'predict')


trainlist = os.listdir(traindir)
testlist = os.listdir(testdir)
predlist = os.listdir(preddir)

#images = []
#for i  in range(1,10):
#    img1  = Image.open('C:/Users/lenovo/PycharmProjects/Tensorflow/predict/'+str(i) + '.jpg')
#    images.append(img1)
#    img1.show()

#for x in images:
#    img2 = Image.open(x)
#   img2.show()

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(150,150,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(filters = 128, kernel_size=(3,3), activation= 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu' ))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(50,activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from  tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
vali_datagen = ImageDataGenerator(rescale=1./255)

train       = train_datagen.flow_from_directory( traindir,   target_size=(150,150),    batch_size=32,
                                          class_mode='binary')

validation = vali_datagen.flow_from_directory(validir,     target_size=(150,150),   batch_size=32 ,
                                              class_mode='binary')


history = model.fit_generator(train, validation_data=validation, epochs=30, validation_steps=50,
                              steps_per_epoch=50)


test = test_datagen.flow_from_directory(testdir, target_size=(150,150), batch_size=32, class_mode='binary')

loss, Acc = model.evaluate_generator(test, steps=120)

pd.DataFrame(history.history).plot(figsize = (6,4))
plt.xlabel('Epoch', fontsize = 13, color = 'red')
plt.ylabel('Loss, Accuracy', fontsize = 13, color = 'red')
plt.title('Dog and Cat clasf Acc, Loss')
plt.show()

print('Test Accurcy : ', Acc)
model.save('Dog&Cat.h5')


model1 = keras.models.load_model('Dog&Cat.h5')

pred = []

for i in range(1,30):
    img3 = image.load_img('C:/Users/lenovo/PycharmProjects/Tensorflow/predict/'+str(i)+'.jpg', target_size=(150,150))
    img3 = image.img_to_array(img3)
    img3 = np.expand_dims(img3, axis=0)
    pred.append(img3)

for x in pred:
    Imgpred = model1.predict(x)
    if Imgpred [0][0] == 1:
        print('Dog')
    else:
        print('Cat')

