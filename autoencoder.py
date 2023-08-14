from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cv2
import os
import glob
from PIL import Image, ImageTk


def train_model(path, epochs, window, output_label, model_no=0):
    files = glob.glob('output/*')
    for f in files:
        os.remove(f)
    size = 256
    img_channels = 3
    np.random.seed(42)
    img_data = []
    img = cv2.imread(path)
    if img_channels == 1:           #grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img_data.append(img_to_array(img))

    img_array = np.reshape(img_data, (len(img_data), size, size, img_channels))
    img_array = img_array.astype('float32')/255.

    #------------------CALLBACK----------------------------

    class PerformancePlotCallback(Callback):
        def __init__(self, image, model_name):
            self.image = image
            self.model_name = model_name

        def on_epoch_end(self, epoch, logs={}):
            pred = self.model.predict(self.image)
            pred[0] = pred[0] / pred[0].max()
            pred = np.reshape(pred, (len(pred), size, size, 3))
            pred = np.squeeze(pred)
            output_pl = Image.fromarray(np.uint8(pred*255))
            output_tk = ImageTk.PhotoImage(output_pl)
            output_label.configure(image=output_tk)
            output_label.photo = output_tk
            window.update_idletasks()

    #-------------------------------------------------------


    #------------------MODEL--------------------------------

    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='same'))
    # model.add(Conv2D(4, (3,3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2,2), padding='same'))

    # model.add(Conv2D(4, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2,2)))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(3, (3,3), activation='relu', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    #---------------------------------------------------------

    #------------------MODEL1--------------------------------

    model1 = Sequential()
    model1.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    model1.add(MaxPooling2D((2,2), padding='same'))
    model1.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model1.add(MaxPooling2D((2,2), padding='same'))
    model1.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model1.add(MaxPooling2D((2,2), padding='same'))

    model1.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model1.add(UpSampling2D((2,2)))
    model1.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model1.add(UpSampling2D((2,2)))
    model1.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    model1.add(UpSampling2D((2,2)))
    model1.add(Conv2D(3, (3,3), activation='relu', padding='same'))
    model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    #---------------------------------------------------------

    
    if model_no==0:
        create_frames = PerformancePlotCallback(img_array, model_name=model)
        model.fit(img_array, img_array, epochs=epochs, shuffle=True, callbacks=[create_frames])
        pred = model.predict(img_array)
        pred[0] = pred[0] / pred[0].max()
        if img_channels == 1:                   #grayscale image
            pred = np.reshape(pred, (len(pred), size, size))
            matplotlib.image.imsave('output/out.png', pred[0])
        else:
            matplotlib.image.imsave('output/out.png', pred[0])

        print(model.summary())

    elif model_no==1:
        create_frames = PerformancePlotCallback(img_array, model_name=model1)
        model1.fit(img_array, img_array, epochs=epochs, shuffle=True, callbacks=[create_frames])
        pred = model1.predict(img_array)
        pred[0] = pred[0] / pred[0].max()
        if img_channels == 1:                   #grayscale image
            pred = np.reshape(pred, (len(pred), size, size))
            matplotlib.image.imsave('output/out.png', pred[0])
        else:
            matplotlib.image.imsave('output/out.png', pred[0])

        print(model1.summary())