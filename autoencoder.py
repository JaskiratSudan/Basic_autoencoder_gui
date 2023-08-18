from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cv2
import os
import io
import glob
from PIL import Image, ImageTk


def train_model(path, epochs, window, output_label, status_lab, latent_lab, num_rows, num_cols):
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
            layer_output=model.get_layer('latent_space').output  #get the Output of the Layer
            intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output) #Intermediate model between Input Layer and Output Layer which we are concerned about 
            intermediate_prediction=intermediate_model.predict(img_array) #predicting in the Intermediate Node
            # print(np.shape(intermediate_prediction))  
            image_array = np.asarray(cv2.split(np.squeeze(intermediate_prediction)))
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(3, 3))
            fig.suptitle("Latent Space")
            for i in range(min(num_rows * num_cols, image_array.shape[0])):
                row = i // num_cols
                col = i % num_cols
                axs[row, col].imshow(image_array[i], cmap='gray')
                axs[row, col].axis('off')
            # Adjust spacing between subplots
            # plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout(pad=1)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            im = Image.open(img_buf)
            latent_tk = ImageTk.PhotoImage(im)
            latent_lab.configure(image=latent_tk)
            latent_lab.photo = latent_tk
            fig.clear()
            plt.close(fig)
            output_pl = Image.fromarray(np.uint8(pred*255))
            output_tk = ImageTk.PhotoImage(output_pl)
            output_label.configure(image=output_tk)
            output_label.photo = output_tk
    
            status_lab.configure(text="Training default model... \n{}/{} epochs done.".format(epoch, epochs))
            window.update_idletasks()

    #-------------------------------------------------------

    #------------------MODEL--------------------------------

    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Conv2D(4, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='same'))
   
    model.add(Conv2D(4, (3,3), activation='relu', padding='same'))
    model.layers[-1]._name = "latent_space"
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(3, (3,3), activation='relu', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    #---------------------------------------------------------

    #------------------MODEL1--------------------------------

    # model1 = Sequential()
    # model1.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    # model1.add(MaxPooling2D((2,2), padding='same'))
    # model1.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    # model1.add(MaxPooling2D((2,2), padding='same'))
    # model1.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    # model1.add(MaxPooling2D((2,2), padding='same'))

    # model1.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    # model1.layers[-1]._name = "Latent_space"
    # model1.add(UpSampling2D((2,2)))
    # model1.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    # model1.add(UpSampling2D((2,2)))
    # model1.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    # model1.add(UpSampling2D((2,2)))
    # model1.add(Conv2D(3, (3,3), activation='relu', padding='same'))
    # model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    #---------------------------------------------------------

    
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

    # elif model_no==1:
    #     create_frames = PerformancePlotCallback(img_array, model_name=model1)
    #     model1.fit(img_array, img_array, epochs=epochs, shuffle=True, callbacks=[create_frames])
    #     pred = model1.predict(img_array)
    #     pred[0] = pred[0] / pred[0].max()
    #     if img_channels == 1:                   #grayscale image
    #         pred = np.reshape(pred, (len(pred), size, size))
    #         matplotlib.image.imsave('output/out.png', pred[0])
    #     else:
    #         matplotlib.image.imsave('output/out.png', pred[0])

    #     print(model1.summary())