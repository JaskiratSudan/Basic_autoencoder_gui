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
import sys
import glob
from PIL import Image, ImageTk


def train_model(path, epochs, window, output_label, status_lab, latent_lab, progress_var, progress_bar, input_size, latent_size, output_size):
    files = glob.glob('output/*')
    for f in files:
        os.remove(f)

    size = 256
    img_channels = 3
    latent_dim = (32,32)
    latent_channels = 4

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
    progress_bar.config(maximum=epochs)
    
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
            intermediate_prediction = intermediate_model.predict(img_array) #predicting in the Intermediate Node
            intermediate_prediction = np.squeeze(intermediate_prediction)
            # print(np.shape(intermediate_prediction)) 
             
            image_array = np.asarray(cv2.split(np.squeeze(intermediate_prediction)))
            print("\n\n shape of imagearray: {}".format(image_array.shape))
            canvas = np.zeros((int(latent_dim[0]*(latent_channels**1/2)), int(latent_dim[0]*(latent_channels**1/2))))
            for i in range(latent_channels):
                for row in range(int(latent_channels**1/2)):
                    for col in range(int(latent_channels**1/2)):
                        print(canvas[row*latent_dim[0]:row*latent_dim[0]+latent_dim[0], col*latent_dim[0]:col*latent_dim[0]+latent_dim[0]].shape)
                        # print("canvas shape: {}".format(canvas.shape))
                        canvas[row*latent_dim[0]:row*latent_dim[0]+latent_dim[0], col*latent_dim[0]:col*latent_dim[0]+latent_dim[0]] = image_array[i]
                        # i+=1
                        # plt.imshow(image_array[i])
                        # plt.show()
                        # canvas[0:32, 32:64] = image_array[1]
                        # canvas[32:64, 0:32] = image_array[2]
                        # canvas[32:64, 32:64] = image_array[3]
            latent_pl = Image.fromarray(np.uint8(canvas*255)).resize((200,200))
            latent_tk = ImageTk.PhotoImage(latent_pl)
            latent_lab.configure(image=latent_tk)
            latent_lab.photo = latent_tk
            output_pl = Image.fromarray(np.uint8(pred*255))
            output_tk = ImageTk.PhotoImage(output_pl)
            output_label.configure(image=output_tk)
            output_label.photo = output_tk
            progress_var.set(epoch+1)

            # input_size.configure(text="{} KB".format(sys.getsizeof(img_array)/1024))
            # latent_size.configure(text="{} KB".format(sys.getsizeof(canvas)/1024))
            # output_size.configure(text="{} KB".format(sys.getsizeof(pred)/1024))
            input_size.configure(text="{}".format(img_array.nbytes/1024))
            latent_size.configure(text="{}".format(canvas.nbytes/1024))
            output_size.configure(text="{}".format(pred.nbytes/1024))
            status_lab.configure(text="Training  Model... \n{}/{} epochs done.".format(epoch, epochs))
            window.update_idletasks()

    #-------------------------------------------------------

    #------------------MODEL--------------------------------

    # model = Sequential()
    # model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    # model.add(MaxPooling2D((2,2), padding='same'))
    # model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2,2), padding='same'))
    # model.add(Conv2D(4, (3,3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2,2), padding='same'))
   
    # model.add(Conv2D(4, (3,3), activation='relu', padding='same'))
    # model.layers[-1]._name = "latent_space"
    # model.add(UpSampling2D((2,2)))
    # model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2,2)))
    # model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(size, size, 3)))
    # model.add(UpSampling2D((2,2)))
    # model.add(Conv2D(3, (3,3), activation='relu', padding='same'))

    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', padding='same', strides=2, input_shape=(size, size, 3)))
    model.add(Dropout(0.2))
    # model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', strides=2, padding='same'))
    model.add(Dropout(0.2))
    # model.add(MaxPooling2D((2,2), padding='same'))
    model.add(Conv2D(latent_channels, (3,3), activation='relu', strides=2, padding='same'))
    # model.add(MaxPooling2D((2,2), padding='same'))
    # model.add(Conv2D(4, (3,3), activation='relu', strides=2, padding='same'))
    # model.add(MaxPooling2D((2,2), padding='same'))
    # model.add(Conv2D(4, (3,3), activation='relu', strides=2, padding='same'))
    model.layers[-1]._name = "latent_space"

    # model.add(Conv2D(4, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2,2)))
    model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(3, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2,2)))
    # model.add(Conv2D(3, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D((2,2)))
    # model.add(Conv2D(3, (3,3), activation='relu', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    #---------------------------------------------------------

    model.summary()

    layer_output=model.get_layer('latent_space').output  #get the Output of the Layer
    intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
    
    create_frames = PerformancePlotCallback(img_array, model_name=model)
    model.fit(img_array, img_array, epochs=epochs, shuffle=True, callbacks=[create_frames])
    pred = model.predict(img_array)
    pred[0] = pred[0] / pred[0].max()

    intermediate_prediction=intermediate_model.predict(img_array) #predicting in the Intermediate Node
    intermediate_prediction = np.squeeze(intermediate_prediction)
    print(np.shape(intermediate_prediction))  
    image_array = np.asarray(cv2.split(np.squeeze(intermediate_prediction)))
    canvas = np.zeros((64, 64))
    canvas[0:32, 0:32] = image_array[0]
    canvas[0:32, 32:64] = image_array[1]
    canvas[32:64, 0:32] = image_array[2]
    canvas[32:64, 32:64] = image_array[3]
    canvas[0] = canvas[0] / canvas[0].max()

    if img_channels == 1:                   #grayscale image
        pred = np.reshape(pred, (len(pred), size, size))
        matplotlib.image.imsave('output/out_e{}.png'.format(epochs), pred[0])
    else:
        matplotlib.image.imsave('output/out_e{}.png'.format(epochs), pred[0])
    matplotlib.image.imsave('output/latent_space_e{}.png'.format(epochs), canvas)
    print(model.summary())

    matplotlib.image.imsave('output/input.png', np.squeeze(img_array))