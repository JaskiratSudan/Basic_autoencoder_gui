from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
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

    
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
    
def ratioFunction(num1, num2):
    int(num1) 
    int(num2) 
    ratio12 = num1/num2
    return int(ratio12)


def train_model(path, epochs, window, output_label, status_lab, latent_lab, progress_var, progress_bar, input_size, latent_size, output_size, psnr, comp_ratio, model_var):
    window.update_idletasks()
    files = glob.glob('output/*')
    for f in files:
        os.remove(f)

    size = 256
    model_size = 256
    latent_dim = (32,32)
    img_channels = 3

    if model_var==256:
        model = tf.keras.models.load_model(
        'models/s256_e100_i2100_ch3reconstruction',
        custom_objects=None,
        compile=True)
        model_size = 256
        latent_dim = (32,32)
        latent_channels = 8

    elif model_var==512:
        model = tf.keras.models.load_model(
        'models/Final_512',
        custom_objects=None,
        compile=True)
        model_size = 512
        latent_dim = (64,64)
        latent_channels = 8

    print("MODEL: ", model_var)

    np.random.seed(42)
    img_data = []
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (model_size, model_size))
    img_data.append(img_to_array(img))

    img_array = np.reshape(img_data, (len(img_data), model_size, model_size, img_channels))
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
            pred = np.reshape(pred, (len(pred), model_size, model_size, img_channels))
            pred = np.squeeze(pred)
            intermediate_prediction = intermediate_model.predict(img_array) #predicting in the Intermediate Node
            intermediate_prediction = np.squeeze(intermediate_prediction)
            # print(np.shape(intermediate_prediction)) 
             
            image_array = np.asarray(cv2.split(np.squeeze(intermediate_prediction)))

            intermediate_shape = intermediate_prediction.shape
            if len(intermediate_shape) == 3:
                latent_channels = intermediate_shape[-1]
            elif len(intermediate_shape) == 2:
                # Assuming the shape is (height, width), and there's only one channel
                latent_channels = 1

            canvas_rows = int(np.ceil(latent_channels / 4))  # Calculate the number of rows needed
            canvas_cols = min(latent_channels, 4)           # Maximum 4 columns
            canvas = np.zeros((latent_dim[0] * canvas_rows, latent_dim[1] * canvas_cols))
            for i in range(latent_channels):
                # Calculate the row and column indices for placing the channel
                row_index = i // canvas_cols
                col_index = i % canvas_cols
                
                # Calculate the start and end indices for placing the channel on the canvas
                start_row = row_index * latent_dim[0]
                end_row = start_row + latent_dim[0]
                start_col = col_index * latent_dim[1]
                end_col = start_col + latent_dim[1]
        
                # Place the channel on the canvas
                if len(intermediate_shape) == 3:
                    canvas[start_row:end_row, start_col:end_col] = intermediate_prediction[:,:,i]
                elif len(intermediate_shape) == 2:
                    canvas[start_row:end_row, start_col:end_col] = intermediate_prediction[:,:]
                        
            latent_pl = Image.fromarray(np.uint8(canvas*255)).resize((200,200))
            latent_tk = ImageTk.PhotoImage(latent_pl)
            latent_lab.configure(image=latent_tk)
            latent_lab.photo = latent_tk
            output_pl = Image.fromarray(np.uint8(pred*255))
            output_tk = ImageTk.PhotoImage(output_pl)
            output_label.configure(image=output_tk)
            output_label.photo = output_tk
            progress_var.set(epoch+1)

            input_size.configure(text="{} KB".format((img_array.nbytes)/1024))
            latent_size.configure(text="{} KB".format((canvas.nbytes)/1024))
            output_size.configure(text="{} KB".format((pred.nbytes)/1024))

            status_lab.configure(text="Training  Model... \n{}/{} epochs done.".format(epoch, epochs))
            window.update_idletasks()

    #-------------------------------------------------------
    if model_var=='train':
        #------------------MODEL--------------------------------
        
        optimizer='Nadam'
        activation='relu'
        loss="MeanSquaredError"

        latent_channels = 8
        kernel = (3,3)
        model = Sequential()
        model.add(Conv2D(32, kernel, activation=activation, strides=2, padding='same', input_shape=(model_size, model_size, img_channels)))
        model.add(Conv2D(32, kernel, activation=activation, strides=2, padding='same'))

        model.add(Conv2D(latent_channels, kernel, activation=activation, strides=2, padding='same'))
        model.layers[-1]._name = "latent_space"

        model.add(Conv2DTranspose(32, kernel, strides=2, activation=activation, padding='same'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2DTranspose(img_channels, kernel, strides=2, activation=activation, padding='same'))

        model.compile(optimizer, loss = loss, metrics=['accuracy']) 

        layer_output=model.get_layer('latent_space').output  #get the Output of the Layer
        intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)

        #---------------------------------------------------------

        create_frames = PerformancePlotCallback(img_array, model_name=model)
        model.fit(img_array, img_array, epochs=epochs, shuffle=True, callbacks=[create_frames])
        # model.predict(img_array, callbacks=[create_frames]) 

    # model.summary()

    layer_output=model.get_layer('latent_space').output  #get the Output of the Layer
    intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
    
   
    pred = model.predict(img_array)
    pred[0] = pred[0] / pred[0].max()
    pred = np.reshape(pred, (len(pred), model_size, model_size, img_channels))
    pred = np.squeeze(pred)
    

    intermediate_prediction=intermediate_model.predict(img_array) #predicting in the Intermediate Node
    intermediate_prediction = np.squeeze(intermediate_prediction)
    print(np.shape(intermediate_prediction))  
    image_array = np.asarray(cv2.split(np.squeeze(intermediate_prediction)))

    # Get the shape of intermediate_prediction
    intermediate_shape = intermediate_prediction.shape

    # Determine the number of channels
    if len(intermediate_shape) == 3:
        latent_channels = intermediate_shape[-1]
    elif len(intermediate_shape) == 2:
        # Assuming the shape is (height, width), and there's only one channel
        latent_channels = 1

    # Initialize the canvas
    canvas_rows = int(np.ceil(latent_channels / 4))  # Calculate the number of rows needed
    canvas_cols = min(latent_channels, 4)           # Maximum 4 columns
    canvas = np.zeros((latent_dim[0] * canvas_rows, latent_dim[1] * canvas_cols))

    # Iterate over each channel
    for i in range(latent_channels):
        # Calculate the row and column indices for placing the channel
        row_index = i // canvas_cols
        col_index = i % canvas_cols
        
        # Calculate the start and end indices for placing the channel on the canvas
        start_row = row_index * latent_dim[0]
        end_row = start_row + latent_dim[0]
        start_col = col_index * latent_dim[1]
        end_col = start_col + latent_dim[1]
        
        # Place the channel on the canvas
        if len(intermediate_shape) == 3:
            canvas[start_row:end_row, start_col:end_col] = intermediate_prediction[:,:,i]
        elif len(intermediate_shape) == 2:
            canvas[start_row:end_row, start_col:end_col] = intermediate_prediction[:,:]


    latent_pl = Image.fromarray(np.uint8(canvas*255)).resize((256,128))
    latent_tk = ImageTk.PhotoImage(latent_pl)
    latent_lab.configure(image=latent_tk)
    latent_lab.photo = latent_tk

    psnr.configure(text="PSNR: {} dB".format(calculate_psnr(img_array, pred)))
    print(calculate_psnr(img_array, pred))
    comp_ratio.configure(text="Compression Ratio: {}".format(ratioFunction(img_array.nbytes/1000, intermediate_prediction.nbytes/1000)))

    input_size.configure(text="{} KB".format((img_array.nbytes)/1024))
    latent_size.configure(text="{} KB".format((intermediate_prediction.nbytes)/1024))
    output_size.configure(text="{} KB".format((pred.nbytes)/1024))

    matplotlib.image.imsave('output/out_e{}.png'.format(epochs), pred)
    matplotlib.image.imsave('output/latent_space_e{}.png'.format(epochs), canvas)

    matplotlib.image.imsave('output/input.png', np.squeeze(img_array))

    pred = cv2.resize(pred, (size, size))
    output_pl = Image.fromarray(np.uint8(pred*255))
    output_tk = ImageTk.PhotoImage(output_pl)
    output_label.configure(image=output_tk)
    output_label.photo = output_tk

    window.update_idletasks()