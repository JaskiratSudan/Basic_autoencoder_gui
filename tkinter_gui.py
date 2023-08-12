import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import glob
import cv2
# from threading import Thread
from autoencoder import train_model


window = ttk.Window(themename='darkly')
window.title("Basic Autoencoder GUI")
window.geometry('1100x600')

path = None
person_img = 'assets/person.png'
img_size = 256

def import_img(labelname):
    global path
    path = filedialog.askopenfile().name
    image_org = Image.open(path).resize((img_size,img_size))
    image = ImageTk.PhotoImage(image_org)
    labelname.configure(image=image)
    labelname.photo = image
    status_lab.config(text="Image Imported.\nEnter number of epochs and Train.")

def training(path, epochs, lab):
    status_lab.config(text="Training Model....")
    train_model(path=path, epochs=epochs)
    list_of_files = glob.glob('output/*')
    latest_image = max(list_of_files, key=os.path.getctime)
    output_pl = Image.open(latest_image).resize((img_size,img_size))
    output_tk = ImageTk.PhotoImage(output_pl)
    lab.configure(image=output_tk)
    lab.photo = output_tk
    status_lab.config(text="Done Training")

import_button = ttk.Button(window, text='Import image', command=lambda:import_img(train_label))

train_pl = Image.open(person_img).resize((img_size,img_size))
train_tk = ImageTk.PhotoImage(train_pl)
train_label=ttk.Label(image=train_tk)

def click(event):
    epochs_val.config(state='normal')
    epochs_val.delete(0, 'end')

epochs_val = ttk.Entry(master=window)
epochs_val.insert(0, "Enter epochs here")
epochs_val.config(state='disabled')
epochs_val.bind('<Button-1>', click)
train_button = ttk.Button(window, text='Train', command=lambda:training(path=path, epochs=int(epochs_val.get()), lab=output_label))

output_pl = Image.open(person_img).resize((256,256))
output_tk = ImageTk.PhotoImage(output_pl)
output_label = ttk.Label(window, image=output_tk)


# def Refresher():
#     list_of_files = glob.glob('frames/*')
#     latest_image = max(list_of_files, key=os.path.getctime)
#     output_pl = Image.open(latest_image).resize((img_size,img_size))
#     output_tk = ImageTk.PhotoImage(output_pl)
#     # print(output_pl)
#     output_label.configure(image=output_tk)
#     output_label.photo = output_tk
#     window.after(100, Refresher)

# progBar = ttk.Progressbar(window,orient='horizontal', length=400,mode="determinate")

status_lab = ttk.Label(text="Please Import image.")

window.columnconfigure(0, weight=5)
window.columnconfigure(1, weight=1)
window.columnconfigure(2, weight=5)
window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=1)
window.rowconfigure(2, weight=1)

import_button.grid(row=0, column=0)
train_label.grid(row=1, column=0)
epochs_val.grid(row=0, column=1)
train_button.grid(row=1,column=1)
output_label.grid(row=1, column=2)
status_lab.grid(row=2, column=1)

# thread = Thread(target=Refresher)
# thread.start()
window.mainloop()