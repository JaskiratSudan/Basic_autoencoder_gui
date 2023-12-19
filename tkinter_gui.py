import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import glob
import cv2
from autoencoder import train_model


window = ttk.Window(themename='darkly')
window.title("Pixel Zipper")
window.geometry('900x600')

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
    status_lab.config(text="Enter number of epochs and Train.")

def training(path, epochs, latent_ch, lab):
    train_model(path=path, epochs=epochs, window=window, output_label=lab, status_lab=status_lab, latent_lab=latent_lab,progress_var=progress_var, progress_bar=pb, latent_ch=latent_ch, input_size=input_size, latent_size=latent_size, output_size=output_size)
    status_lab.config(text="Done Training")

import_button = ttk.Button(window, text='Import image', command=lambda:import_img(train_label))

train_pl = Image.open(person_img).resize((img_size,img_size))
train_tk = ImageTk.PhotoImage(train_pl)
train_label=ttk.Label(image=train_tk)

def click(event):
    event.config(state='normal')
    event.delete(0, 'end')

epochs_val = ttk.Entry(master=window)
epochs_val.insert(0, "Enter epochs here")
epochs_val.config(state='disabled')
epochs_val.bind('<Button-1>', click(epochs_val))

latent_val = ttk.Entry(master=window)
latent_val.insert(0, "Enter latent channels here")
latent_val.config(state='disabled')
latent_val.bind('<Button-1>', click(latent_val))

train_button = ttk.Button(window, text='Train', command=lambda:training(path=path, epochs=int(epochs_val.get()), latent_ch=int(latent_val.get()), lab=output_label))

output_pl = Image.open(person_img).resize((256,256))
output_tk = ImageTk.PhotoImage(output_pl)
output_label = ttk.Label(window, image=output_tk)

status_lab = ttk.Label(anchor="center", text="Please Import image.")

check_var = tk.IntVar()
# model_check = ttk.Checkbutton(window, text="increase depth of latent space.", variable=check_var)
progress_var = tk.DoubleVar()
progress_var.set(0)
pb = ttk.Progressbar(window, variable=progress_var, orient='horizontal', mode='determinate', length=200)

latent_pl = Image.open(person_img).resize((256,256))
latent_tk = ImageTk.PhotoImage(latent_pl)
latent_lab = ttk.Label(window, image=output_tk)

input_info = ttk.Label(window, text="Input Image")
latent_info = ttk.Label(window, text="Latent Space Channels")
output_info = ttk.Label(window, text="Reconstructed Image")

input_size = ttk.Label(window, text="Input image size.")
latent_size = ttk.Label(window, text="Latent image size.")
output_size = ttk.Label(window, text="Output image size.")



window.columnconfigure(0, weight=5)
window.columnconfigure(1, weight=1)
window.columnconfigure(2, weight=5)
window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=1)
window.rowconfigure(2, weight=1)

import_button.grid(row=0, column=0)
train_label.grid(row=1, column=0)
epochs_val.grid(row=0, column=1)
latent_val.grid(row=0,column=1,sticky='s')
# model_check.grid(row=0, column=2, sticky='s')
latent_lab.grid(row=1, column=1)
train_button.grid(row=0,column=2, sticky='w')
output_label.grid(row=1, column=2)
status_lab.grid(row=2, column=1, sticky='n')
# input_size.grid(row=2, column=0, sticky='n')
# latent_size.grid(row=2, column=1, sticky='n')
# output_size.grid(row=2, column=2, sticky='n')
pb.grid(row=2,column=1)
input_info.grid(row=1,column=0, sticky='n')
latent_info.grid(row=1,column=1, sticky='n')
output_info.grid(row=1,column=2, sticky='n')

window.mainloop()