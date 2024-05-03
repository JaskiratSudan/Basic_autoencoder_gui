import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import glob
import cv2
from autoencoder import train_model


window = ttk.Window(themename='darkly')
window.title("Scalable Convolutional Autoencoder (SCA)")
window.geometry('900x600')

font_style = ("Helvetica", 12, "bold")

image_path = "assets/waves.jpg"  # Change this to the path of your image
image = Image.open(image_path)

# Resize the image to fit the window size
window_width = window.winfo_screenwidth()
window_height = window.winfo_screenheight()
image = image.resize((900,600))

# Convert the image to a Tkinter-compatible format
tk_image = ImageTk.PhotoImage(image)

# Create a label with the image as the background
background_label = tk.Label(window, image=tk_image, font=font_style)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

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


model_var=256
def training(path, epochs, lab):
    if var.get() == True:
        model_var = 'train'
    elif model_512.get() == 1:
        model_var = 512
    elif model_512.get() == 0:
        model_var = 256

    train_model(path=path, epochs=epochs, window=window, output_label=lab, status_lab=status_lab, latent_lab=latent_lab,progress_var=progress_var, progress_bar=pb, input_size=input_size, latent_size=latent_size, output_size=output_size, psnr=psnr, comp_ratio=comp_ratio, model_var=model_var)
    status_lab.config(text="Done Training")

import_button = ttk.Button(window, text='Import image', command=lambda:import_img(train_label))

train_pl = Image.open(person_img).resize((img_size,img_size))
train_tk = ImageTk.PhotoImage(train_pl)
train_label=ttk.Label(image=train_tk, font=font_style)

def click(event):
    event.config(state='normal')
    event.delete(0, 'end')

epochs_val = ttk.Entry(master=window)
# epochs_val.insert(0, "Enter epochs here")
# epochs_val.config(state='disabled')
epochs_val.bind('<Button-1>', click(epochs_val))
epochs_val.config(state=tk.DISABLED)

# latent_val = ttk.Entry("", master=window, state=tk.DISABLED)
# latent_val.insert(0, "Enter latent channels here")
# latent_val.config(state='disabled')
# latent_val.bind('<Button-1>', click(latent_val))
    
model_512 = tk.IntVar()

model_512_box = ttk.Checkbutton(window, text="Scaled Model (512 pixel)", variable=model_512)

def toggle_widgets():
    """Toggle the state of labels and the button based on the checkbox state."""
    if var.get():
        model_512.set(False)
        model_512_box.config(state=tk.DISABLED)
        compress_button.config(state=tk.DISABLED)
        epochs_val.config(state=tk.NORMAL)
        # latent_val.config(state=tk.NORMAL)
        train_button.config(state=tk.NORMAL)
    else:
        model_512_box.config(state=tk.NORMAL)
        compress_button.config(state=tk.NORMAL)
        epochs_val.config(state=tk.DISABLED)
        # latent_val.config(state=tk.DISABLED)
        train_button.config(state=tk.DISABLED)

# Create a Checkbutton widget
var = tk.BooleanVar()
train_checkbox = tk.Checkbutton(window, text="Training Mode", variable=var, command=toggle_widgets)

train_button = ttk.Button(window, text='Train', command=lambda:training(path=path, epochs=int(epochs_val.get()), lab=output_label), state=tk.DISABLED)


# train_button = ttk.Button(window, text='Train', command=lambda:training(path=path, epochs=int(epochs_val.get()), latent_ch=int(latent_val.get()), lab=output_label))
compress_button = ttk.Button(window, text='Compress', command=lambda:training(path=path, epochs=0, lab=output_label))


output_pl = Image.open(person_img).resize((img_size,img_size))
output_tk = ImageTk.PhotoImage(output_pl)
output_label = ttk.Label(window, image=output_tk)

status_lab = ttk.Label(anchor="center", text="Please Import image.", font=font_style)

progress_var = tk.DoubleVar()
progress_var.set(0)
pb = ttk.Progressbar(window, variable=progress_var, orient='horizontal', mode='determinate', length=200)

latent_pl = Image.open(person_img).resize((int(img_size/2),int(img_size/2)))
latent_tk = ImageTk.PhotoImage(latent_pl)
latent_lab = ttk.Label(window, image=latent_tk)

input_info = ttk.Label(window, text="Input Image", font=font_style)
latent_info = ttk.Label(window, text="Latent Space Channels", font=font_style)
output_info = ttk.Label(window, text="Reconstructed Image", font=font_style)

input_size = ttk.Label(window, text="Input image size.", font=font_style)
latent_size = ttk.Label(window, text="Latent image size.", font=font_style)
output_size = ttk.Label(window, text="Output image size.", font=font_style)

psnr = ttk.Label(window, text="PSNR: ", font=font_style)
comp_ratio = ttk.Label(window, text="Compression Ratio: ", font=font_style)


window.columnconfigure(0, weight=5)
window.columnconfigure(1, weight=1)
window.columnconfigure(2, weight=5)
window.rowconfigure(0, weight=1)
window.rowconfigure(1, weight=1)
window.rowconfigure(2, weight=1)
window.rowconfigure(3, weight=1)

import_button.grid(row=0, column=0)
train_label.grid(row=2, column=0)
epochs_val.grid(row=0, column=1)
# latent_val.grid(row=0,column=1,sticky='s')
model_512_box.grid(row=0, column=2)
train_checkbox.grid(row=0, column=2, sticky='s')
latent_lab.grid(row=2, column=1)
compress_button.grid(row=1,column=1, sticky='e')
train_button.grid(row=1,column=1, sticky='w')
output_label.grid(row=2, column=2)
# status_lab.grid(row=3, column=1, sticky='n')
input_size.grid(row=2, column=0, sticky='s')
latent_size.grid(row=2, column=1, sticky='s')
output_size.grid(row=2, column=2, sticky='s')
pb.grid(row=3,column=1, sticky='n')
input_info.grid(row=2,column=0, sticky='n')
latent_info.grid(row=2,column=1, sticky='n')
output_info.grid(row=2,column=2, sticky='n')
psnr.grid(row=3,column=1)
comp_ratio.grid(row=3,column=2)

window.mainloop()