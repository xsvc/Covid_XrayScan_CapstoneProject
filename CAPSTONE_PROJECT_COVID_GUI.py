import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

root = Tk()
root.title("Image Classifier Application")

frame = tk.Frame(root, bg='#45aaf2')
lbl1 = tk.Label(text ="Image Classifier Result" )

lbl_show_pic = tk.Label(frame, bg='#45aaf2')
#entry_pic_path = tk.Entry(frame, font=('verdana',16))
btn_browse = tk.Button(frame, text='Select Image',bg='grey', fg='#ffffff',
                       font=('verdana',16))

def selectPic():
    global img
    filename = filedialog.askopenfilename(initialdir="/images", title="Select Image",
                           filetypes=(("jpeg images","*.jpeg"),("png images","*.png"),))
    img = Image.open(filename)
    img = img.resize((200,200))
    img = ImageTk.PhotoImage(img)
    lbl_show_pic['image'] = img

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(filename).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    #print("Class:", class_name[2:], end="")
    lbl1.config(text = class_name[2:] )
    print("Confidence Score:", confidence_score)
     
btn_browse['command'] = selectPic
lbl1.pack()
frame.pack()

#entry_pic_path.grid(row=0, column=1, padx=(0,20))
lbl_show_pic.grid(row=1, column=0, columnspan="2")
btn_browse.grid(row=2, column=0, columnspan="2", padx=10, pady=10)

root.mainloop()

# 1.[1BestCsharp blog]. "Python - How To Browse and Display Image in a label Using Filedialog In Tkinter 
# [ With Source Code ]," YouTube, [27/10/2023], [https://www.youtube.com/watch?v=ZbSncQYZ56U].