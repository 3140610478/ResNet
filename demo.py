import os
import sys
import torch
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
pass
base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Data.FashionMNIST import demo_set, demo_transform, classes
    from config import device
    
resnet32 = torch.load(os.path.abspath(os.path.join(
    base_folder, "./Networks/ResNet32/ResNet32.resnet32")))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The Demonstration GUI is programmed assisted by ChatGPT #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

current_image_index = 0

# This function simulates image classification
def classify_image(img):
    x = demo_transform(img).to(device).unsqueeze(dim=0)
    x = torch.cat((x, x, x), dim=1)
    y = classes[resnet32(x).flatten().argmax(dim=0)]
    return y

# Function to update the image and classification text
def update_image():
    global current_image_index
    if current_image_index >= len(demo_set):
        current_image_index = 0  # Reset to loop continuously

    img, cls = demo_set[current_image_index]
    prediction = classify_image(img)
    actual = classes[cls]
    img = img.resize((250, 250), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)

    image_label.configure(image=photo)
    image_label.image = photo  # Keep a reference so it's displayed correctly

    prediction_label.configure(text=f"prediction:\t{prediction}\t\t")
    actual_label.configure(text=f"actual:\t\t{actual}\t\t")

    current_image_index += 1
    # Schedule the next update; 2000 ms is the display time per image
    window.after(2000, update_image)

# Set up the GUI
window = tk.Tk()
window.title("ResNet Demo")

# Label to display the image
image_label = Label(window)
image_label.pack()

# Label to display the classification
prediction_label = Label(window, text="", anchor="w", width=32)
prediction_label.pack()
actual_label = Label(window, text="", anchor="w", width=32)
prediction_label.pack()
actual_label.pack()


if __name__ == "__main__":
    print("Demostration Start")
    # Start the image loop
    update_image()

    window.mainloop()
