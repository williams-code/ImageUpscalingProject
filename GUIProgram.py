from tkinter import filedialog
from tkinter import *
from shutil import copyfile

import tensorflow as tensorflow
import numpy as numpy
import matplotlib.pyplot as matplotlib
import uuid
import os
import datetime
from tensorflow_addons import optimizers

from tensorflow.keras import models, layers, datasets
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from numpy import save, asarray

from PIL import Image

outputResolution = (1280, 1920)
modelSelected = 1

def loadModel(number):
    global savedModel
    if(number == 1):
        savedModel = tensorflow.keras.models.load_model(r'426to1280.h5', compile=False)
        savedModel.compile(optimizer=optimizers.Lookahead(optimizers.RectifiedAdam(amsgrad=True)))
    if(number == 2):
        savedModel = tensorflow.keras.models.load_model(r'1280to1600.h5')
    if(number == 3):
        savedModel = tensorflow.keras.models.load_model(r'426to1600.h5', compile=False)
        savedModel.compile(optimizer=optimizers.Lookahead(optimizers.RectifiedAdam(amsgrad=True)))


def testImage(input, model, label):
    label.set("Running...")
    inputBytes = tensorflow.io.read_file(input)
    normalizedTensor = tensorflow.cast(tensorflow.image.decode_jpeg(inputBytes), tensorflow.float32) / 255
    resizedInputImageToModel  = tensorflow.image.resize(normalizedTensor, size=outputResolution, method="gaussian")  
    inputImageToModel = numpy.expand_dims(resizedInputImageToModel , axis = 0)
    tempResized = numpy.squeeze(inputImageToModel)
    numpyOutput = tempResized
    minimum = numpy.min(numpyOutput)
    maximum = numpy.max(numpyOutput)
    rescaledOutput = (numpyOutput - minimum) / (maximum - minimum)
    matplotlib.imsave(str(os.getcwd()) + r'\AfterImageResize\guassianOnly.png', rescaledOutput)
    outputTensor = model.predict(inputImageToModel)
    outputTensor_ = numpy.squeeze(outputTensor)
    numpyOutput = outputTensor_
    minimum = numpy.min(numpyOutput)
    maximum = numpy.max(numpyOutput)
    testOutput = (numpyOutput - minimum) / (maximum - minimum)
    matplotlib.imsave(str(os.getcwd()) + r'\OutputImage\test.png', testOutput)

    # Show gaussian compared to neural network upscaled image for comparison.
    figure, imagesArray = matplotlib.subplots(1, 2)
    imagesArray[0].imshow(rescaledOutput)
    imagesArray[0].set_title('Gaussian Only')
    imagesArray[1].imshow(testOutput)
    imagesArray[1].set_title('Gaussian + Neural Network')
    matplotlib.show(block=False)

    savedImage = saveImage()
    if savedImage:
        if not savedImage.endswith(".png"):
            savedImage = savedImage + ".png"
        print(str(savedImage))
        matplotlib.imsave(savedImage, testOutput)

def openFileDiag(label):
    label.set("Select an image upscale and wait.")
    tk.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select File",filetypes = [("all files","*")])
    print (tk.filename)
    # Convert the image to a JPEG if the image selected is a PNG.
    if(tk.filename.endswith(".png")):
        selectedImage = Image.open(tk.filename)
        convertedImage = selectedImage.convert("RGB")
        convertedImage.save(str(os.getcwd()) + r'\UploadedImage\input.jpg', "JPEG")
    else:
        copyfile(tk.filename, str(os.getcwd()) + r'\UploadedImage\input.jpg')
    if(modelSelected == 1):
        loadModel(1)
        testImage(os.getcwd() + r'\UploadedImage\input.jpg', savedModel, label)
    elif(modelSelected == 2):
        loadModel(2)
        testImage(os.getcwd() + r'\UploadedImage\input.jpg', savedModel, label)
    elif(modelSelected == 3):
        loadModel(3)
        testImage(os.getcwd() + r'\UploadedImage\input.jpg', savedModel, label)
    label.set("Select your photo to upscale:")

def saveImage():
    fileTypes=[('Portable Network Graphics','*.png')]
    try:
        fileName = filedialog.asksaveasfilename(title="Save Your Upscaled Image", filetypes=fileTypes)
    finally: 
        print("User likely closed the save dialog without specifying a location.")
    return fileName

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)        
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.labelText = StringVar()
        self.label = Label( tk, textvariable=self.labelText, relief=RAISED, bd=0)
        self.labelText.set("Select your photo to upscale:")
        self.label.place(x=80, y=0)
        openFile = Button(self, text="Open File", command=self.clickOpenFileButton)
        openFile.place(x=135, y=30)
        
        def model1():
            global modelSelected
            modelSelected = 1
            global outputResolution 
            outputResolution = (1280, 1920)
            model = 1
        def model2():
            global modelSelected
            modelSelected = 2
            global outputResolution
            outputResolution = (1600, 2400)
        def model3():
            global modelSelected
            modelSelected = 3
            global outputResolution
            outputResolution = (1600, 2400)
        model = 1
        radioButton1 = Radiobutton(tk, text="640x427 to 1920x1280", variable=model, value=1, command=model1)
        radioButton1.pack( anchor = W )
        radioButton2 = Radiobutton(tk, text="1920x1280 to 2400x1600", variable=model, value=2, command=model2)
        radioButton2.pack( anchor = W )
        radioButton3 = Radiobutton(tk, text="640x427 to 2400x1600", variable=model, value=3, command=model3)
        radioButton3.pack( anchor = W )
        radioButton1.select()
        model = 1
        model1()

    def clickOpenFileButton(self):
        openFileDiag(self.labelText)
        

    def clickSaveFileButton(self):
        saveImage()
   
tk = Tk()
app = Window(tk)
tk.wm_title("Image Upscaling")
tk.geometry("320x200")
tk.mainloop()