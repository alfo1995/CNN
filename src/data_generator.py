
# coding: utf-8



import PIL
from PIL import Image,ImageDraw,ImageFont
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.utils import np_utils
import os
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import string

lett=list(string.printable[:-6])



#font
ft = os.listdir('../data/fonts')


npy=np.ndarray((len(lett)*(len(ft)-1),64,64,1))

cont=0

for i in lett:
    for j in ft:
        if j!='desktop.ini':
            font = ImageFont.truetype(os.path.join('../data/fonts',j), 55)
        
            # get the line size
            # create a blank canvas with extra space between lines
            canvas = Image.new('L', (64 , 64 ),color="white")

            # draw the text onto the text canvas, and use black as the text color
            draw = ImageDraw.Draw(canvas)
            draw.text((15,5), i, font=font, fill=0)
                      
            data = np.array(canvas)/255
            npy[cont,:,:,0]=data
            cont+=1



np.save('../data/generator.npy', npy)