#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 21:46:56 2018

@author: alfonsodamelio
"""
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
import pickle



lett=list(string.printable[:-6])



#font
ft = os.listdir('../data/fonts')


arr=[]
for i in lett:
    for j in ft:
        if j!='desktop.ini':
            font = ImageFont.truetype(os.path.join('../data/fonts',j), 30)

            # get the line size
            # create a blank canvas with extra space between lines
            canvas = Image.new('L', (64 , 64 ),color="white")

            # draw the text onto the text canvas, and use black as the text color
            draw = ImageDraw.Draw(canvas)
            draw.text((25,5), i, font=font, fill=0)
            
            if i=='/':
                i='slash'
            
            c=j[:-4].split('_')
            arr.append([i]+c) #append in a list [char,font,...]

#in this part of the code we build up the list that we use us y_font,y_char,y_bold and y_italic and then we will pass to one hot encoder
output=[]
for i in arr:
   
    if len(i)<3:
        i.append(0)
        i.append(0)
    elif len(i)==3:
        i.append(0)
  
    output.append(i)
for j in output:
    if len(j)==4 and j[2]=='I':
        j[2],j[3]=j[3],j[2]
    
        

#multiply 34 times, because it's the number of augmentation done
#so if we have ['a','Arial',B,0] we multiply this 34 times, because we augment this 34 times
output=[i for i in output for x in range(34)]



#here we divide ,from the output list, char, font ,bold and italics

font =[]
char =[]
bold =[]
italic=[]
for i in range(len(output)):
    char.append(output[i][0])
    font.append(output[i][1])
    bold.append(output[i][2])
    italic.append(output[i][3])
    

for i in range(len(char)):
    if char[i]=='slash':
        char[i]='/'
 
    

#save list of font,char,bold,italic
with open("../data/font.p", "wb") as fp:   
  pickle.dump(font, fp)
  
with open("../data/char.p", "wb") as fp:  
  pickle.dump(char, fp)

with open("../data/bold.p", "wb") as fp:   
  pickle.dump(bold, fp)

with open("../data/italic.p", "wb") as fp:   
  pickle.dump(italic, fp)




