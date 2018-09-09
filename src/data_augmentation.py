#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 21:32:03 2018

@author: alfonsodamelio
"""
import PIL
from PIL import Image,ImageDraw,ImageFont
import os
import random
from scipy import ndarray
from scipy import ndimage
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
import numpy as np
from scipy.ndimage import zoom
from skimage import transform


###augmentation function
def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-60, 60)
    return sk.transform.rotate(image_array, random_degree, cval=1)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

    
def random_rotation2(image_array: ndarray):
    # pick a random degree of rotation between 65% on the left and 25% on the right
    random_degree = random.uniform(-65, 25)
    return sk.transform.rotate(image_array, random_degree, cval=1)
    

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def random_rotation3(image_array: ndarray):
    # pick a random degree of rotation between 65% on the left and 25% on the right
    random_degree = random.uniform(-60, 30)
    return sk.transform.rotate(image_array, random_degree, cval=1)

def sp_noise(image,prob):
    
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

#translate random an image
def transl_random(arry_image):
    transl=transform.rescale(arry_image,scale=np.random.random()/2+0.5,mode="reflect")
    scl_x, scl_y =transl.shape
    pos=(np.random.randint(0,64-scl_y+1),np.random.randint(0,64-scl_y+1))
    transl_img=Image.new("L",(64,64),color='white')
    transl_img.paste(Image.fromarray(transl*255),pos)
    transl=np.array(transl_img)/255
    return transl


generator_data=np.load('../data/generator.npy')

#array n_samples equals to length of generator data times the number of augmentation done.
npy=np.ndarray((len(generator_data)*34,64,64,1))


cont=0
for i in range(0,len(generator_data)):
    
#augment original
    npy[cont,:,:,0]=generator_data[i,:,:,0]
    cont+=1
    
    rotation=random_rotation(generator_data[i,:,:,0])
    npy[cont,:,:,0]=rotation
    cont+=1
    
    noise=random_noise(generator_data[i,:,:,0])
    npy[cont,:,:,0]=noise
    cont+=1

    
    rotation2=random_rotation2(generator_data[i,:,:,0])
    npy[cont,:,:,0]=rotation2
    cont+=1
    
    
    zoomed=clipped_zoom(generator_data[i,:,:,0],1.5)
    npy[cont,:,:,0]=zoomed
    cont+=1
    
    rotation3=random_rotation3(generator_data[i,:,:,0])
    npy[cont,:,:,0]=rotation3
    cont+=1
    
    salt_pepp=sp_noise(generator_data[i,:,:,0],0.02)
    npy[cont,:,:,0]=salt_pepp
    cont+=1
    
    transl=transl_random(generator_data[i,:,:,0])
    npy[cont,:,:,0]=transl
    cont+=1
    
    transl2=transl_random(generator_data[i,:,:,0])
    npy[cont,:,:,0]=transl2
    cont+=1
    
    transl3=transl_random(generator_data[i,:,:,0])
    npy[cont,:,:,0]=transl3
    cont+=1
    
    transl4=transl_random(generator_data[i,:,:,0])
    npy[cont,:,:,0]=transl4
    cont+=1
    
    transl5=transl_random(generator_data[i,:,:,0])
    npy[cont,:,:,0]=transl5
    cont+=1
    
    transl6=transl_random(generator_data[i,:,:,0])
    npy[cont,:,:,0]=transl6
    cont+=1
    
    transl7=transl_random(generator_data[i,:,:,0])
    npy[cont,:,:,0]=transl7
    cont+=1
    
   
#augment rotation1
    rotation_rotation=random_rotation(rotation)
    npy[cont,:,:,0]=rotation_rotation
    cont+=1
    
    rotation_rotation2=random_rotation2(rotation)
    npy[cont,:,:,0]=rotation_rotation2
    cont+=1
    
    
    rotation_zoom=clipped_zoom(rotation,1.5)
    npy[cont,:,:,0]=rotation_zoom
    cont+=1
    
    rotation_rotation3=random_rotation3(rotation)
    npy[cont,:,:,0]=rotation_rotation3
    cont+=1
    
    rotat_transl=transl_random(rotation)
    npy[cont,:,:,0]=rotat_transl
    cont+=1
   

#augment rotation2   
    rotation_rotation2=random_rotation(rotation2)
    npy[cont,:,:,0]=rotation_rotation2
    cont+=1
    
      
    rotation_rotation2_2=random_rotation2(rotation2)
    npy[cont,:,:,0]=rotation_rotation2_2
    cont+=1
    
    
    rotation2_zoom=clipped_zoom(rotation2,1.5)
    npy[cont,:,:,0]=rotation2_zoom
    cont+=1
    
    rotation2_rotation3=random_rotation3(rotation2)
    npy[cont,:,:,0]=rotation2_rotation3
    cont+=1
    
    rotat2_transl=transl_random(rotation2)
    npy[cont,:,:,0]=rotat2_transl
    cont+=1
 
#augment zoom
    zoom_rotation2=random_rotation(zoomed)
    npy[cont,:,:,0]=zoom_rotation2
    cont+=1
    
    zoom_rotation2_2=random_rotation2(zoomed)
    npy[cont,:,:,0]=zoom_rotation2_2
    cont+=1
    
    
    zoom_zoom=clipped_zoom(zoomed,1.5)
    npy[cont,:,:,0]=zoom_zoom
    cont+=1
    
    zoom_rotation3=random_rotation3(zoomed)
    npy[cont,:,:,0]=zoom_rotation3
    cont+=1
    
    zoom_transl=transl_random(zoomed)
    npy[cont,:,:,0]=zoom_transl
    cont+=1

#augment rotation3 
    rotation_rotation3=random_rotation(rotation3)
    npy[cont,:,:,0]=rotation_rotation3
    cont+=1 
    
    rotation_rotation3_2=random_rotation2(rotation3)
    npy[cont,:,:,0]=rotation_rotation3_2
    cont+=1
    
    
    rotation3_zoom=clipped_zoom(rotation3,1.5)
    npy[cont,:,:,0]=rotation3_zoom
    cont+=1
    
    rotation3_rotation3=random_rotation3(rotation3)
    npy[cont,:,:,0]=rotation3_rotation3
    cont+=1
    
    rota3_transl=transl_random(rotation3)
    npy[cont,:,:,0]=rota3_transl
    cont+=1



np.save('../data/augment.npy', npy)

