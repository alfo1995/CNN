#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 23:31:28 2018

@author: alfonsodamelio
"""

# # usa per Conv NN


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten,Input,Convolution2D,Concatenate,concatenate,AveragePooling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam
from keras.models import model_from_json
import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np




with open("../data/font.p", "rb") as fp:   # Unpickling
 font = pickle.load(fp)
 
with open("../data/char.p", "rb") as fp:   # Unpickling
 char = pickle.load(fp)
 
with open("../data/bold.p", "rb") as fp:   # Unpickling
 bold = pickle.load(fp)
 
with open("../data/italic.p", "rb") as fp:   # Unpickling
 italic = pickle.load(fp)








# Now create the output thanks to OneHotEncoder
enc = LabelEncoder()

# #### Font  encoder

encoder=enc.fit(font)
enc_transf=enc.transform(font)
y_font=np_utils.to_categorical(enc_transf)
y_font.shape


# #### char  encoder


encoder=enc.fit(char)
enc_transf_char=enc.transform(char)
y_char=np_utils.to_categorical(enc_transf_char)
y_char.shape


# #### Bold  encoder

encoder=enc.fit(bold)
enc_transf_bold=enc.transform(bold)
y_bold=np_utils.to_categorical(enc_transf_bold)
y_bold.shape


# #### Italics  encoder



encoder=enc.fit(italic)
enc_transf_italic=enc.transform(italic)
y_italic=np_utils.to_categorical(enc_transf_italic)
y_italic.shape



X_train=np.load('../data/augment.npy')



input_shape = X_train.shape[1:] 

#lookup the functional API of keras...  new_layer_name = new_layer_type(parameters) (previous_layer)
inp = Input(shape=input_shape)

###### NET1
conv_start=Convolution2D(filters = 32, kernel_size=5, padding = 'same') (inp)
conv=Convolution2D(filters = 32, kernel_size=3, padding = 'same') (conv_start)
pool_start = MaxPooling2D(pool_size=2) (conv)
drop_start = Dropout(0.25) (pool_start)

## FONT
Conv_font=Convolution2D(filters = 32, kernel_size=3, padding = 'same') (drop_start)
flat_font = Flatten()(Conv_font)
net_font=(Dense(32,activation='relu'))(flat_font)
drop_font = Dropout(0.25)(net_font)
net_font1=(Dense(32,activation='relu'))(drop_font)
drop_font1 = Dropout(0.25)(net_font1)
out1=(Dense(y_font.shape[1],activation='relu',name='font'))(drop_font1)


###### NET2
conv_2=Convolution2D(filters = 16, kernel_size=3, padding = 'same') (drop_start)
pool2 = MaxPooling2D(pool_size=2) (conv_2)
drop2= Dropout(0.25) (pool_start)

## CHAR
Conv_char=Convolution2D(filters = 32, kernel_size=3, padding = 'same') (drop2)
flat_char = Flatten()(Conv_char)
net_char=(Dense(32,activation='relu'))(flat_char)
drop_char = Dropout(0.25)(net_char)
net_char1=(Dense(16,activation='relu'))(drop_char)
drop_char1 = Dropout(0.25)(net_char1)
out2=(Dense(y_char.shape[1],activation='relu',name='char'))(drop_char1)

###### NET3
conv3=Convolution2D(filters = 32, kernel_size=5, padding = 'same') (drop2)
conv_3=Convolution2D(filters = 32, kernel_size=3, padding = 'same') (conv3)
pool3 = MaxPooling2D(pool_size=2) (conv_3)
drop3= Dropout(0.25) (pool3)

## BOLD
Conv_bold=Convolution2D(filters = 32, kernel_size=3, padding = 'same') (drop3)
flat_bold = Flatten()(Conv_bold)
net_bold=(Dense(16,activation='softmax'))(flat_bold)
drop_bold = Dropout(0.25)(net_bold)
net_bold1=(Dense(16,activation='softmax'))(drop_bold)
drop_bold1 = Dropout(0.25)(net_bold1)
out3=(Dense(y_bold.shape[1],activation='softmax',name='bold'))(drop_bold1)

## italic
Conv_italic=Convolution2D(filters = 32, kernel_size=3, padding = 'same') (drop3)
flat_italic = Flatten()(Conv_italic)
net_italic=(Dense(16,activation='softmax'))(flat_italic)
drop_italic = Dropout(0.25)(net_italic)
net_italic1=(Dense(16,activation='softmax'))(drop_italic)
drop_italic1 = Dropout(0.25)(net_italic1)
out4=(Dense(y_italic.shape[1],activation='softmax',name='italic'))(drop_italic1)





model = Model(inputs=inp, outputs=[out1,out2,out3,out4])


##adam=Adam(lr=0.01)
model.compile(loss = {'font':'categorical_crossentropy','char':'categorical_crossentropy',
                     'bold':'binary_crossentropy','italic':'binary_crossentropy'},
              optimizer='Adam', metrics = ['accuracy'])

model.fit(X_train,{'font':y_font,'char':y_char,'bold':y_bold,
                  'italic':y_italic},epochs = 4,batch_size=200)




#200 batchsize
#200 epochs




######Show intermediate result of convolution

def show_intermediate_result(conv,X_train,name):
    
    single_image=X_train[0] #show on image n°200 of the ndarray
    single_image = np.expand_dims(single_image, axis=0)
    #single_image.shape
    model_new = Model(inputs=inp, outputs=conv) # To define a model, just specify its input and output layers
    
    model_new.layers[0].set_weights(model.layers[0].get_weights())
    
    #create intermediate image
    convolved_single_image = model_new.predict(single_image)
    convolved_single_image = convolved_single_image[0]
    
    #plot the output of each intermediate filter
    conv_depth_2=20 #number of filters applied (we print just 20)
    for i in range(conv_depth_2):
            filter_image = convolved_single_image[:,:,i]
            plt.subplot(6,int(conv_depth_2/6)+1,i+1)
            plt.imshow(filter_image,cmap='gray'); plt.axis('off');
            plt.savefig('../data_out/img/img_intermediate_'+str(name)+'.png')



show_intermediate_result(conv_start,X_train,1)
show_intermediate_result(conv,X_train,2)
show_intermediate_result(Conv_font,X_train,3)
show_intermediate_result(Conv_char,X_train,4)
show_intermediate_result(Conv_bold,X_train,5)
show_intermediate_result(Conv_italic,X_train,6)





#save model
model_json = model.to_json()
with open("../data_out/model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../data_out/model/model.h5")





##model summary to do
from contextlib import redirect_stdout 

with open('../data_out/model/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()







