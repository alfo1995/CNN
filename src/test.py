#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 16:40:46 2018

@author: alfonsodamelio
"""
from keras.models import model_from_json
import sys
import numpy as np
import string
from sklearn.metrics import accuracy_score
import csv  
import os 



# load json and create model
json_file = open('../data_out/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../data_out/model/model.h5")


# load test from input
X_test=np.load(sys.argv[1])

#make prediction on the test
prediction=loaded_model.predict(X_test,verbose=1)

#font list sorted
ft = os.listdir('../data/fonts')
fon_t=[i[:-4] for i in ft]
fon_t=set(fon_t)
sort=sorted([i for i in fon_t])
sorted_fonts=[]
for i in sort:
    sorted_fonts.append(i.split('_'))
font=[]
for i in sorted_fonts:
    font.append(i[0])
font=font[:-1]
sorted_font=[i for i in set(font)]
sorted_font=sorted(sorted_font)

#char list sorted
charact=sorted(string.printable[:-6])

#bold list sorted
bold=sorted([1.,0.])

#italic list sorted
italic=sorted([1.,0.])



font_column=[] 
for i in prediction[0]:#sono 89488 liste tante quante sono le foto, ognuna di 11----> numero di font
    lista=list(i)
    index=lista.index(max(lista))
    font_column.append(sorted_font[index])
 
char_column=[]
for i in prediction[1]:#sono 89488 liste tante quante sono le foto,ognuna di 94...----->numero di char
    lista=list(i)
    index=lista.index(max(lista))
    char_column.append(charact[index])

bold_column=[]    
for i in prediction[2]:#sono 89488 liste tante quante sono le foto,ognuna di 2...----->numero di bold
    lista=list(i)
    index=lista.index(max(lista))
    bold_column.append(bold[index])
    
italic_column=[]    
for i in prediction[3]:#sono 89488 liste tante quante sono le foto,ognuna di 2...----->numero di italic
    lista=list(i)
    index=lista.index(max(lista))
    italic_column.append(italic[index])
    


#save prediction in csv file
with open('../data_out/test/Partial_result.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile,delimiter=',')
    #wr.writerow(('char','font','bold','italics'))
    wr.writerows(zip(char_column,font_column,bold_column,italic_column))





    
    

