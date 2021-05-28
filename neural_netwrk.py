import sys
import os
import json
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Input


data=pd.read_csv('phishing.csv')
labels=data['class']

#model = Sequential()
model = Sequential(
    [
        Dense(32,activation='relu', input_shape=(30,)),
        
        
        Dense(32,activation='relu'),
        #Dropout(0.5),
        Dense(32,activation='relu'),
        
        Dense(32,activation='relu'),
        Dropout(0.2),
        
        
        Dense(16,activation='relu'),
        Dense(16,activation='relu'),
        
        Dropout(0.2),
        
        Dense(16,activation='relu'),
        Dense(16,activation='relu'),
        Dropout(0.2),
        
        Dense(8,activation='relu'),
        
        Dense(8,activation='relu'),
        
        Dense(8,activation='relu'),
        Dense(8,activation='relu'),
        
        
        Dense(4,activation='relu'),
        Dense(4,activation='relu'),
        
        
        Dense(2,activation='softmax')
        
        
       
    ]
)

'''
x=Input(shape=(30,))
l1=Dense(32,activation='relu',input_shape=x.shape)(x)
d1=Dropout(0.5,input_shape=l1.shape)(l1)

l2=Dense(32,activation='relu',input_shape=d1.shape)(d1)
d2=Dropout(0.5,input_shape=l1.shape)(l2)

l2=Dense(16,activation='relu',input_shape=d1.shape)(d1)
d2=Dropout(0.5,input_shape=l1.shape)(l2)

l3=Dense(8,activation='relu',input_shape=d2.shape)(d2)
d3=Dropout(0.5,input_shape=l1.shape)(l3)

l4=Dense(4,activation='relu',input_shape=d3.shape)(d3)
d4=Dropout(0.5,input_shape=l1.shape)(l4)

y=Dense(2,activation='softmax', input_shape=d4.shape)(d4)
'''
#model=Model(x,y)


