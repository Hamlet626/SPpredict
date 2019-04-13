from __future__ import print_function, absolute_import, division

import tensorflow
from tensorflow import keras
#import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, GRU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.optimizers import rmsprop
import matplotlib.pyplot as plt
import os
import time
import csv
import numpy as np

class TimeHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def normalise_windows(window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data=np.array(window_data)
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            x=[float(p) for p in window[:, 0]]
            '''
            s=0
            for xx in range(int(len(x)/3),int(len(x)*2/3)):
                s=s+x[xx]
            s=3*s/len(x)
            for yy in range(len(x)):
                x[yy]=x[yy]/s
            '''
            normalised_window.append(x)
            for col_i in range(1,window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
    
def id(d,l,b,s):
    i=0
    x=[]
    y=[]
    while i<(l-s)/b:
        x1=[]
        y1=[]
        for j in range(i*b,i*b+b):
            x1.append(d[j:j+s-1])
            y1.append(d[j+s])
        yield x1,y1
        i=i+1
        


filename=os.path.join(os.getcwd(),'data','dataset.csv')
dataframe1=csv.DictReader(open(filename,mode='rb'))
dataframe=[]
x=0
for r in dataframe1:
    if(x>0):
        dataframe.append([r['ASPFWR5'],
                          r['OPEN'],r['HIGH'],r['LOW'],r['CLOSE']
                          #r['ASPFWR5'],r['US10YR'],r['EPS'],r['PER'],r['OPEN'],
                          #r['HIGH'],r['LOW'],r['CLOSE'],r['BDIY'],r['VIX'],r['PCR'],r['MVOLE'],
                          #r['DXY'],r['ASP'],r['ADVDECL'],r['FEDFUNDS']
                          #,r['NYSEADV'],r['IC'],r['BAA'],r['NOS'],r['BER'],
                          #r['DVY'],r['PTB'],r['AAA'],r['SI'],r['RV'],r['VVIX'],r['NAPMNEWO']
                          #,r['NAPMPRIC'],r['NAPMPMI'],r['US3M'],r['DEL'],r['BBY'],
                          #r['HTIME'],r['LTIME'],r['TYVIX'],r['PUC'],r['CRP'],r['TERM']
                          #,r['UR'],r['INDPRO'],r['HS'],r['VRP'],r['CAPE'],r['CATY'],
                          #r['INF'],r['sentiment1'],r['sentiment2'],r['Hulbert.sentiment']
                        ])
    x+=1
d=list(dataframe)
nm=len(d)
d=np.float_(d)

'''
li=[]
for o in range(len(d[0])):
    li.append([d[0][o],d[0][o]])
for u in d:
    for uu in range(len(u)):
        if u[uu]>li[uu][1]:
            li[uu][1]=u[uu]
        if u[uu]<li[uu][0]:
            li[uu][0]=u[uu]

for v in range(len(d)):
    for vv in range(len(d[0])):
        d[v][vv]=(d[v][vv]-li[vv][0])/(li[vv][1]-li[vv][0])
print('xxxx',li[0],li[1])
'''
#x,y=id(data,4000,32,50)
l=x*0.8
b=32
s=50
i=0
x=[]
y=[]
x2=[]
y2=[]
while i<(l-s)/b:
    #x1=[]
    #y1=[]
    for j in range(i*b,i*b+b):
        tt=normalise_windows(d[j:j+s+1],single_window=True)[0]
        #print(tt.shape)
        #x1.append(d[j:j+s])
        #y1.append([d[j+s][0]])
               
        x2.append(tt[:-1])
        y2.append([tt[-1][0]])
        #x2.append(d[j:j+s])
        #y2.append([d[j+s][0]])
            
        #yield x1,y1

    i=i+1

x=np.array(x)
y=np.array(y)
x2=np.array(x2)
y2=np.array(y2)

model = Sequential()
print('xxxxx',x.shape[1:])

model.add(LSTM(100,input_shape=x2.shape[1:],return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100,return_sequences=True))
model.add(LSTM(100,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))


model.summary()
model_name = 'model.h5'
model_path = os.path.join(os.getcwd(), model_name)

#opt = tensorflow.keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#we can self define an optimizer like this

# optimizer=opt if using another optimizer
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=2),
	    ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
	    ]
y=model.fit(x2,y2,batch_size=32,epochs=200,callbacks=callbacks)
model.save(model_path)
print(type(x2[0][0][0]),'xxxxx',x2.shape[1:])
