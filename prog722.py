from __future__ import print_function, absolute_import, division
import logging
import tensorflow
from tensorflow import keras
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
#from tensorflow.keras.optimizers import rmsprop
import matplotlib.pyplot as plt
import os
#import time
import csv
import numpy as np

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
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

logging.getLogger('tensorflow').disabled=True
num_classes=10
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
filename1=os.path.join(os.getcwd(),'date.txt')
file=open(filename1,'r')
dat=file.readline()
dl=dat.split('-')
dl=[int(ab) for ab in dl]

filename=os.path.join(os.getcwd(),'dataset.csv')
dataframe1=csv.DictReader(open(filename,mode='r'))
dataframe=[]
an=True
ind=[]
si=[]
xx=0
x=0
for r in dataframe1:
    #print(r)
    if an:
        if r['ASPFWR5']!='NA':
            an=False
    else:
        ss=r['']
        s1=ss.split('-')
        ind.append([int(e) for e in s1])
        if r['ASPFWR5']=='NA':
            xx=x
        if xx==0:
            dataframe.append([r['ASPFWR5'],
                              r['OPEN'],r['HIGH'],r['LOW'],r['CLOSE']])
        else:
            if ind[-1][0]==dl[0] and ind[-1][1]==dl[1] and ind[-1][2]==dl[2]:
                break
            else:
                si.append([r['OPEN'],r['HIGH'],r['LOW'],r['CLOSE']])
        x+=1
#print(len(ind))
d=list(dataframe)
d=np.float_(d)
si=np.float_(si)
#x,y=id(data,4000,32,50)
#print(d)
#print(si)

model_name = 'model.h5'
model_path = os.path.join(os.getcwd(), model_name)
model=load_model(model_path)

for el in si:
    wind=normalise_windows(d[-50:],single_window=True)[0]
    ne=model.predict(np.array([wind]),verbose=0)
    #print(ne)
    #print(type(ne),type(el),ne.shape,el.shape)
    nel=np.array([ne[0]]+list(el))
    #print(nel.shape,d.shape)
    d=np.append(d,[nel],axis=0)
wind1=normalise_windows(d[-50:],single_window=True)[0]
output=model.predict(np.array([wind1]),verbose=0)
print(output[0][0])
'''
l=x
b=32
s=50
i=int(x*0.5/32)
x=[]
y=[]
x2=[]
y2=[]
li=[0,0]
while i<(l-s)/b-1:
    for j in range(i*b,i*b+b):
        tt=normalise_windows(d[j:j+s+1],single_window=True)[0]
        #print(tt.shape)
        x2.append(tt[:-1])
        y2.append([tt[-1][0]])
    i=i+1
x=np.array(x)
y=np.array(y)
x2=np.array(x2)
y2=np.array(y2)

'''

'''
model_name = 'model.h5'
model_path = os.path.join(os.getcwd(), model_name)
model=load_model(model_path)
print(x2.shape,x2.shape[1:])
r=0
c=model.predict(x2)
for j in range(len(x2)-32):
    r=r+abs((y2[j]-c[j])/y2[j])
    #c=model.predict(x2)
    #print(c.shape)
    #print(y2.shape)
    #for o in range(32):
    #+y2[j+o][1]+c[o][1]
    #r=r+abs(y2[j][0]-c[j][0])+abs(y2[j][1]-c[j][1])
    #j=j+32


scores = model.evaluate(x2, y2, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

print(r,r[0]/len(c))
plot_results(c,y2)
'''
