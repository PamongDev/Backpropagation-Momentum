import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import random as rdm
import json 
import time
import math
# %matplotlib inline
plt.rcParams['figure.figsize'] = [25, 7]

def normalization(array): #dataset pandas dataframe
    array_norm = (array-np.min(array)) / (np.max(array)-np.min(array))
    return array_norm

def toPandas(array,var):
    colName=[]
    size = array.shape[1]
    numArray=array
    reshaped_data = numArray.reshape(-1, size)
    for i in range(size,0,-1):
        if i!=1:
            colName.append(var+"-"+str(i-1))
        else:
            colName.append(var)
    tabel=pd.DataFrame(reshaped_data, columns = colName)
    return tabel

def convToDataProccess(dataset, dataInput):
    x=dataInput
    num = dataset.to_numpy()
    num=num.reshape(num.size)
    newData=[]
    for i in range(num.size-x):
        row=[]
        for j in range(i,x+i+1):
            row.append(num[j])
        newData.append(row)
    hasil = np.array(newData)
#     hasil = hasil.reshape(len(hasil), dataInput, 1)
    return hasil[:, :dataInput].reshape(len(hasil), dataInput, 1), hasil[:, dataInput:].reshape(len(hasil),1)

def dataSplitting(data, num):#persen / desimal 0.7 0.3
    train=round(len(data[0])*num)
    # test=round(len(data[0])*(1-train))
    xtrain, ytrain, xtest, ytest = data[0][:train], data[1][:train], data[0][train:], data[1][train:]
#     print(len(xtrain), len(xtest), len(ytrain), len(ytest))
    return xtrain, ytrain, xtest, ytest

def denormalisasi(arraynorm, minActual, maxActual):
    denormalized_array = (arraynorm * (maxActual - minActual)) + minActual
    return denormalized_array

def grafPrediksi(denorm, datatest):
#     mengatur ukuran grafik
    fig, ax = plt.subplots()

    ax.plot(denorm, label='hasil prediksi')
    ax.plot(datatest, label='Data sebenarnya')
    ax.set_xlabel('data ke-')
    ax.set_ylabel('Prediksi')
    ax.set_title('Hasil prediksi Backpropagation')
    ax.legend()
    return fig
    

def process(epoch, i):
#     time.sleep(0.5)
    progress = (i + 1) / epoch * 100
    # Membuat string loading
    loading_string = f"Proses Training: {int(progress)}% [{'=' * round(progress*60/100)}>] {i+1}/{epoch} "
    # Mencetak string loading
    print(loading_string, end='\r')

class backpropagation:
    def __init__(self, arch,lr, momentum, epochs, mse):# arch = [4,7,1]
        self.lr = lr
        self.momentum=momentum
        self.mse = mse
        self.timer = None
        self.waktu = 0
        self.epochs = epochs
        if arch!=[]:
            self.x, self.h, self.y = arch[0], arch[1], arch[2]
            self.weights(self.x, self.h, self.y)        
        
    def start_time(self):
        self.timer = time.time()
        
    def stop_timer(self):
        if self.timer is not None:
            elapsed_time = time.time() - self.timer
            self.timer = None
            return elapsed_time
        else:
            return None
    def weights(self, x, h, y):
        self.W1 = np.random.randn(h, x) #7,4
        self.b1 = np.zeros((h, 1)) #(7,1)
        self.W2 = np.random.randn(y, h) #1,7
        self.b2 = np.zeros((y, 1))
    
    def Funcz_in(self, x, W, b):
        z_in =np.dot(W, x) + b
        return z_in
    
    def sigmoid(self, z_in):
        z=1/(1 + np.exp(-z_in))
        return z
    
    def sigmoid_derivative(self, y_in):
        sigmoidx = self.sigmoid(y_in)
        return sigmoidx * (1 - sigmoidx)
    
    def forward(self, x):
#       lapisan pertama
        self.z_in=self.Funcz_in(x, self.W1, self.b1)
        self.z=self.sigmoid(self.z_in)
#       lapisan output
        self.y_in=self.Funcz_in(self.z, self.W2, self.b2)
        self.y=self.sigmoid(self.y_in)
    
    def backpro(self, x, Y, lambda_reg=0.01):
        #lapisan akhir
        self.err=self.y-Y
        self.delta2 = self.err*self.sigmoid_derivative(self.y_in)
        self.dW2 = np.dot(self.delta2, self.z.T)
        self.db2 = np.sum(self.delta2, axis=1, keepdims=True)
        #lapisan hiden layer
        self.delta1 = np.dot(self.W2.T, self.delta2) * self.sigmoid_derivative(self.z_in)
        self.dW1 = np.dot(self.delta1, x.T) 
        self.db1 = np.sum(self.delta1, axis=1, keepdims=True)
        #koreksi bobot dan bias lapisan terakhir
        if hasattr(self, 'dW1_prev'):
            self.dW1 += self.dW1_prev *self.momentum
            self.dW2 += self.dW2_prev *self.momentum
        self.W1correction=self.lr*self.dW1 
        self.b1correction=self.lr*self.db1 
        #lapisan hiden layer
        self.W2correction=self.lr*self.dW2 
        self.b2correction=self.lr*self.db2 
        #simpan delta sebelumnya
        self.dW1_prev = self.dW1.copy()
        self.dW2_prev = self.dW2.copy()
        #updatebobot
        self.W2 -= self.W2correction 
        self.b2 -= self.b2correction 
        self.W1 -= self.W1correction 
        self.b1 -= self.b1correction 
    
    def Funcmse(self, actual, prediction):
        error =actual-prediction
        squared = np.square(error)
        return squared
    
    def fit(self, X, Y):
        self.Xtrain, self.Ytrain = X, Y
        self.W1correction, self.b1correction, self.W2correction, self.b2correction = np.array([]), np.array([]), np.array([]), np.array([])
        data, epoch, msenow, mseepoch=0, 0, 0, 0
        self.arrayError=np.array([]) #menghitung error per epoch
        num_batch = len(self.Xtrain) // 32
        self.start_time()
        while epoch<self.epochs: #epoch 
            self.error=np.array([]) #menyimpan error setiap data   
            for batch in range (num_batch):
                start, end = batch * 32, (batch +1) * 32
                for data in range(start, end) : #iteration
                    self.forward(self.Xtrain[data])
                    self.backpro(self.Xtrain[data], self.Ytrain[data])
                    msenow = self.Funcmse(self.Ytrain[data], self.y)
                    self.error=np.append(self.error, msenow)
            mseepoch=np.mean(self.error)
            process(self.epochs, epoch)
            self.arrayError=np.append(self.arrayError, mseepoch)
            epoch+=1
            if mseepoch<=self.mse:
                break
        self.waktu=self.stop_timer()
        
    def test(self, X, Y):
        self.Xtest, self.Ytest = X, Y
        self.msenow = 0
        self.arraymse = np.array([])
        self.arrayofY=np.array([])
        for data in range(len(self.Xtest)):
            self.forward(self.Xtest[data])
            msenow = self.Funcmse(self.Ytest, self.y)
            self.arraymse = np.append(self.arraymse, msenow)
            self.arrayofY = np.append(self.arrayofY, self.y)
        self.msetest = np.mean(self.arraymse)
        print(f"Testing MSE: {self.msetest}")
        
#         grafik sewaktu training
    def graf(self):
        plt.plot(range(1, self.epochs+1), self.arrayError)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('MSE setiap Epoch')
        plt.show()
        
    def prediksi(self, data,xinput,yinput):
        self.databaruArray = denormalisasi(self.X, np.min(xinput), np.max(xinput))
        self.databaruArray = self.databaruArray[len(self.databaruArray)-1]
        r=len(self.databaruArray)
        self.dataY = denormalisasi(self.Y, np.min(yinput), np.max(yinput))
        self.dataY = self.dataY[len(self.dataY)-1]
        newlist=np.concatenate((np.squeeze(self.databaruArray[2:r]),self.dataY, np.array([data])))
        newlist=newlist.reshape(-1, 1)
        self.forward(newlist)
        self.hasil = denormalisasi(self.y, np.min(yinput), np.max(yinput))
        print("Hasil Prediksi : ", round(self.hasil[0,0]))

    def predict(self, listinput):#numpy type
        if len(listinput)!=self.x:
            print("Jumlah data Input harus", self.arch[0])
        else:
            self.dataInput = normalization(listinput)
            self.forward(self.dataInput)
            return self.y
    def simpanModel(self,name):       
        self.model_name=name
        model_data = {
            'W1': self.W1.tolist(),
            'W2': self.W2.tolist(),
            'b1' : self.b1.tolist(),
            'b2' : self.b2.tolist(),
            'err' : self.arrayError.tolist(),
            'timer' : self.waktu
        }
        
        with open(name, 'w') as file:
            json.dump(model_data, file)  
        print("Model telah disimpan dalam berkas:", name)
                
    def loadModel(self, jsonfile):
        with open(jsonfile, 'r') as file:
            loadedFile = json.load(file)
        self.W1 = np.array(loadedFile["W1"])
        self.W2 = np.array(loadedFile["W2"])
        self.b1 = np.array(loadedFile["b1"])
        self.b2 = np.array(loadedFile["b2"])
        self.arrayError = np.array(loadedFile["err"])
        self.waktu = loadedFile['timer']
        print("Berhasil Load Model",jsonfile)
                        
        
    