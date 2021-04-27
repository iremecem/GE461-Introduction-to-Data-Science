import matplotlib.pyplot as plt
import numpy as np
from random import random
import pandas as pd
import sys

#Loading data to numpy arrays
f_train = open("./data_supervised_learning/train1", "r")
train_data = np.zeros(shape = (60,2),dtype = float)
count = 0
for line in f_train:
    temp = line[:-1].split("\t")
    train_data[count, 0] = temp[0]
    train_data[count, 1] = temp[1]
    count += 1

f_test = open("./data_supervised_learning/test1", "r")
test_data = np.zeros(shape = (41,2),dtype = float)
count = 0
for line in f_test:
    temp = line[:-1].split("\t")
    test_data[count, 0] = temp[0]
    test_data[count, 1] = temp[1]
    count += 1

class ANN:
    def __init__(self, n_hidden_nodes):
        self.n_hidden_nodes = n_hidden_nodes                         
        self.weights = np.zeros((n_hidden_nodes,3))
        self.weights[:,0] = np.array([random() for i in range(n_hidden_nodes)])
        self.weights[:,1] = np.array([random() for i in range(n_hidden_nodes)])
        self.weights[:,2] = 1/float(n_hidden_nodes)

        print("0", self.weights[:,0])
        print("1", self.weights[:,1])
        print("2", self.weights[:,2])
        
    def sigmoid(self, z):
        return 1 / (1 + (np.exp(- (z) )))
    
    def train(self, X, y, epoch, lr):
        losses = [0, 0,0,0]
        for i in range(epoch):

            index = np.random.randint(0,len(X))              
            w0 = self.weights[:,0]
            w1 = self.weights[:,1]
            wh = self.weights[:,2]
            
            forward = w0 + w1*X[index]
            hx = self.sigmoid(forward)
            output = sum(hx*wh)
            dervivative_sigmoid = hx * (1-hx)
            
            self.weights[:,0] += lr * (y[index] - output) * wh * dervivative_sigmoid
            self.weights[:,1] += lr * (y[index] - output) * wh * dervivative_sigmoid * X[index]
            self.weights[:,2] += lr * (y[index] - output) * hx
        
            forward = w0 + X.reshape(len(X),1) * w1
            hx = self.sigmoid(forward)
            predictions = np.dot(hx,wh)
            loss = np.sum((predictions - y)**2) 
            losses.append(loss)
            print("Epoch ", i , "loss",loss)
            if (np.sum(np.array(losses[-5:-1]))/5 >= losses[-1]):
                print(i)
                break
            
        
    def predict(self, X, y, isTest):
         w0 = self.weights[:,0]
         w1 = self.weights[:,1]
         wh = self.weights[:,2]
         
         forward = w0 + X.reshape(len(X),1) * w1
         hx = self.sigmoid(forward)
         predictions = np.dot(hx,wh)
         loss = np.sum((predictions - y)**2) 
         print(loss)
         plt.scatter(X,y, label = 'real value')
         plt.scatter(X, predictions, label='prediction')
         plt.legend()
         plt.xlabel('Inputs')
         plt.ylabel('Output')
         temp = 'Train'
         if(isTest == True):
             temp = 'Test'
         plt.title('ANN for Linear Regressor ' + temp + ' Data - Loss ' + str(loss)) 
         plt.show() 

        
epochs = 10000
lr = 0.001
model = ANN(16)
model.train(train_data[:,0], train_data[:,1], epochs, lr)
model.predict(train_data[:,0], train_data[:,1], False)

model.predict(test_data[:,0], test_data[:,1], True)