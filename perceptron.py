# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:10:21 2018

@author: gautam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self , learning_rate = 0.01 , n_iter = 10):
        
        self.learning_rate = learning_rate
    
        self.n_iter = n_iter
    
    def fit(self , X , y):
        
        self.w_ = np.zeros(1 + X.shape[1])
        
        self.errors = []
        
        for _ in range(self.n_iter):
            
            errors = 0
            
            for xi , target in zip(X , y):
            
                update = self.learning_rate * (target - self.predict(xi))
                
                self.w_[1:] += update * xi
                
                self.w_[0] += update
                
                errors += int(update != 0.0)
            
            self.errors.append(errors)
        
        return self
    
    def net_input(self , X):
        
        return np.dot(X,self.w_[1:]) + self.w_[0] #F(W,x) = w1.x1+w2.x2
    
    def predict(self, X):
        
        return np.where(self.net_input(X) >= 0.0 , 1 ,-1)
       
df = pd.read_csv('iris.csv') 

y = df.iloc[0:100 , 4].values

y = np.where( y == 'setosa' , -1 , 1)

X = df.iloc[0:100 , 0:4].values

#print(X)

plt.scatter(X[0:50 , 0] , X[0:50 ,1 ] , color='blue' , marker = 'o' ,label='setosa')

plt.scatter(X[50:100 , 0] , X[50:100,1 ] , color='red' , marker = 'x' ,label='versicolor') 

plt.xlabel('sepal_length')

plt.ylabel('sepal_width')

plt.legend(loc = 'upper-left')

plt.show()

p_object = Perceptron( learning_rate = 0.1 , n_iter = 10)

p_object.fit(X,y)

print('The updated weight matrix->',p_object.w_)

plt.plot(range(1 , len(p_object.errors) + 1),p_object.errors)

plt.xlabel('Attempts')

plt.ylabel('Number of misclassification')

plt.show()

print('Enter values for prediction:(sep_len,sep_wid,petal_len,petal_wid)')

go = 'y'
while go == 'y':
    x , y , z ,w = map(float , input().split())
    if p_object.predict([x , y , z ,w]) == -1:
        print('The flower is setosa')
    else:
        print('The flower is versicolo  r')
    go = input('Do yo want to continue:(y/n)')
    


           