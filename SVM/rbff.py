import numpy as np
import math
import csv
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

result=[]

def rbf(x, xt):
    gamma = 0.7
    return np.exp(-gamma * (np.linalg.norm(x - xt))** 2)
def train(X,Y):
    X_count = X.shape[0]
    #print(X_count)
    alpha = np.zeros(X_count)
    #print(alpha)
   
    K = np.zeros((X_count, X_count))
    for i in range(X_count):
        for j in range(X_count):
            K[i,j] = rbf(X[i], X[j])
            #print(K[i,j])
    #max_iterations = 10
    for ite in range(length):
        for i in range(X.shape[0]):   
            #print("start")
            sum = 0
            val = 0
            for j in range(X.shape[0]):
                #print("test")
                val= alpha[j] * Y[j] * K[i,j]    
                sum = sum + val
            if sum <= 0:
                val = -1
            elif sum >0:
                val = 1
            if val != Y[i]:
                alpha[i] = alpha[i] + 1
    #print(alpha)
    return alpha
pred=[]
def predictions(train_X,train_Y,test_X,test_Y,alpha):
    m = test_Y.size
    corrrect_pred = 0
    
    for i in range(m):
        s = 0
        for a, x_train,y_train  in zip(alpha, train_X,train_Y):
            s += a * y_train * rbf(test_X[i],x_train)
            #pred.append(s)
        
        if s >0:
            s = 1
            pred.append(s)
        elif s <=0:
            s = -1
            pred.append(s)
        if test_Y[i] == s:
            corrrect_pred +=1
    print(pred)
    #print(right)
  
    #print (" Correct : ",right," Accuracy : ",right*100/test_X.shape[0])
        #return y_predict	
    #return pred
	
filename='testt.csv'

dataset=pd.read_csv(filename)
X=dataset.iloc[:,0:2].values
Y=dataset.iloc[:,2].values
#print(X,Y)
length=len(X)
#print(length)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# iris=load_iris()
# x,y=iris.data,iris.target
# xtrain,ytrain,xlabel,ylabel=train_test_split(x,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#print(X_train)

#for i in range(len(X_train)):
        #print(i)
theta= train(X_train,Y_train)
predictions(X_train,Y_train,X_test,Y_test,theta)
print(accuracy_score(pred,Y_test))