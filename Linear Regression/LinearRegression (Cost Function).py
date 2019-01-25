'''
Created on 31-Jul-2018
@author: Sahil Agarwal
Data Scientist @  Dummy Bytes Intelligent Solutions
'''

import numpy as np
import matplotlib.pyplot as plt

#creating Sample Dataset (y = x*2)
data = np.linspace(0, 9, 10, dtype=int)
print("Input Data ",data)
label = data*2
print("Output Data ",label)

#Visualizing Dataset
plt.xlabel('Input Data: x')
plt.ylabel('Output Data: y')
plt.scatter(data,label)
plt.axis('square')
plt.show()



def getJ_theeta(w0,w1):
    predicted_label = []
    sum = 0
    for i in range(0,len(data)):
        h0x = w0 + w1*data[i]
        predicted_label.append(h0x)
        squared_error = (h0x - label[i])**2
        sum = sum + squared_error
    j_theeta = (sum/len(data))/2
    
    #Visualizing Actual and Predicted Function
    plt.xlabel('Input Data')
    plt.ylabel('Output Data')
    plt.scatter(data,label)
    plt.plot(data,predicted_label,'red')
    plt.title('Weight w1 = '+str(round(w1,6))+'    Error = '+str(round(j_theeta,6)))
    plt.legend(('Predicted at w1', 'Actual w1 = 2', 'composition', 'sharp interface'),)
    plt.axis('square')
    plt.show()
    
    return j_theeta

#Initializing weights and assuming w0(Intercept) is 0(Zero)
w0 = 0
w1 = 1
learning_rate = 0.1

error_list = []
w1_list = []
learning_rate_reverse_count = 0

while(True):
    error = getJ_theeta(w0,w1)
    w1_list.append(w1)
    error_list.append(error)
    
    #Setting Threshold of Error to break Loop
    if(error<0.01):
        break
    
    #Updating Learning rate and it's direction
    if(len(error_list)>1):
        
        if(error_list[len(error_list)-1]>error_list[len(error_list)-2]):
            # -ve LearningRate Only describes direction.
            learning_rate = -learning_rate
            learning_rate_reverse_count+= 1
            
        if(learning_rate_reverse_count > 4):
            learning_rate = learning_rate/10
            learning_rate_reverse_count = 0
    
    #when Learning rate is +ve or -ve, w1 is incremented and decremented respectively
    w1 = w1 + learning_rate

#Visualizing J_Theeta (Error Minimisation) with respect to w1
plt.xlabel('Weight (w1)')
plt.ylabel(r"J($\dot{\Theta}$) : Minimising Error")
plt.plot(w1_list,error_list,'red')
plt.scatter(w1_list,error_list)
plt.legend(('', 'Error at each W1', 'composition', 'sharp interface'),)
plt.show()