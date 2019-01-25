'''
Created on 01-Aug-2018

@author: Sahil Agarwal
Data Scientist @  Dummy Bytes Intelligent Solutions
'''


import numpy as np
import matplotlib.pyplot as plt

#creating Sample Dataset (y = x*4 + 5)
data = np.linspace(0, 9, 10, dtype=int)
print("Input Data ",data)
label = data*4 + 5
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
    #Uncomment if required to see graphs, It may generate thousands
    '''
    plt.xlabel('Input Data')
    plt.ylabel('Output Data')
    plt.scatter(data,label)
    plt.plot(data,predicted_label,'red')
    plt.title('Weight w0 = '+str(round(w0,6))+'    Weight w1 = '+str(round(w1,6))+'    Error = '+str(round(j_theeta,6)))
    plt.legend(('Predicted at w1, w2', 'Actual w1, w2', 'composition', 'sharp interface'),)
    plt.axis('square')
    plt.show()
    '''
    return j_theeta

def findw0(w0,w1):
    sum = 0
    for i in range(0,len(data)):
        h0x = w0 + w1*data[i]
        error = h0x - label[i]
        sum = sum + error
    return sum/len(data)

def findw1(w0,w1):
    sum = 0
    for i in range(0,len(data)):
        h0x = w0 + w1*data[i]
        error = (h0x - label[i])*data[i]
        sum = sum + error
    return sum/len(data)

w0 = 1
w1 = -1
#keeping learning_rate 0.1 doesn't converge so keeping it 0.01
learning_rate = 0.01
error_list = []
w0_list = []
w1_list = []
while(True):
    error = getJ_theeta(w0,w1)
    w0_list.append(w0)
    w1_list.append(w1)
    error_list.append(error)
    
    #Setting Threshold of Error to break Loop
    if(error<0.00000000001):
        break
    
    #Updating Learning rate's direction
    if(len(error_list)>1):
        
        if(error_list[len(error_list)-1]>error_list[len(error_list)-2]):
            # -ve LearningRate Only describes direction.
            learning_rate = -learning_rate

    #Updating w0 and w1
    t0 = w0 - learning_rate*(findw0(w0,w1))
    t1 = w1 - learning_rate*(findw1(w0,w1))
    w0 = t0
    w1 = t1
    print(w0, " ", w1)


#Visualizing the values of w0 and w1 reaching to their actual value
plt.xlabel('Weight (w0)')
plt.ylabel('Weight (w1)')
plt.plot(w0_list,w1_list,'red')
plt.scatter(w0_list,w1_list)
plt.show()
