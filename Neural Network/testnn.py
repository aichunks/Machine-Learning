import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

filename='nndata.csv'

dataset=pd.read_csv(filename)
#print(dataset)
X=dataset.iloc[:,1:10].values
#print(X)
y=dataset.iloc[:,-1].values
#print(y)
size=len(y)
#print(size)
y=y.reshape(size,1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#print(X_train[0])


#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return x * (1 - x)
def predict(w_h,w_o,b_o,b_h,xtest,ytest):
	wh=w_h
	wout=w_o
	bh=b_h
	bout=b_o 
	input=xtest
	#print(synap_h)
	#print(synap_o)
	#print(bias_hidd)
	#print(bias_out)
	hiddenlayer=sigmoid(np.dot(input,wh)+bh)
	outputlayer=sigmoid(np.dot(hiddenlayer,wout)+bout)
	#print(outputlayer)
	#print(np.round(outputlayer))
	print(accuracy_score(np.round(outputlayer),ytest))
	print(confusion_matrix(np.round(outputlayer),ytest))
#Variable initialization
a=0
epoch=1000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = 9 #number of features in data set
#print(inputlayer_neurons)
hiddenlayer_neurons = 5 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(9,5))
bh=np.random.uniform(size=(1,5))

wout=np.random.uniform(size=(5,1))
#print(wout)
bout=np.random.uniform(size=(1,1))
#print(bout)
prediction=[]
for i in range(epoch):

#Forward Propogation
	hidden_layer_input1=np.dot(X_train,wh)
	hidden_layer_input=hidden_layer_input1 + bh
	hiddenlayer_activations = sigmoid(hidden_layer_input)
	output_layer_input1=np.dot(hiddenlayer_activations,wout)
	output_layer_input= output_layer_input1+ bout
	output = sigmoid(output_layer_input)

	#Backpropagation
	E = Y_train-output
	slope_output_layer = derivatives_sigmoid(output)
	slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
	d_output = E * slope_output_layer
	#print(d_output)
	Error_at_hidden_layer = d_output.dot(wout.T)
	d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
	wout += hiddenlayer_activations.T.dot(d_output) *lr
	bout += np.sum(d_output, axis=0,keepdims=True) *lr
	wh += X_train.T.dot(d_hiddenlayer) *lr
	bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
print(accuracy_score(np.round(output),Y_train))
#print(confusion_matrix(np.round(output),Y_train))
predict(wh,wout,bout,bh,X_test,Y_test)
#print(np.round(output),Y_train)
#print(accuracy_score(np.round(output),Y_train))
#print(confusion_matrix(np.round(output),Y_train))

