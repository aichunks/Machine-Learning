from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
#function to calculate gini index
def gini_index(groups,classes):
	ginindex=0
	data=float(sum([len(groups) for group in groups]))
	for group in groups:
		size=float(len(group))
		if size==0:
			continue
		proportion=0
		for values in classes:
			p = [row[-1] for row in group].count(values) / size
			#res=p.count(values/size)
			proportion+=p*p
		ginindex=(1-proportion)*(size/data)
	return ginindex
	
#function to split tree
def split(feature,value,dataset):
	left=[]
	right=[]
	for row in dataset:
		#print(row[feature])
		if row[feature]<value:
			left.append(row)
		else:
			right.append(row)
	return left,right
	
#function to find best split
def best_split(dataset):
	values=list(set(row[-1] for row in dataset))
	best_index, best_value, best_score, best_groups = 999, 999, 999, None
	for feature in range(len(dataset[0])-1):
		for row in dataset:
			groups=split(feature,row[feature],dataset)
			giniscore=gini_index(groups,values)
			if giniscore < best_score:
				best_index, best_value, best_score, best_groups = feature, row[feature], giniscore, groups
	return {'index':best_index, 'value':best_value, 'groups':best_groups}
		

#function of terminal nodes
def endnodes(group):
	result=[row[-1] for row in group]
	#print(max(set(result),key=result.count))
	return max(set(result),key=result.count)


#function to create tree
def create_tree(max_depth,min_size,node,depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left']=node['right']=endnodes(left+right)
		return
	if depth>=max_depth:
		node['left']=endnodes(left)
		node['right']=endnodes(right)
		return
	if len(left)<=min_size:
		node['left']=endnodes(left)
	else:
		node['left']=best_split(left)
		create_tree(max_depth,min_size,node['left'],depth+1)
	if len(right)<=min_size:
		node['right']=endnodes(right)
	else:
		node['right']=best_split(right)
		create_tree(max_depth,min_size,node['right'],depth+1)

	
#function to build final decision tree
def build_tree(train, max_depth, min_size):
	root = best_split(train)
	create_tree(max_depth, min_size,root, 1)
	
	return root
#print tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
#function to evaluate the split
def evaluate_tree(testdata,node):
	#print(tdata[node['index']])
	#print(node['value'])
	if testdata[node['index']]<node['value']:
		if isinstance(node['left'],dict):
			return evaluate_tree(testdata,node['left'])
		else:
			return node['left']
	else:
		if isinstance(node['right'],dict):
			return evaluate_tree(testdata,node['right'])
		else:
			return node['right']



iris=load_iris()
x,y=iris.data,iris.target
xtrain,ytrain,xlabel,ylabel=train_test_split(x,y,test_size=0.2)
data=np.c_[xtrain,xlabel]
cols=['sepal_length','sepal_width','petal_length','petal_width']
header=cols+['species']
iris_df=pd.DataFrame(data=data,columns=header)
#print(data)
data1=np.c_[ytrain,ylabel]
tree = build_tree(data, 30, 10)
#print_tree(tree)

predictions=[]
for i in data1:
	result=evaluate_tree(i,tree)
	predictions.append(result)
#print(predictions)
print(accuracy_score(predictions,ylabel))	
print(confusion_matrix(ylabel,predictions))
print(precision_score(ylabel,predictions,average='micro'))
print(recall_score(ylabel,predictions,average='micro'))
#res=evaluate_tree(data1,tree)
#result=predict_tree(data1)
#print(result)
#print(tree)









