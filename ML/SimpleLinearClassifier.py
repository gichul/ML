'''
TO DO: Define a class __SimpleLinearClassifier__ composed of the following functions:

* __\_\_init\_\___(self, weight, bias) - takes a weight matrix and a bias vector as input and stores them in instance variables

* __predict(self, X)__ - takes an array of rank 2 as input, and returns the classification results as an array of rank 1

* __probability(self, X)__ - takes an array of rank 2 as input and returns the probabilities that each sample belongs to each class as an array of rank 2. (You need to implement the softmax function to convert logits into probabilities.)

* __score(self, X, y)__ - takes an input data (rank 2) and a target vector (rank 1) as input, and returns the accuracy as a scalar value.


The __SimpleLinearClassifier__ should work as the following examples.

'''
import numpy as np
import pickle
#data
with open('./W_lr','rb') as f:
    W_lr=pickle.load(f)

with open('./b_lr','rb') as f:
    b_lr=pickle.load(f)

# Define SimpleLienarClassfier here.
class SimpleLinearClassifier:
    def __init__(self,weight,bias):
        self.weight=weight
        self.bias=bias
    
    def predict(self,X):
        ans=[]
        self.t_weight=np.transpose(self.weight)
        result=np.dot(X,self.t_weight)+self.bias
        for i in range(len(result)):
            tmp=max(result[i])
            score_list=result[i].tolist()
            
            idx=score_list.index(tmp)
            ans.append(idx)
        return ans
    
    def softmax(self,x):
        exp_x=np.exp(x)
        y=exp_x/np.sum(exp_x)
        return y
        
    def probability(self,X):
        self.t_weight=np.transpose(self.weight)
        result=np.dot(X,self.t_weight)+self.bias
        soft_result=[]
        for i in range(len(result)):
            soft_result.append(self.softmax(result[i]))
        
        return np.array(soft_result)

    
    
    def score(self,X,y):
        self.t_weight=np.transpose(self.weight)
        result=np.dot(X,self.t_weight)+self.bias
        perfect=0
        for i in range(len(result)):
            tmp=max(result[i])
            score_list=result[i].tolist()
            
            idx=score_list.index(tmp)
            if idx==y[i]:
                perfect+=1
        ans=perfect/len(result)
        return ans



data=np.array([[1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00,
        3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00,
        1.065e+03],
       [1.320e+01, 1.780e+00, 2.140e+00, 1.120e+01, 1.000e+02, 2.650e+00,
        2.760e+00, 2.600e-01, 1.280e+00, 4.380e+00, 1.050e+00, 3.400e+00,
        1.050e+03]])
print(data)

s=SimpleLinearClassifier(W_lr,b_lr)
print(s) 
print(s.show())
