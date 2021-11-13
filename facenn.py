#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = np.sqrt(6) / np.sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W




# In[2]:


# Replace this with your sigmoid implementation
def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))


# In[3]:


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
   
    # Your code here
    shape_ip=training_data.shape
    
    n_rows_ip=shape_ip[0]
    
    biases_ip=np.full((n_rows_ip,1),1)
    
    aj=np.dot(np.concatenate((biases_ip,training_data),axis=1),np.transpose(w1))
              
    zj=sigmoid(aj)
   
   
    #Hidden
    shape_hidden=zj.shape
    n_rows_hidden=shape_hidden[0]
   
    biases_hidden=np.full((n_rows_hidden,1),1)
    m=np.concatenate((biases_hidden,zj),axis=1)
    bl=np.dot(m,np.transpose(w2))
   
    ol=sigmoid(bl)
    #ERROR
    
    yl=np.zeros((n_rows_ip,n_class))
   
    for i in range(n_rows_ip):
        yl[i,train_label[i]]=1
   
    #error function
             
    error=np.sum(np.multiply(yl,np.log(ol))+np.multiply((1-yl),np.log(1-ol)))/(-1*n_rows_ip)
   
        
    #Gradient
             
    gradient2=np.dot(np.transpose((ol-yl)),m)
   
    gradient1=np.dot(np.transpose(np.dot((ol-yl),w2)*(m*(1-m))),np.concatenate((biases_ip,training_data),axis=1))         
    gradient1=gradient1[1:,:]
    
    #Regularization
             
    obj_val=error+(lambdaval/(2*n_rows_ip))*(np.sum(w1**2)+np.sum(w2**2))
    
    gradient1reg = (gradient1 + lambdaval * w1)/n_rows_ip
    gradient2reg = (gradient2 + lambdaval * w2)/n_rows_ip

    obj_grad = np.concatenate((gradient1reg.flatten(), gradient2reg.flatten()), 0)

    return (obj_val, obj_grad)


# In[4]:


# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    n = data.shape[0]

    biases1 = np.full((n,1),1)
    training_data = np.concatenate((biases1, data), axis=1)

    aj = np.dot(training_data, w1.T)
    zj = sigmoid(aj)
    
    m = zj.shape[0]
    
    biases2 = np.full((m,1), 1)
    zj = np.concatenate((biases2, zj), axis=1)

    bl = np.dot(zj, w2.T)
    ol = sigmoid(bl)

    labels = np.argmax(ol, axis=1)

    return labels


# In[5]:


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


# In[6]:


"""**************Neural Network Script Starts here********************************"""
from scipy.optimize import minimize
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')


# In[ ]:





# In[ ]:




