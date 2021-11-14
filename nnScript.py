import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pandas as pd


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return (1.0 / (1.0 + np.exp(-z)))


featureIndices=[]
def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.

    train_data, train_label, validation_data, validation_label = preprocess_train_data(mnist)
    test_data, test_label = preprocess_test_data(mnist)
    
    # Feature selection
    combined  = np.concatenate((train_data, validation_data),axis=0)
    reference = combined[0,:]
    boolean_value_columns = np.all(combined == reference, axis = 0)
    
    
    featureCount = 0
    global featureIndices
    
    for i in range(len(boolean_value_columns)):
        if boolean_value_columns[i]==False:
            featureCount += 1
            featureIndices.append(i)
    print("Total number of selected features : ", featureCount)
    
    final = combined[:,~boolean_value_columns]
    tr_R = train_data.shape[0]
    vl_R = validation_data.shape[0]
    
    train_data      = final[0:tr_R,:]
    validation_data = final[tr_R:,:]
    test_data = test_data[:,~boolean_value_columns]

    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label

def preprocess_train_data(mnist):
    """This is called from preprocess()
    to process training data"""
    
    label_lst = []
    for i in range(10):
        idx = 'train'+ str(i)
        train_mat = mnist[idx]
        labels = np.full((train_mat.shape[0],1),i)
        labeled_train_mat = np.concatenate((train_mat,labels),axis=1)
        label_lst.append(labeled_train_mat)

    all_labeled_train = np.concatenate((label_lst[0],label_lst[1],label_lst[2],
                                        label_lst[3],label_lst[4],label_lst[5],
                                        label_lst[6],label_lst[7],label_lst[8],
                                        label_lst[9]), axis=0)
    
    np.random.shuffle(all_labeled_train)
    
    labeled_train = all_labeled_train[0:50000,:]
    train_data    = (labeled_train[:,0:784])/255.0
    train_label   = labeled_train[:,784]

    labeled_validation = all_labeled_train[50000:60000,:]
    validation_data    = (labeled_validation[:,0:784])/255.0
    validation_label   = labeled_validation[:,784]
    
    return train_data, train_label, validation_data, validation_label

def preprocess_test_data(mnist):
    """This is called from preprocess()
    to process test data"""
        
    label_lst = []
    for i in range(10):
        idx = 'test'+ str(i)
        test_mat = mnist[idx]
        labels = np.full((test_mat.shape[0],1),i)
        labeled_test_mat = np.concatenate((test_mat,labels),axis=1)
        label_lst.append(labeled_test_mat)

    all_labeled_test = np.concatenate((label_lst[0],label_lst[1],label_lst[2],
                                       label_lst[3],label_lst[4],label_lst[5],
                                       label_lst[6],label_lst[7],label_lst[8],
                                       label_lst[9]), axis=0)

    np.random.shuffle(all_labeled_test)
    
    test_data    = (all_labeled_test[:,0:784])/255.0
    test_label   = all_labeled_test[:,784]
    
    return test_data, test_label


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
   
    #Error function            
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


    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    
    # Your code here
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


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0.7

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
