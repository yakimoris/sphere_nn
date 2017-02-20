from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from nn.module import Module

class Linear(Module):
    counter = 0
    def __init__(self,n_input,n_output,learning_rate=0.001,lmbda=0.001):
        super(Linear,self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.weights = np.random.normal(loc=0.,scale=0.5/max(self.n_output,self.n_input),
                                        size=(self.n_output,self.n_input))
        self.bias = np.zeros(shape=self.n_output)
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.X = None #input data
        self.batch_size = None
        Linear.counter += 1
        self.name ="Linear_(in={0},out={1})_{2}".format(self.n_input,self.n_output,Linear.counter)
    
    def reset_params(self):
        self.weights = np.random.normal(loc=0.,scale=0.5/max(self.n_output,self.n_input),
                                        size=(self.n_output,self.n_input))
        self.bias = np.zeros(shape=self.n_output)

    def forward(self,X,Y=None):
        self.X = X
        self.batch_size = self.X.shape[0]
        self.output = np.dot(self.X,self.weights.T) + self.bias
        return self.output

    def backward(self, input_grad):
        gradient = self.update_grad_input(input_grad)
        self.update_parameters() #I'm not sure that it's correct. Where should learning rate be? 
        
        return gradient
    
    def update_grad_input(self, input_grad):
        """compute gradient in linear layer with respect to input gradient
        input_grad df/dx. The input_grad is (b x n_output)-matrix 
        where b is a batch size, and n_output - is numbers of output
        (w is n_output x n_input matrix))"""
        
        self.input_grad = input_grad
        self.gradient_weights = np.dot(self.input_grad.T,self.X) + self.lmbda*self.weights
        self.gradient_bias = np.dot(self.input_grad.T, np.ones(shape=(self.batch_size,1)))
        self.gradient_bias = np.reshape(self.gradient_bias,newshape=self.bias.shape[0]) #make broadcasting great again!
        return np.dot(self.input_grad,self.weights)
        

    def update_parameters(self):
        self.weights = self.weights - self.learning_rate*self.gradient_weights
        self.bias = self.bias - self.learning_rate*self.gradient_bias

if __name__ == "__main__":

    print("running gradient check for linear layer!")
    
    X = np.random.normal(loc=0.5,scale=0.5,size = (50,20))
    print("initialize feature matrix with shape {}".format(X.shape))
    
    n_input = 20
    n_output = 10 #number of input and output features
    input_grad = np.eye(n_output) #identity input gradient 
                                  #to obtain derivative exactly on this layer 
    L = Linear(n_input,n_output)

    eps = 1e-4
    tol = 1e-4
    print('eps = ',eps,'tolerance = ',tol)
    num_grad = np.zeros(shape=(n_output,n_input))
    for l in xrange(X.shape[0]):
        num_grad_per_object = np.zeros(shape=(n_output,n_input))
        for i in xrange(n_output):
            for j in xrange(n_input):
                X[l,j] += eps
                Y_plus = L.forward(X)
                X[l,j] -= 2*eps
                Y_minus = L.forward(X)
                X[l,j] += eps
                num_grad_per_object[i,j] = ((Y_plus[l,i] - Y_minus[l,i])/(2.*eps))
        num_grad += (1./X.shape[0])*num_grad_per_object

    print("Frobenius norm is equal to:",np.linalg.norm(num_grad - L.update_grad_input(input_grad)))    
    print("Analytical and numerical gradient is equal:",np.linalg.norm(num_grad - L.update_grad_input(input_grad))<tol)