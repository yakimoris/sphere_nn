from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np

class SoftMax(Module):

    counter = 0
    def __init__(self):
    	super(SoftMax,self).__init__()
        self.output = None
        self.X = None
        self.S = None #softmax(X)

        SoftMax.counter += 1
        self.name = "SoftMax_{}".format(SoftMax.counter)

    def softmax(self,x):
    	return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

    def forward(self,X,Y=None):
        self.X = X
        self.S = self.softmax(self.X)
        return self.S

    def backward(self, input_grad):
        gradient = self.update_grad_input(input_grad)
        self.update_parameters()
        return gradient
      
    def update_grad_input(self,input_grad):
        return input_grad*self.S - self.S*np.sum(input_grad*self.S,axis=1,keepdims=True)

    def update_parameters(self):
        pass
    
    def predict_proba(self,X):
        """predict probability for each class"""
        return self.softmax(X)

    def predict(self,X):
        """predict label for given objects"""
        return 1.*(self.softmax(X) >= np.max(self.softmax(X),axis=1,keepdims=True))

if __name__ == "__main__":

    print("running gradient check for Softmax layer!")
    X = np.random.normal(loc=0.5,scale=0.5,size = (40,20))
    print("initializing feature matrix with shape {}".format(X.shape))
    eps = 1e-4
    tol = 1e-4 

    num_grad = np.zeros(shape=(X.shape[1],X.shape[0],X.shape[1]))
    an_grad = np.zeros(shape=(X.shape[1],X.shape[0],X.shape[1]))

    S = SoftMax()

    for i in xrange(X.shape[1]): #3d tensor of gradients
        input_grad = np.zeros(shape=X.shape)
        input_grad[:,i] = np.ones(shape=input_grad.shape[0]) #identity gradient to obtain derivative on current layer
        S.forward(X)
        an_grad[i,:,:] = S.backward(input_grad)
        for l in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                X[l,j] += eps
                Y_plus = S.forward(X)
                X[l,j] -= 2*eps
                Y_minus = S.forward(X)
                num_grad[i,l,j] = (Y_plus[l,i] - Y_minus[l,i])/(2.*eps)
                X[l,j] += eps

    print("Frobenius norm of difference is equal to:",np.linalg.norm(num_grad - an_grad))
    print("Analytical and numerical gradient is equal:",np.linalg.norm(num_grad - an_grad)<tol)   