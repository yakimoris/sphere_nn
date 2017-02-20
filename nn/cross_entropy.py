from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


class CrossEntropyCriterion(Module):
	counter = 0
	def __init__(self):
		super(CrossEntropyCriterion,self).__init__()
		self.output = None
		self.grad_input = None
		self.X = None #input data

		CrossEntropyCriterion.counter += 1
		self.name ="CrossEntropy_{}".format(CrossEntropyCriterion.counter)

	def softmax(self,x):
		return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

	def forward(self, X,Y=None):
		"""in forward pass this layer returns loss
		X - feature matrix, Y - true labels of objects"""
		self.X = X
		self.S = self.softmax(self.X)
		self.LogS = np.log(	np.abs(self.S))
		#print("This is log(S)",self.S)
		self.T = Y
		self.output = -np.sum(self.T*self.LogS)
		return self.output

	def predict_proba(self,X):
		"""predict probability for each class"""
		return self.softmax(X)

	def predict(self,X):
		"""predict label for given objects"""
		return 1.*(self.softmax(X) >= np.max(self.softmax(X),axis=1,keepdims=True))

	def backward(self, input_grad=None):
		gradient = self.update_grad_input(input_grad)
		self.update_parameters()
		return gradient
	  
	def update_grad_input(self,input_grad=None):
		"""grad_input there should be equal to 1, because
		   it's usually the last layer in NN"""
		return (self.S - self.T)

	def update_parameters(self):
		pass

if __name__ == "__main__":

	print("running gradient check for CrossEntropyCriterion layer!")
	X = np.random.uniform(low=1e-9,high = 1.0,size = (50,30))
	print("initializing feature matrix with shape {}".format(X.shape))
	T = 1.*(X>=np.max(X,axis=1,keepdims=True)) #random labels

	eps = 1e-4
	tol = 1e-4 

	CEC = CrossEntropyCriterion()

	num_grad = np.zeros(shape=(X.shape[0],X.shape[1]))

	for i in xrange(X.shape[0]):
		for j in xrange(X.shape[1]):
			X[i,j] += eps
			Y_plus = CEC.forward(X,T)
			X[i,j] -= 2*eps
			Y_minus = CEC.forward(X,T)
			X[i,j] += eps
			num_grad[i,j] = (Y_plus - Y_minus)/(2.*eps)

	an_grad = CEC.backward()
	print("Frobenius norm of difference is equal to:",np.linalg.norm(num_grad - an_grad))
	print("Analytical and numerical gradient is equal:",np.linalg.norm(num_grad - an_grad)<tol)