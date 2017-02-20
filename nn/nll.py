from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


class NLL(Module):
	counter = 0
	def __init__(self):
		super(NLL,self).__init__()
		self.X = None #input data
		self.T = None #input targets 
		NLL.counter += 1
		self.name ="NLL_{}".format(NLL.counter)

	def forward(self, X,Y=None):
		"""in forward pass this layer returns loss
		X - input matrix, Y - true labels of objects"""
		self.X = X
		self.T = Y
		self.output = -np.sum(self.T*np.log(self.X))
		return self.output

	def backward(self, input_grad=None):
		gradient = self.update_grad_input(input_grad)
		self.update_parameters()
		return gradient
	  
	def update_grad_input(self,input_grad=None):
		"""grad_input there should be equal to 1, or even shouldn't be used,
		   because it's usually the last layer in NN"""
		return -self.T/self.X

	def update_parameters(self):
		pass
if __name__ == "__main__":

	print("running gradient check for NLL layer!")
	X = np.random.uniform(low=1e-9,high = 1.0,size = (50,30))
	print("initializing feature matrix with shape {}".format(X.shape))
	T = 1.*(X>=np.max(X,axis=1,keepdims=True)) #random labels

	eps = 1e-4
	tol = 1e-4 

	nll = NLL()

	num_grad = np.zeros(shape=(X.shape[0],X.shape[1]))

	for i in xrange(X.shape[0]):
		for j in xrange(X.shape[1]):
			X[i,j] += eps
			Y_plus = nll.forward(X,T)
			X[i,j] -= 2*eps
			Y_minus = nll.forward(X,T)
			X[i,j] += eps
			num_grad[i,j] = (Y_plus - Y_minus)/(2.*eps)

	an_grad = nll.backward()
	print("Frobenius norm of difference is equal to:",np.linalg.norm(num_grad - an_grad))
	print("Analytical and numerical gradient is equal:",np.linalg.norm(num_grad - an_grad)<tol)