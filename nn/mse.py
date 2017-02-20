from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


class MSE(Module):
	counter = 0
	def __init__(self):
		super(MSE,self).__init__()
		self.X = None #input data
		self.T = None #input targets 
		MSE.counter += 1
		self.name ="MSE_{}".format(MSE.counter)

	def forward(self, X,Y=None):
		"""in forward pass this layer returns mean squared error
		X - input matrix, Y - true labels of objects"""
		self.X = X
		self.batch_size = self.X.shape[0]
		self.T = Y
		self.output = (1./(2*self.batch_size))*np.sum((self.T - self.X)**2)
		return self.output

	def backward(self, input_grad=None):
		gradient = self.update_grad_input(input_grad)
		self.update_parameters()
		return gradient
	  
	def update_grad_input(self,input_grad=None):
		"""input_grad here should be equal to 1, or even shouldn't be used,
		   because it's usually the last layer in NN"""
		return (1./self.batch_size)*(self.X - self.T)

	def update_parameters(self):
		pass
if __name__ == "__main__":

	print("running gradient check for MSE layer!")
	X = np.random.uniform(low=1e-9,high = 1.0,size = (50,30))
	print("initializing feature matrix with shape {}".format(X.shape))
	T = 1.*(X>=np.max(X,axis=1,keepdims=True)) #random labels

	eps = 1e-4
	tol = 1e-4 

	mse = MSE()

	num_grad = np.zeros(shape=(X.shape[0],X.shape[1]))

	for i in xrange(X.shape[0]):
		for j in xrange(X.shape[1]):
			X[i,j] += eps
			Y_plus = mse.forward(X,T)
			X[i,j] -= 2*eps
			Y_minus = mse.forward(X,T)
			X[i,j] += eps
			num_grad[i,j] = (Y_plus - Y_minus)/(2.*eps)

	an_grad = mse.backward()
	print("Frobenius norm of difference is equal to:",np.linalg.norm(num_grad - an_grad))
	print("Analytical and numerical gradient is equal:",np.linalg.norm(num_grad - an_grad)<tol)