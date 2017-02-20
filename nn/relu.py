from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np

class ReLU(Module):
	counter = 0
	def __init__(self):
		super(ReLU,self).__init__()
		self.X = None
		ReLU.counter += 1
		self.name = 'ReLU_{}'.format(ReLU.counter)

	def relu(self,x):
		return x*(x>0)

	def forward(self,X,Y=None):
		self.X = X
		return self.relu(X)

	def backward(self, input_grad):
		gradient = self.update_grad_input(input_grad)
		self.update_parameters()
		return gradient

	def update_grad_input(self,input_grad):
		"""grad_input there should be equal to 1, because
		   it's usually the last layer in NN"""
		return input_grad*(self.X > 0)

	def update_parameters(self):
		pass

	def predict_proba(self,X):
		"""predict probability for each class
		this is one-vs-all probability, so be careful -
		probability vector isn't normalized"""
		return self.relu(X)

	def predict(self,X):
		"""predict label for given objects"""
		return 1.*(self.relu(X) >= np.max(self.relu(X),axis=1,keepdims=True))

if __name__ == "__main__":

	print("running gradient check for ReLU layer!")
	X = np.random.normal(loc=0.5,scale=0.5,size = (50,30))
	print("initializing input matrix with shape {}".format(X.shape))
	input_grad = np.ones(shape=X.shape) #identity input gradient
	eps = 1e-4 
	tol = 1e-4
	R = ReLU()

	num_grad = np.zeros(shape=X.shape)
	for i in xrange(X.shape[0]):
		for j in xrange(X.shape[1]):
			X[i,j] += eps
			Y_plus = R.forward(X)
			X[i,j] -= 2*eps
			Y_minus = R.forward(X)
			X[i,j] += eps
			num_grad[i,j] = (Y_plus[i,j] - Y_minus[i,j])/(2.*eps)
	
	R.forward(X) #memorize initial matrix
	print("Frobenius norm of difference is equal to:",np.linalg.norm(num_grad - R.backward(input_grad)))    
	print("Analytical and numerical gradient is equal:",np.linalg.norm(num_grad - R.backward(input_grad))<tol)
	print("sometimes it is false, because derivative in zero doesn't exist!")