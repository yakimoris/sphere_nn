from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Module(object):
    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, X,Y=None):
        raise NotImplementedError('implement forward pass!')

    def backward(self, *args, **kwargs):
        self.update_grad_input(*args, **kwargs)
        self.update_parameters(*args, **kwargs)

    def update_grad_input(self, *args, **kwargs):
        raise NotImplementedError('implement computation of gradient w.r.t. input! df(x)/dx!')

    def update_parameters(self, *args, **kwargs):
        # that's fine not to implement this method
        # module may have not parameters (for example - MSE criterion)
        pass
