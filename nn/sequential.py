from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
from collections import OrderedDict
import numpy as np
class Sequential(Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.layer_sequence = OrderedDict()     

    def add(self, module):
        self.layer_sequence[module.name] = module

    def remove(self, module_name):
        if module_name in self.layer_sequence.keys():
            del self.layer_sequence[module_name]
            print('module successfully deleted')
        else:
            print('Such module not presented in sequence!')

    def added_layers(self):
        """this function print added 
        in neural networks layers"""
        for layer in self.layer_sequence.keys():
            print(layer)
            
    def forward(self, X,Y=None):
        input_data = X
        for module in self.layer_sequence.values():
            input_data = module.forward(input_data,Y)
        loss = input_data
        return loss

    def backward(self, *args, **kwargs):
        input_grad = None
        for module in self.layer_sequence.values()[::-1]:
            input_grad = module.backward(input_grad)

    def predict(self,X):
        input_data = X
        last_module_name = self.layer_sequence.values()[-1].name.split('_')[0]
        if last_module_name == 'CrossEntropy':
            for module in self.layer_sequence.values()[:-1]:
                input_data = module.forward(input_data)
            return self.layer_sequence.values()[-1].predict(input_data)
        else:
            for module in self.layer_sequence.values()[:-2]:
                input_data = module.forward(input_data)
            return self.layer_sequence.values()[-2].predict(input_data)

    def predict_proba(self,X):
        input_data = X
        last_modul_name = self.layer_sequence.values()[-1].name.split('_')[0]

        if last_module_name == 'CrossEntropy':
            for module in self.layer_sequence.values()[:-1]:
                input_data = module.forward(input_data)
            return self.layer_sequence.values()[-1].predict_proba(input_data)
        else:
            for module in self.layer_sequence.values()[:-2]:
                input_data = module.forward(input_data)
            return self.layer_sequence.values()[-2].predict_proba(input_data)

#def gradient_check(self,layer_to_check,eps):
#        module = self.layer_sequence['layer_to_check']