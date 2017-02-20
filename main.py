from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import nn

def main():
    model = nn.Sequential()
    model.add(nn.Linear())
    model.add(nn.LogSoftMax())
    model.add(nn.CrossEntropyCriterion())

    loss = model.forward()


if __name__ == '__main__':
    main()