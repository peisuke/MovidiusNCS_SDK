# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Models/CaffeEnumDeclarations.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 911 bytes
from enum import Enum

class CaffeStage(Enum):
    NONE = 0
    ABSVAL = 35
    ACCURACY = 1
    ARGMAX = 30
    BNLL = 2
    CONCAT = 3
    CONTRASTIVE_LOSS = 37
    CONVOLUTION = 4
    DATA = 5
    DECONVOLUTION = 39
    DROPOUT = 6
    DUMMY_DATA = 32
    EUCLIDEAN_LOSS = 7
    ELTWISE = 25
    EXP = 38
    FLATTEN = 8
    HDF5_DATA = 9
    HDF5_OUTPUT = 10
    HINGE_LOSS = 28
    IM2COL = 11
    IMAGE_DATA = 12
    INFOGAIN_LOSS = 13
    INNER_PRODUCT = 14
    LRN = 15
    MEMORY_DATA = 29
    MULTINOMIAL_LOGISTIC_LOSS = 16
    MVN = 34
    POOLING = 17
    POWER = 26
    RELU = 18
    SIGMOID = 19
    SIGMOID_CROSS_ENTROPY_LOSS = 27
    SILENCE = 36
    SOFTMAX = 20
    SOFTMAX_LOSS = 21
    SPLIT = 22
    SLICE = 33
    TANH = 23
    WINDOW_DATA = 24
    THRESHOLD = 31
    RESHAPE = 40
# okay decompiling CaffeEnumDeclarations.pyc
