# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Models/MyriadParam.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 1376 bytes
import numpy as np
from ctypes import *
from Controllers.MiscIO import *

class MyriadParam:

    def __init__(self, fs=0, ls=1, optimization_list=None):
        self.firstShave = c_ushort(fs)
        self.lastShave = c_ushort(ls)
        self.leonMemLocation = c_uint(0)
        self.leonMemSize = c_uint(0)
        self.dmaAgent = c_uint(0)
        self.optimization_list = optimization_list

    def generate(self, f):
        f.write(self.firstShave)
        f.write(self.lastShave)
        f.write(self.leonMemLocation)
        f.write(self.leonMemSize)
        f.write(self.dmaAgent)

    def binary_size(self):
        file_size = byte_size(self.firstShave)
        file_size += byte_size(self.lastShave)
        file_size += byte_size(self.leonMemLocation)
        file_size += byte_size(self.leonMemSize)
        file_size += byte_size(self.dmaAgent)
        return file_size

    def display_opts(self):
        print('\nAvailable Optimizations:')
        [print('* ' + str(x)) for x in self.optimization_list]
# okay decompiling MyriadParam.pyc
