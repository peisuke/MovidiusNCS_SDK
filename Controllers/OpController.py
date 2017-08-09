# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/OpController.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 490 bytes
from Models.Convolution import *
from Models.MaxPooling import *

class OpController:

    def __init__(self):
        self.stages = [
         Convolution(),
         MaxPooling()]

    def add_available_optimization(self, op, conf):
        for s in self.stages:
            if op == s.optimization_name:
                s.add_opt(conf)
                return
# okay decompiling OpController.pyc
