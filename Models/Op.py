# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Models/Op.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 208 bytes


class Op:

    def __init__(self, name, conf_name):
        self.name = name
        self.optimization_name = conf_name
        self.opt_list = []

    def add_opt(self, conf):
        self.opt_list += [conf]
# okay decompiling Op.pyc
