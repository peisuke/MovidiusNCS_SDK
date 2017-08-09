# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/Scheduler.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 1832 bytes
import numpy as np
import os
import sys
from Models.Network import *
from Models.NetworkStage import *
from Models.MyriadParam import *
from Models.EnumDeclarations import *

def load_myriad_config(no_shaves):
    return MyriadParam(0, no_shaves - 1)


def load_network(arguments, parser, myriad_conf, file_gen=False):
    if parser == Parser.Debug:
        throw_error(ErrorTable.ParserNotSupported)
    else:
        if parser == Parser.TensorFlow:
            from Controllers.TensorFlowParser import parse_tensor
            network = parse_tensor(arguments, myriad_conf, file_gen=file_gen)
            network.finalize()
            network.optimize()
            return network
        if parser == Parser.Caffe:
            from Controllers.CaffeParser import parse_caffe
            network = parse_caffe(arguments, myriad_conf, file_gen=file_gen)
            print('____________ PARSED ________________')
            network.debug()
            network.finalize()
            print('____________ FINALIZED ________________')
            network.debug()
            network.optimize()
            print('____________ OPTIMIZED ________________')
            network.debug()
            print('____________ DONE ________________')
            return network
        throw_error(ErrorTable.ParserNotSupported, parser.name)
# okay decompiling Scheduler.pyc
