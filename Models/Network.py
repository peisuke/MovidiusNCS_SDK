# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Models/Network.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 8406 bytes
from Controllers.MiscIO import *
import numpy as np
from Models.NetworkStage import *
import ctypes

class Network:

    def __init__(self, name, data):
        self.name = name
        self.head = []
        self.count = 0
        self.stageslist = []
        self.outputInfo = None
        self.outputNeedsTransforming = False
        self.outputTensor = None
        self.inputTensor = data
        self.datatype = DataType.fp16

    def attach(self, stage, debug=False):
        self.stageslist.append(stage)
        if len(self.head) == 0:
            self.head.append(stage)
            stage.data = self.inputTensor
            stage.dataIndex = MemoryIndex.input.value
            self.storageOrder = stage.storageOrder
            self.count = 1
            if debug:
                print('attached.')
            return 1
        if stage.top is None:
            stage.data = self.inputTensor
            stage.dataIndex = MemoryIndex.input.value
            self.head.append(stage)
            self.count += 1
        elif len(stage.top) > 1:
            appropriate_nodes = self.search_several(stage.top)
            stage.attach_eltwise(appropriate_nodes)
            if debug:
                print('attached.')
            self.count += 1
        else:
            parent = stage.top[0]
            appropriate_nodes = self.search_several(parent)
            if appropriate_nodes == 0:
                throw_error(ErrorTable.GraphConstructionFailure, parent)
            else:
                stage.attach_several(appropriate_nodes)
                self.count += 1
                if debug:
                    print('attached.')
            return 1

    def search(self, seek_name):
        if seek_name == 0:
            throw_error(ErrorTable.GraphConstructionFailure, seek_name)
        for stage in self.head:
            ret = stage.search(seek_name)
            if ret != 0:
                return ret

        return 0

    def search_several(self, seek_names):
        if type(seek_names) == str:
            return self.search(seek_names)
        nodes = []
        for name in seek_names:
            if type(name) == str:
                nodes.append(self.search(name))
            else:
                nodes.append(self.search_several(name))

        return nodes

    def generate_info(self, f):
        sz = 0
        for stage in self.stageslist:
            sz += stage.generate(f)

        for nul in range(align(sz, np.zeros(1), align_to=8)[0] - sz):
            f.write(c_char(0))

    def generate_data(self, f):
        write_data(f)

    def debug(self):
        for stage in self.head:
            stage.debug()

    def finalize(self):
        sizes = []
        pointers = []
        names = []
        for stage in self.head:
            t_res = stage.assign_remnant_buffers(self)
            sizes.extend(t_res[0])
            pointers.extend(t_res[1])
            names.extend(t_res[2])

        self.outputInfo = (sizes, pointers, names)
        for stage in self.head:
            stage.finalize()
            stage.set_blob_vars()

    def optimize(self):
        self.convert_network_input_to_yxz()
        for stage in self.head:
            stage.convert_inputs_outputs_to_yxz(True)
            stage.convert_taps_to_hwck(True)

        for idx, out_node in enumerate(self.outputInfo[0]):
            self.outputInfo[0][idx] = (
             out_node[2], out_node[1], out_node[0])

        self.outputNeedsTransforming = True

    def gather_metrics(self, timings):
        prev_len = 0
        for stage in self.stageslist:
            stage.calculate_metrics(timings[prev_len:])
            prev_len = prev_len + 1

    def convert_network_input_to_yxz(self, debug=False):
        if self.storageOrder.value == StorageOrder.orderYXZ.value:
            if debug:
                print('Already in this form')
        elif self.storageOrder.value == StorageOrder.orderZYX.value:
            if len(self.inputTensor.shape) == 4:
                self.inputTensor = np.reshape(self.inputTensor, (self.inputTensor.shape[1], self.inputTensor.shape[2], self.inputTensor.shape[3]))
                self.inputTensor = zyx_to_yxz(self.inputTensor, self.datatype.value).astype(dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            else:
                self.inputTensor = zyx_to_yxz(self.inputTensor, self.datatype.value).astype(dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
        elif self.storageOrder.value == StorageOrder.orderXYZ.value:
            if len(self.inputTensor.shape) == 4:
                self.inputTensor = np.reshape(self.inputTensor, (self.inputTensor.shape[1], self.inputTensor.shape[2], self.inputTensor.shape[3]))
                self.inputTensor = xyz_to_yxz(self.inputTensor, self.datatype.value).astype(dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            else:
                throw_error(ErrorTable.ConversionNotSupported, self.storageOrder.name)
        else:
            throw_error(ErrorTable.ConversionNotSupported, self.storageOrder.name)

    def verify(self):
        for stage in self.head:
            stage.verify()

    def newick(self):
        nw = '( '
        for idx, t in enumerate(self.head):
            nw += t.newick(head=True)
            if idx + 1 != len(self.head):
                nw += ','

        nw += ' );'
        return nw
# okay decompiling Network.pyc
