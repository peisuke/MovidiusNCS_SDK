# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Models/NetworkStage.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 35041 bytes
import numpy as np
from ctypes import *
from Controllers.MiscIO import *
from Controllers.DataTransforms import *
from Controllers.EnumController import *
from Models.EnumDeclarations import *
from Views.Graphs import *
from linecache import getline

class NetworkStage:

    def __init__(self, name, top, s_order, pad_x, pad_y, pad_type, dtype, precision, op_type, op_x, op_y, sx, sy, x, y, c, fw, fh, k, taps, taps_order, bias, pre_op_type, post_op_type, post_1, post_sx, post_sy, slicing=None, myriad_config=None, args=None, opParams=None, new_x=0, new_y=0, new_c=0):
        self.changeName(name)
        if op_type == StageType.convolution and op_x == 1 and op_y == 1 and x == 1 and y == 1:
            op_type = StageType.fully_connected_layer
        self.top = top
        self.tail = []
        self.op = op_type
        self.radixX = op_x
        self.radixY = op_y
        self.padX = pad_x
        self.padY = pad_y
        self.alias = [name]
        if self.radixX == -1 and self.radixY == -1:
            self.radixX = x
            self.radixY = y
        self.strideX = sx
        self.strideY = sy
        self.optMask = readOptimisationMask(name, self, myriad_config, args)
        self.inputStrideX = 2 * c
        self.inputStrideY = 2 * c * x
        self.inputStrideZ = 2
        self.inputOffset = 0
        if slicing:
            if top is None:
                for slice in slicing:
                    if slice[0] == None:
                        c = slice[2] - slice[1]
                        self.inputOffset = slice[1] * 2
                        break

            else:
                for input in top:
                    for slice in slicing:
                        if slice[0] == input:
                            c = slice[2] - slice[1]
                            self.inputOffset = slice[1] * 2
                            break

            if op_type == StageType.eltwise_sum or op_type == StageType.eltwise_prod or op_type == StageType.eltwise_max:
                k = c
            self.inputDimX = x
            self.inputDimY = y
            self.inputDimZ = c
            self.tapDimX = fw * fh
            self.tapDimY = c
            self.tapDimZ = k
            self.outputDimZ = k
        if self.op in [StageType.fully_connected_layer]:
            self.inputDimX = 1
            self.inputDimY = 1
            self.inputDimZ = x * y * c
            self.tapDimX = 1
            self.tapDimY = x * y * c
            self.outputDimX = 1
            self.outputDimY = 1
        elif self.op in [StageType.convolution, StageType.max_pooling, StageType.average_pooling]:
            if pad_type == PadStyle.tfsame:
                self.outputDimX = math.ceil(x / self.strideX)
                self.outputDimY = math.ceil(y / self.strideY)
            elif pad_type == PadStyle.tfvalid:
                self.outputDimX = math.ceil((x - self.radixX + 1) / self.strideX)
                self.outputDimY = math.ceil((y - self.radixY + 1) / self.strideY)
            elif self.op == StageType.convolution:
                if self.radixX == 1 and self.radixY == 1 and self.padX == 1 and self.padY == 1:
                    throw_error(ErrorTable.StageDetailsNotSupported, 'Padding 1 not supported for 1x1 convolution in ' + name)
                self.outputDimX = (x + 2 * self.padX - self.radixX) // self.strideX + 1
                self.outputDimY = (y + 2 * self.padY - self.radixY) // self.strideY + 1
            else:
                self.outputDimX = math.ceil((x + 2 * self.padX - self.radixX) / self.strideX) + 1
                self.outputDimY = math.ceil((y + 2 * self.padY - self.radixY) / self.strideY) + 1
                self.outputDimX = min(self.outputDimX, math.ceil((x + self.padX) / self.strideX))
                self.outputDimY = min(self.outputDimY, math.ceil((y + self.padY) / self.strideY))
        elif self.op in [StageType.deconvolution]:
            if pad_type == PadStyle.tfsame:
                pad_X = math.floor(self.radixX / 2)
                pad_Y = math.floor(self.radixY / 2)
            elif pad_type == PadStyle.tfvalid:
                pad_X = self.radixX - 1
                pad_Y = self.radixY - 1
            else:
                if pad_type == PadStyle.caffe:
                    pad_X = self.padX
                    pad_Y = self.padY
                else:
                    pad_X = 0
                    pad_Y = 0
                self.outputDimX = self.strideX * (x - 1) + self.radixX - 2 * pad_X
                self.outputDimY = self.strideY * (y - 1) + self.radixY - 2 * pad_Y
        elif self.op == StageType.toplanemajor:
            self.outputDimX = 1
            self.outputDimY = 1
            self.outputDimZ = x * y * c
        elif self.op in [StageType.reshape]:
            self.outputDimX = new_x
            self.outputDimY = new_y
            self.outputDimZ = new_c
            if new_x == 0:
                self.outputDimX = x
            elif new_x > 0:
                self.outputDimX = new_x
            if new_y == 0:
                self.outputDimY = y
            elif new_y > 0:
                self.outputDimY = new_y
            if new_c == 0:
                self.outputDimZ = c
            elif new_c > 0:
                self.outputDimZ = new_c
            if new_x == -1:
                self.outputDimX = x * y * c // (self.outputDimY * self.outputDimZ)
            if new_y == -1:
                self.outputDimY = x * y * c // (self.outputDimX * self.outputDimZ)
            if new_c == -1:
                self.outputDimZ = x * y * c // (self.outputDimX * self.outputDimY)
        else:
            if self.op in [StageType.crop]:
                self.outputDimX = new_x
                self.outputDimY = new_y
                self.outputDimZ = new_c
            else:
                self.outputDimX = x
                self.outputDimY = y
            self.output = np.zeros((int(self.outputDimZ),
             int(self.outputDimY),
             int(self.outputDimX))).astype(enum_as_dtype(dtype))
            self.tapStrideX = 2 * self.tapDimZ
            self.tapStrideY = 2 * self.tapDimZ
            self.tapStrideZ = 2
            self.outputStrideX = 2 * self.outputDimZ
            self.outputStrideY = 2 * self.outputDimZ * self.outputDimX
            self.outputStrideZ = 2
            self.unprocessed_w = x
            self.unprocessed_h = y
            self.unprocessed_c = c
            self.unprocessed_k = k
            self.unprocessed_output = self.output
            self.datatype = dtype
            self.precision = precision
            self.data = np.zeros((int(self.inputDimZ),
             int(self.inputDimY),
             int(self.inputDimX))).astype(enum_as_dtype(dtype))
            self.taps = taps
            self.tapsPointer = 0
            self.tapsIndex = 0
            self.tapsOrder = taps_order
            self.bias = bias
            self.biasPointer = 0
            self.biasIndex = 0
            self.opParams = opParams
            self.opParamsPointer = 0
            self.opParamsIndex = 0
            self.concatResult = False
            self.storageOrder = s_order
            self.padStyle = pad_type
            self.dataPointer = self.inputOffset
            self.dataIndex = 0
            self.outputPointer = 0
            self.outputIndex = 0
            if pre_op_type:
                self.preOp = pre_op_type
            else:
                self.preOp = StageType.none
            if post_op_type and post_op_type != StageType.none:
                self.postOp = post_op_type
                if post_1:
                    self.post_param1 = post_1
                else:
                    self.post_param1 = int(0)
                self.post_strideX = post_sx
                self.post_strideY = post_sy
            else:
                if (self.op == StageType.convolution or self.op == StageType.fully_connected_layer or self.op == StageType.scale) and bias is-not None:
                    self.postOp = StageType.bias
                else:
                    self.postOp = StageType.none
                self.post_param1 = 0
                self.post_strideX = 0
                self.post_strideY = 0
            if self.op in [StageType.reshape]:
                self.outputDimX = new_x
                self.outputDimY = new_y
                self.outputDimZ = new_c
            self.flops = None
            self.ms = None
            self.BWs = None
            self.isoutput = False
            self.isconcat = False

    def addBias(self, bias):
        self.bias = bias
        if self.bias is-not None:
            self.postOp = StageType.bias

    def putBias(self):
        if self.bias is-not None:
            self.biasPointer, self.biasBufferIndex = get_buffer(self.bias.astype(np.float16), self.datatype)
            self.biasIndex = MemoryIndex.blob.value

    def putTaps(self):
        if self.taps is-not None:
            self.tapsPointer, self.tapsBufferIndex = get_buffer(self.taps.astype(np.float16), self.datatype)
            self.tapsIndex = MemoryIndex.blob.value

    def putOpParams(self):
        if self.opParams is-not None:
            self.opParamsPointer, self.opParamsBufferIndex = get_buffer(self.opParams, DataType.fp32)
            self.opParamsIndex = MemoryIndex.blob.value

    def changeName(self, new_name):
        self.unprocessed_name = new_name
        self.name = set_string_range(new_name, 100).encode('ascii')

    def close(self):
        self.outputPointer = 0
        self.outputIndex = MemoryIndex.output

    def attach(self, stage):
        self.tail.append(stage)
        if self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value:
            self.outputPointer, self.outputIndex = get_zero_buffer(self.output, self.datatype)
        stage.dataPointer = stage.inputOffset + self.outputPointer
        stage.dataIndex = self.outputIndex
        if stage.op != StageType.fully_connected_layer and not self.isconcat:
            stage.inputDimX, stage.inputDimY, stage.inputDimZ = self.outputDimX, self.outputDimY, self.outputDimZ
            stage.tapDimY = self.outputDimZ
        if stage.op in [StageType.max_pooling]:
            stage.output = np.zeros((stage.outputDimZ, stage.outputDimY, stage.outputDimX))

    def setoutput(self, outputStride, outputPointer=None, outputIndex=None):
        if self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value:
            self.output = np.zeros((int(outputStride / 2), int(self.outputDimY), int(self.outputDimX))).astype(enum_as_dtype(self.datatype))
            if outputPointer is-not None and outputIndex is-not None:
                self.outputPointer = outputPointer
                self.outputIndex = outputIndex
            else:
                self.outputPointer, self.outputIndex = get_zero_buffer(self.output, self.datatype)
            self.outputStrideX = outputStride
        self.isconcat = True
        return (
         self.outputPointer, self.outputIndex)

    def concat(stages, lastlayer=True):
        z = sum([int(stage.unprocessed_k) for stage in stages])
        x = int(stages[0].outputDimX)
        y = int(stages[0].outputDimY)
        concat_size = (
         y, x, z)
        dtype = stages[0].datatype
        if lastlayer:
            for stage in stages:
                stage.isoutput = True

            buffer = 0
            buffer_index = MemoryIndex.output.value
        else:
            if stages[0].outputPointer == 0 and stages[0].outputIndex == MemoryIndex.none.value:
                buffer, buffer_index = get_zero_buffer(np.zeros(concat_size).astype(enum_as_dtype(dtype)), dtype)
            else:
                buffer = stages[0].outputPointer
                buffer_index = stages[0].outputIndex
            concat_offset = 0
            for s_num, stage in enumerate(stages):
                offset_pointer = buffer
                if stage.outputPointer == 0:
                    stage.outputPointer = offset_pointer + concat_offset * 2
                stage.outputIndex = buffer_index
                stage.concatResult = True
                concat_offset += int(stage.outputDimZ)
                stage.outputStrideX = z * 2
                stage.outputStrideY = z * 2 * stage.outputDimX
                stage.tapStrideY = stage.outputDimZ * 2

    def attach_eltwise(self, parents):
        if hasattr(parents[0], '__iter__'):
            NetworkStage.concat(parents[0], False)
            parents[0] = parents[0][0]
        if parents[1] == 0:
            parents[0].outputPointer, parents[0].outputIndex = get_zero_buffer(parents[0].output, self.datatype)
            self.dataPointer = self.inputOffset + parents[0].outputPointer
            self.dataIndex = parents[0].outputIndex
            self.tapsPointer = 0
            self.tapsIndex = MemoryIndex.input.value
            parents[0].tail.append(self)
        else:
            if hasattr(parents[1], '__iter__'):
                NetworkStage.concat(parents[1], False)
                parents[1] = parents[1][0]
            if parents[0].outputIndex == 0:
                parents[0].outputPointer, parents[0].outputIndex = get_zero_buffer(parents[0].output, self.datatype)
            if parents[1].outputIndex == 0:
                parents[1].outputPointer, parents[1].outputIndex = get_zero_buffer(parents[1].output, self.datatype)
            self.dataPointer = self.inputOffset + parents[0].outputPointer
            self.dataIndex = parents[0].outputIndex
            self.tapsPointer = parents[1].outputPointer
            self.tapsIndex = parents[1].outputIndex
            parents[1].tail.append(self)

    def attach_several(self, parents):
        if not hasattr(parents, '__iter__'):
            parents.attach(self)
            return
        NetworkStage.concat(parents, False)
        z = sum([int(p.unprocessed_k) for p in parents])
        parents[len(parents) - 1].tail.append(self)
        self.inputDimZ = z
        self.inputStrideX = z * 2
        self.dataPointer = self.inputOffset + parents[0].outputPointer
        self.dataIndex = parents[0].outputIndex
        self.tapDimY = z
        if self.op in [StageType.max_pooling]:
            self.outputDimZ = self.inputDimZ
            self.outputStrideX = self.inputStrideX

    def search(self, seek_name):
        if self.name == seek_name or self.unprocessed_name == seek_name or seek_name in self.alias:
            return self
        for t in self.tail:
            if t.name == seek_name or t.unprocessed_name == seek_name or seek_name in t.alias:
                return t
            recursive_result = t.search(seek_name)
            if recursive_result != 0 and recursive_result.name == seek_name or recursive_result != 0 and recursive_result.unprocessed_name == seek_name or recursive_result != 0 and seek_name in recursive_result.alias:
                return recursive_result

        return 0

    def set_blob_vars(self):
        self.write_items = [
         self.name,
         c_char(self.op.value),
         c_uint32(self.optMask),
         c_int8(self.radixX),
         c_int8(self.radixY),
         c_uint8(self.strideX),
         c_uint8(self.strideY),
         c_uint8(self.padX),
         c_uint8(self.padY),
         c_uint8(self.padStyle.value),
         c_uint32(self.inputDimX),
         c_uint32(self.inputDimY),
         c_uint32(self.inputDimZ),
         c_uint32(self.tapDimX),
         c_uint32(self.tapDimY),
         c_uint32(self.tapDimZ),
         c_uint32(self.outputDimX),
         c_uint32(self.outputDimY),
         c_uint32(self.outputDimZ),
         c_uint32(self.inputStrideX),
         c_uint32(self.inputStrideY),
         c_uint32(self.inputStrideZ),
         c_uint32(self.tapStrideX),
         c_uint32(self.tapStrideY),
         c_uint32(self.tapStrideZ),
         c_uint32(self.outputStrideX),
         c_uint32(self.outputStrideY),
         c_uint32(self.outputStrideZ),
         c_uint8(self.datatype.value),
         c_uint8(self.precision.value),
         c_uint8(self.storageOrder.value),
         c_uint32(self.dataPointer),
         c_uint16(self.dataIndex),
         c_uint32(self.tapsPointer),
         c_uint16(self.tapsIndex),
         c_uint32(self.biasPointer),
         c_uint16(self.biasIndex),
         c_uint32(self.opParamsPointer),
         c_uint16(self.opParamsIndex),
         c_uint32(self.outputPointer),
         c_uint16(self.outputIndex),
         c_uint8(self.preOp.value),
         c_uint8(self.postOp.value),
         c_float(self.post_param1) if type(self.post_param1) == float else c_int32(self.post_param1),
         c_ushort(self.post_strideX),
         c_ushort(self.post_strideY)]
        for t in self.tail:
            t.set_blob_vars()

    def generate(self, f):
        sz = 0
        for item in self.write_items:
            f.write(item)
            sz += byte_size(item)

        return sz

    def binary_size(self):
        file_sizes = [byte_size(t) for t in self.write_items if isinstance(t, ctypes._SimpleCData) or isinstance(t, bytes)]
        return sum(file_sizes)

    def debug(self, to_file=False, f=None):
        if to_file:
            pass
        for t in self.tail:
            t.debug(to_file, f)

    def finalize(self):
        self.putTaps()
        self.putBias()
        self.putOpParams()
        for t in self.tail:
            t.finalize()

    def assign_remnant_buffers(self, net):
        sizes = []
        offsets = []
        names = []
        if self.top is-not None and isinstance(self.top[0], str):
            parent = net.search(self.top[0])
            self.inputStrideX = parent.outputStrideX
        if self.isoutput:
            sizes.append(self.output.shape)
            offsets.append(self.outputPointer)
            names.append(self.name)
        else:
            if self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value and (self.top is None or len(self.top) <= 1 or get_class_of_op(self.op) != 'Pooling'):
                self.outputIndex = MemoryIndex.output.value
                sizes.append(self.output.shape)
                offsets.append(self.outputPointer)
                names.append(self.name)
                self.isoutput = True
            elif self.outputPointer == 0 and self.outputIndex == MemoryIndex.none.value and len(self.top) > 1 and get_class_of_op(self.op) == 'Pooling':
                node = net.head[0].search(self.top[0])
                self.output = np.zeros((node.outputDimZ, node.outputDimY, node.outputDimX)).astype(np.float16)
                self.outputIndex = MemoryIndex.output.value
                sizes.append(self.output.shape)
                offsets.append(self.outputPointer)
                names.append(self.name)
                self.isoutput = True
            for t in self.tail:
                t_res = t.assign_remnant_buffers(net)
                sizes.extend(t_res[0])
                offsets.extend(t_res[1])
                names.extend(t_res[2])

        return (
         sizes, offsets, names)

    def convert_inputs_outputs_to_yxz(self, recurse, debug=False):
        if self.storageOrder == StorageOrder.orderYXZ:
            if debug:
                print('Already in this form')
        elif self.storageOrder == StorageOrder.orderZYX:
            if len(self.data.shape) == 4:
                self.data = np.reshape(self.data, (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
                self.data = zyx_to_yxz(self.data, self.datatype).astype(dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
            elif not self.concatResult:
                self.output = zyx_to_yxz(self.output, self.datatype).astype(dtype=np.float16)
                self.storageOrder = StorageOrder.orderYXZ
        else:
            if self.storageOrder == StorageOrder.orderXYZ:
                if len(self.data.shape) == 4:
                    self.data = np.reshape(self.data, (self.data.shape[1], self.data.shape[2], self.data.shape[3]))
                    self.data = xyz_to_yxz(self.data, self.datatype).astype(dtype=np.float16)
                    self.storageOrder = StorageOrder.orderYXZ
                else:
                    throw_error(ErrorTable.ConversionNotSupported, self.storageOrder.name)
            else:
                throw_error(ErrorTable.ConversionNotSupported, self.storageOrder.name)
            if recurse:
                for node in self.tail:
                    node.convert_inputs_outputs_to_yxz(recurse)

    def convert_taps_to_hwck(self, recurse):
        if self.tapsOrder == TapsOrder.orderHWCK:
            return
        if get_class_of_op(self.op) in ('Convolution', 'FCL', 'Deconvolution'):
            if self.op in [StageType.fully_connected_layer]:
                if self.unprocessed_h > 1 or self.unprocessed_w > 1:
                    self.taps = self.taps.reshape(self.unprocessed_k, self.unprocessed_c, self.unprocessed_h, self.unprocessed_w)
                else:
                    self.taps = self.taps.reshape(self.taps.shape[0], self.taps.shape[1], 1, 1)
                self.taps = kchw_to_hwck(self.taps)
                replace_buffer(self.taps, self.tapsBufferIndex, self.datatype)
            else:
                if self.taps is None or get_class_of_op(self.op) == 'FCL' or self.op == StageType.scale:
                    pass
                else:
                    throw_error(ErrorTable.ConversionNotSupported, self.op.name)
                self.storageOrder = StorageOrder.orderYXZ.value
                if recurse:
                    for node in self.tail:
                        node.convert_taps_to_hwck(recurse)

    def getBWs(self):
        in_dim = self.data.flatten().shape[0]
        if self.taps is-not None:
            tap_dim = self.taps.flatten().shape[0]
        else:
            tap_dim = 0
        out_dim = self.output.shape[0]
        KB = 1024
        MB = KB * KB
        MS = self.ms
        S = MS / 1000
        if self.op == StageType.convolution:
            arrays = in_dim * self.radixX * self.radixY
            arrays += tap_dim
            arrays += out_dim * self.radixX * self.radixY
        else:
            arrays = in_dim + tap_dim + out_dim
        self.BWs = arrays * 2 / MB / S
        return self.BWs

    def getBW(self):
        in_dim = self.data.flatten().shape[0]
        if self.taps is-not None:
            tap_dim = self.taps.flatten().shape[0]
        else:
            tap_dim = 0
        out_dim = self.output.shape[0]
        if self.op == StageType.convolution:
            arrays = in_dim * self.radixX * self.radixY
            arrays += tap_dim
            arrays += out_dim * self.radixX * self.radixY
        else:
            arrays = in_dim + tap_dim + out_dim
        return arrays * 2

    def minmax(self, attr, min, max):
        if min > getattr(self, attr):
            min = getattr(self, attr)
        if max < getattr(self, attr):
            max = getattr(self, attr)
        for t in self.tail:
            min, max = t.minmax(attr, min, max)

        return (
         min, max)

    def calculate_metrics(self, timings):
        self.flops = self.getFlops()
        self.ms = timings[0]
        self.BWs = self.getBWs()

    def getFlops(self):
        flops = 0
        if self.op == StageType.convolution:
            flops = self.unprocessed_k * self.outputDimX * self.outputDimY * self.inputDimZ * self.radixX * self.radixY * 2
        elif self.op == StageType.max_pooling:
            flops = self.unprocessed_k * self.outputDimX * self.outputDimY * self.radixX * self.radixY
        elif self.op == StageType.average_pooling:
            flops = self.unprocessed_k * self.outputDimX * self.outputDimY * self.radixX * self.radixY * 2
        elif self.op == StageType.fully_connected_layer:
            in_dim = self.data.flatten().shape[0]
            out_channels = self.output.shape[0]
            flops = in_dim * out_channels * 2
        elif self.op == StageType.soft_max:
            in_dim = self.data.flatten().shape[0]
            flops = in_dim * 3
        return flops / 1000000

    def summaryStats(self):
        totalTime = self.ms
        totalBW = self.getBW()
        for t in self.tail:
            a, b = t.summaryStats()
            totalTime += a
            totalBW += b

        return (
         totalTime, totalBW)

    def newick(self, head=False):
        nw = str(self.unprocessed_name) + ':' + str(len(self.tail))
        if len(self.tail) != 0:
            nw += ',('
            for idx, t in enumerate(self.tail):
                nw += t.newick()
                if idx + 1 != len(self.tail):
                    nw += ','

            nw += ')'
        return nw

    def graphviz(self, dot, ms_min, ms_max, bws_min, bws_max, flop_min, flop_max):
        table = '<\n<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">\n<TR>\n    <TD  BGCOLOR = "#A3A3A3" COLSPAN="3">{0}</TD>\n</TR>\n<TR>\n    <TD  BGCOLOR = "#DDDDDD" COLSPAN="3">{7}</TD>\n</TR>\n<TR>\n    <TD BGCOLOR = "{1}"> {2} <br/> (MFLOPs) </TD>\n    <TD BGCOLOR = "{3}"> {4} <br/> (MB/s) </TD>\n    <TD BGCOLOR = "{5}"> {6} <br/> (ms)</TD>\n</TR>\n</TABLE>>\n'.format(self.unprocessed_name, get_normalized_color('#B1F1EF', '#2ED1C6', flop_min, flop_max, self.flops), self.flops, get_normalized_color('#FFE5FC', '#B2189E', bws_min, bws_max, format(self.BWs, '.2f')), format(self.BWs, '.2f'), get_normalized_color('#FFFFCC', '#FFFF00', ms_min, ms_max, format(self.ms, '.2f')), format(self.ms, '.2f'), str(self.unprocessed_output.shape))
        dot.node(self.unprocessed_name, table, shape='plaintext')
        if self.top is-not None:
            for t in self.top:
                if not isinstance(t, str):
                    for tt in t:
                        dot.edge(tt, self.unprocessed_name)

                else:
                    dot.edge(t, self.unprocessed_name)

        else:
            dot.edge('Input', self.unprocessed_name)
        last_nodes = []
        for t in self.tail:
            dot, last = t.graphviz(dot, ms_min, ms_max, bws_min, bws_max, flop_min, flop_max)
            last_nodes.extend(last)

        if len(self.tail) == 0:
            last_nodes = [
             self.unprocessed_name]
        return (
         dot, last_nodes)
# okay decompiling NetworkStage.pyc
