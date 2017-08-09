# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/CaffeParser.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 41313 bytes
import numpy as np
from Models.NetworkStage import *
from Models.Network import *
from Models.CaffeEnumDeclarations import *
from Controllers.MiscIO import *
from google.protobuf import message
from google.protobuf import text_format
import os
import ctypes
try:
    os.environ['GLOG_minloglevel'] = '2'
    import caffe
    from caffe.proto import caffe_pb2
except ImportError:
    caffe = None
    print('Caffe Import Error')
    quit()

concat_tracker = []
slice_tracker = []
data_type = np.float16

def isConvolution(layer):
    return layer in ['Convolution', CaffeStage.CONVOLUTION.value]


def isReLU(layer):
    return layer in ['ReLU', CaffeStage.RELU.value]


def isSigmoid(layer):
    return layer in ('Sigmoid', )


def isTanH(layer):
    return layer in ('TanH', )


def isPReLU(layer):
    return layer in ('PReLU', )


def isPooling(layer):
    return layer in ['Pooling', CaffeStage.POOLING.value]


def isSoftMax(layer):
    return layer in ['Softmax', CaffeStage.SOFTMAX.value]


def isFCL(layer):
    return layer in ['InnerProduct', CaffeStage.INNER_PRODUCT.value]


def isLRN(layer):
    return layer in ['LRN', CaffeStage.LRN.value]


def isInnerLRN(layer):
    return layer.type in ['LRN', CaffeStage.LRN.value] and layer.lrn_param.norm_region == 1


def isConcat(layer):
    return layer in ['Concat', CaffeStage.CONCAT.value]


def isDropout(layer):
    return layer in ['Dropout', CaffeStage.DROPOUT.value]


def isEltwise(layer):
    return layer in ['Eltwise', 'Bias', CaffeStage.ELTWISE.value]


def isSlice(layer):
    return layer in ['Slice', CaffeStage.SLICE.value]


def isBatchNorm(layer):
    return layer in ('BatchNorm', )


def isScale(layer):
    return layer in ('Scale', )


def isDeconvolution(layer):
    return layer in ['Deconvolution', CaffeStage.DECONVOLUTION.value]


def isPower(layer):
    return layer in ['Power', CaffeStage.POWER.value]


def isReshape(layer):
    return layer in ['Reshape', CaffeStage.RESHAPE.value]


def isELU(layer):
    return layer in ('ELU', )


def isFlatten(layer):
    return layer in ['Flatten', CaffeStage.FLATTEN.value]


def isCrop(layer):
    return layer in ('Crop', )


def caffe_search_pre_op(msg, name):
    pass


def get_caffe_kernel_size(layer):
    if isConvolution(layer.type):
        if layer.convolution_param.kernel_w:
            return (layer.convolution_param.kernel_h, layer.convolution_param.kernel_w)
        return (
         layer.convolution_param.kernel_size[0], layer.convolution_param.kernel_size[0])
    if isPooling(layer.type):
        if layer.pooling_param.kernel_w:
            return (layer.pooling_param.kernel_h, layer.pooling_param.kernel_w)
        return (
         layer.pooling_param.kernel_size, layer.pooling_param.kernel_size)
    if isDeconvolution(layer.type):
        if layer.convolution_param.kernel_w:
            return (layer.convolution_param.kernel_h, layer.convolution_param.kernel_w)
        return (
         layer.convolution_param.kernel_size[0], layer.convolution_param.kernel_size[0])
    return (0, 0)


def get_caffe_group(layer):
    if isConvolution(layer.type):
        return layer.convolution_param.group
    if isDeconvolution(layer.type):
        return layer.convolution_param.group
    return 1


def get_caffe_output_channels(layer, prev_output_shape, top, network):
    if isConvolution(layer.type):
        return layer.convolution_param.num_output
    if isFCL(layer.type):
        return layer.inner_product_param.num_output
    if isReshape(layer.type):
        return layer.reshape_param.shape[0]
    if isPooling(layer.type) or isSoftMax(layer.type) or isLRN(layer.type) or isSigmoid(layer.type) or isTanH(layer.type) or isPower(layer.type):
        if top is-not None and len(top) > 1:
            sum_of_k_from_parents = 0
            for parent in top:
                prev_node = network.search(parent)
                sum_of_k_from_parents += prev_node.output.shape[0]

            return sum_of_k_from_parents
        return prev_output_shape[0]
    if isEltwise(layer.type) or isBatchNorm(layer.type) or isScale(layer.type) or isPReLU(layer.type):
        return prev_output_shape[0]
    if isDeconvolution(layer.type):
        return layer.convolution_param.num_output
    if isFlatten(layer.type):
        return prev_output_shape[0] * prev_output_shape[1] * prev_output_shape[2]
    return 1


def get_caffe_op_radix(layer):
    if isConvolution(layer.type):
        if layer.convolution_param.kernel_w:
            return (layer.convolution_param.kernel_h, layer.convolution_param.kernel_w)
        return (
         layer.convolution_param.kernel_size[0], layer.convolution_param.kernel_size[0])
    else:
        if isLRN(layer.type):
            return (layer.lrn_param.local_size, 0)
        if isPooling(layer.type):
            if layer.pooling_param.kernel_size == 0 and layer.pooling_param.global_pooling:
                return (-1, -1)
            if layer.pooling_param.kernel_w:
                return (layer.pooling_param.kernel_h, layer.pooling_param.kernel_w)
            return (
             layer.pooling_param.kernel_size, layer.pooling_param.kernel_size)
        else:
            if isPooling(layer.type):
                return (
                 layer.pooling_param.kernel_size, layer.pooling_param.kernel_size)
            if isDeconvolution(layer.type):
                if layer.convolution_param.kernel_w:
                    return (layer.convolution_param.kernel_h, layer.convolution_param.kernel_w)
                return (
                 layer.convolution_param.kernel_size[0], layer.convolution_param.kernel_size[0])
            else:
                return (1, 1)


def get_caffe_op_type(layer):
    if isConvolution(layer.type):
        return StageType.convolution
    if isFCL(layer.type):
        return StageType.fully_connected_layer
    if isSoftMax(layer.type):
        return StageType.soft_max
    if isPooling(layer.type):
        pooling_type = layer.pooling_param.pool
        if pooling_type == 0:
            return StageType.max_pooling
        if pooling_type == 1:
            return StageType.average_pooling
        if pooling_type == 2:
            throw_error(ErrorTable.StageTypeNotSupported, 'Stochastic Pooling')
            return StageType.stochastic_pooling
    if isLRN(layer.type):
        return StageType.LRN
    if isEltwise(layer.type):
        if layer.type == 'Bias':
            return StageType.eltwise_sum
        if layer.eltwise_param.operation == 0:
            return StageType.eltwise_prod
        if layer.eltwise_param.operation == 2:
            return StageType.eltwise_max
        return StageType.eltwise_sum
    if isBatchNorm(layer.type) or isScale(layer.type):
        return StageType.scale
    if isPReLU(layer.type):
        return StageType.prelu
    if isSigmoid(layer.type):
        return StageType.sigmoid
    if isTanH(layer.type):
        return StageType.tanh
    if isDeconvolution(layer.type):
        return StageType.deconvolution
    if isReshape(layer.type):
        return StageType.reshape
    if isFlatten(layer.type):
        return StageType.toplanemajor
    if isPower(layer.type):
        return StageType.power
    if isCrop(layer.type):
        return StageType.crop
    throw_error(ErrorTable.StageTypeNotSupported, layer.type)


def get_caffe_op_padding(layer):
    if isConvolution(layer.type) or isDeconvolution(layer.type):
        if layer.convolution_param.pad_w:
            return (layer.convolution_param.pad_h, layer.convolution_param.pad_w)
        if layer.convolution_param.pad:
            return (layer.convolution_param.pad[0], layer.convolution_param.pad[0])
    if isPooling(layer.type):
        if layer.pooling_param.pad_w:
            return (layer.pooling_param.pad_h, layer.pooling_param.pad_w)
        if layer.pooling_param.pad:
            return (layer.pooling_param.pad, layer.pooling_param.pad)
    return (0, 0)


def get_caffe_op_stride(layer):
    if isConvolution(layer.type):
        if layer.convolution_param.stride_w:
            return (layer.convolution_param.stride_h, layer.convolution_param.stride_w)
        if layer.convolution_param.stride:
            return (layer.convolution_param.stride[0], layer.convolution_param.stride[0])
    if isPooling(layer.type):
        if layer.pooling_param.stride_w:
            return (layer.pooling_param.stride_h, layer.pooling_param.stride_w)
        if layer.pooling_param.stride:
            return (layer.pooling_param.stride, layer.pooling_param.stride)
    if isDeconvolution(layer.type):
        if layer.convolution_param.stride_w:
            return (layer.convolution_param.stride_h, layer.convolution_param.stride_w)
        if layer.convolution_param.stride:
            return (layer.convolution_param.stride[0], layer.convolution_param.stride[0])
    return (1, 1)


def get_caffe_params(layer, blobs):
    global data_type
    if isLRN(layer.type):
        return (
         None, np.array([layer.lrn_param.k, layer.lrn_param.alpha, layer.lrn_param.beta, 0], dtype=data_type))
    if isConvolution(layer.type) or isDeconvolution(layer.type):
        if layer.convolution_param.bias_term:
            return (blobs[layer.name][0].data.astype(dtype=data_type), blobs[layer.name][1].data.astype(dtype=data_type))
        return (
         blobs[layer.name][0].data.astype(dtype=data_type), None)
    elif isFCL(layer.type):
        if layer.inner_product_param.bias_term:
            return (blobs[layer.name][0].data.astype(dtype=data_type), blobs[layer.name][1].data.astype(dtype=data_type))
        return (
         blobs[layer.name][0].data.astype(dtype=data_type), None)
    elif isBatchNorm(layer.type):
        if blobs[layer.name][2].data[0] == 0:
            mean = np.zeros(blobs[layer.name][0].data.shape)
            var = np.zeros(blobs[layer.name][1].data.shape) + layer.batch_norm_param.eps
        else:
            mean = blobs[layer.name][0].data * (1 / blobs[layer.name][2].data[0])
            var = blobs[layer.name][1].data * (1 / blobs[layer.name][2].data[0]) + layer.batch_norm_param.eps
        mult = np.reciprocal(np.sqrt(var))
        bias = -mean * mult
        return (
         mult.astype(dtype=data_type), bias.astype(dtype=data_type))
    if isScale(layer.type):
        if layer.scale_param.bias_term:
            return (blobs[layer.name][0].data.astype(dtype=data_type), blobs[layer.name][1].data.astype(dtype=data_type))
        return (
         blobs[layer.name][0].data.astype(dtype=data_type), None)
    else:
        if isPReLU(layer.type):
            return (None, blobs[layer.name][0].data)
        return (None, None)


def caffe_apply_minor_op(network, layer, top):
    global slice_tracker
    global concat_tracker
    if isReLU(layer.type) or isELU(layer.type):
        if top is-not None and type(top[0]) != str:
            top = top[0]
        for prevlayer in top:
            if isReLU(layer.type):
                applicable_node = network.search(prevlayer)
                applicable_node.postOp = StageType.relu
                applicable_node.post_param1 = layer.relu_param.negative_slope
            if isELU(layer.type):
                applicable_node = network.search(prevlayer)
                applicable_node.postOp = StageType.elu
                applicable_node.post_param1 = layer.elu_param.alpha
            if len(top) == 1:
                applicable_node.unprocessed_name = layer.top[0]
                applicable_node.name = set_string_range(layer.top[0], 100).encode('ascii')

    elif isConcat(layer.type):
        concat_tracker.append((layer.top[0], layer.bottom))
    elif isSlice(layer.type):
        slice_tracker.append((layer.top, layer.bottom, layer.slice_param.slice_point))
    else:
        throw_error(ErrorTable.StageTypeNotSupported, layer.type)


def parse_caffe(arguments, myriad_conf, debug=False, file_gen=False):
    path = arguments.net_description
    weights = arguments.net_weights
    input_image = arguments.image
    outputNodeName = arguments.output_node_name
    inputNodeName = arguments.input_node_name
    raw_scale = arguments.raw_scale
    filename = arguments.outputs_name
    mean = arguments.mean
    channel_swap = arguments.channel_swap
    caffe.set_mode_cpu()
    description = path
    if weights is None:
        open('zero_weights.caffemodel', 'wb').close()
        weights = 'zero_weights.caffemodel'
        print('\x1b[91m****** WARNING: using empty weights ******\x1b[0m')
    if not os.path.isfile(weights):
        throw_error(ErrorTable.ArgumentErrorWeights)
    try:
        net = caffe.Net(description, weights, caffe.TEST)
    except MemoryError:
        throw_error(ErrorTable.CaffeMemoryError)

    try:
        f = open(description)
        file_contents = f.read()
        f.close()
    except:
        throw_error(ErrorTable.ArgumentErrorDescription)

    msg = caffe_pb2.NetParameter()
    text_format.Merge(str(file_contents), msg)
    layers = msg.layer
    if len(layers) == 0:
        layers = msg.layers
    startNodeName = inputNodeName
    if layers[0].type == 'Input':
        try:
            input_shape = layers[0].input_param.shape[0].dim
            input_bottom = layers[0].top[0]
        except:
            throw_error(ErrorTable.InputSyntaxNotSupported)

    else:
        try:
            input_shape = msg.input_shape[0].dim
            input_bottom = layers[0].bottom[0]
        except:
            throw_error(ErrorTable.InputSyntaxNotSupported)

        if input_shape[0] != 1:
            throw_error(ErrorTable.AttemptedBatchMode)
        if inputNodeName:
            input_bottom = net.bottom_names[inputNodeName][0]
            for i, layername in enumerate(net._layer_names):
                if input_bottom in net.top_names[layername] and net.layers[i].type == 'Split':
                    input_bottom = net.bottom_names[layername][0]
                    startNodeName = layername

            input_shape = [
             net.blobs[input_bottom].shape[0], net.blobs[input_bottom].shape[1],
             net.blobs[input_bottom].shape[2], net.blobs[input_bottom].shape[3]]
        if input_image is None or input_image == 'Debug':
            try:
                input_data = np.ones(input_shape).astype(data_type)
            except:
                throw_error(ErrorTable.InputSyntaxNotSupported)

            np.random.seed(1)
            input_data = np.random.uniform(-1, 1, input_shape).astype(dtype=data_type)
        else:
            input_data = parse_img(input_image, input_shape, raw_scale=raw_scale, mean=mean, channel_swap=channel_swap)
        if outputNodeName == None:
            outputNodeName = net.outputs[0]
        network = Network(msg.name, input_data)
        prev_output_shape = [
         input_data[0].shape]
        last_layer = None
        first_layer = None
        nlayers = len(layers)
        for idx, layer in enumerate(layers):
            if debug:
                print('------------')
                print(layer)
            if layer.type == 'Input':
                continue
                if inputNodeName:
                    if inputNodeName == layer.name:
                        first_layer = layer
                    else:
                        if first_layer == None:
                            pass
                        continue
                    if isEltwise(layer.type) and len(layer.bottom) == 2 and layer.bottom[0] == input_bottom:
                        tmp = layer.bottom[0]
                        layer.bottom[0] = layer.bottom[1]
                        layer.bottom[1] = tmp
                    curslicing = []
                    if layer.bottom[0] == input_bottom:
                        top = None
                        prev_output_shape = [input_data[0].shape]
                    else:
                        for slice in slice_tracker:
                            for i in range(len(slice[0])):
                                for j in range(len(layer.bottom)):
                                    if layer.bottom[j] == slice[0][i]:
                                        inputplanes = net.blobs[slice[1][0]].shape[1]
                                        start = 0 if i == 0 else slice[2][i - 1]
                                        end = slice[2][i] if i < len(slice[2]) else inputplanes
                                        if slice[1][0] == input_bottom:
                                            curslicing.append([None, start, end])
                                        else:
                                            curslicing.append([slice[1][0], start, end])
                                        layer.bottom[j] = slice[1][0]
                                        break

                        top = []
                        for obj in layer.bottom:
                            top.append(obj)

                        if len(concat_tracker) != 0:
                            for concat in concat_tracker:
                                for i in range(len(top)):
                                    if concat[0] == top[i]:
                                        top[i] = concat[1]

                        if top[0] == input_bottom:
                            top = None
                            prev_output_shape = [input_data[0].shape]
                        else:
                            prev_output_shape = []
                            nodes = network.search_several(top)
                            if len(nodes) == 0:
                                throw_error(ErrorTable.GraphConstructionFailure, top)
                            for i, node in enumerate(nodes):
                                if node == 0:
                                    throw_error(ErrorTable.GraphConstructionFailure, top[i])
                                if hasattr(node, '__iter__'):
                                    shape = node[0].output.shape
                                    for i in range(len(node)):
                                        if i > 0:
                                            if shape[1] != node[i].output.shape[1] or shape[2] != node[i].output.shape[2]:
                                                throw_error(ErrorTable.StageDetailsNotSupported, layer.name)
                                            shape = (
                                             shape[0] + node[i].output.shape[0], shape[1], shape[2])

                                    prev_output_shape.append(shape)
                                else:
                                    prev_output_shape.append(node.output.shape)

                        inshape = prev_output_shape[0]
                        if isEltwise(layer.type) or isConcat(layer.type):
                            for i in range(len(prev_output_shape)):
                                if i > 0:
                                    if inshape[1] != prev_output_shape[i][1] or inshape[2] != prev_output_shape[i][2]:
                                        throw_error(ErrorTable.StageDetailsNotSupported, layer.name)
                                    inshape = (
                                     max(inshape[0], prev_output_shape[i][0]), inshape[1], inshape[2])

            if isDropout(layer.type):
                continue
                if isBatchNorm(layer.type) or isScale(layer.type):
                    node = network.search(layer.bottom[0])
                    if node != 0 and node.op == StageType.convolution:
                        w, b = get_caffe_params(layer, net.params)
                        node.taps = (node.taps.T * w).T
                        if node.bias is-not None:
                            if b is-not None:
                                node.bias = node.bias * w + b
                            else:
                                node.bias = node.bias * w
                        elif b is-not None:
                            node.addBias(np.array(b).astype(np.float16))
                        node.name = layer.name
                        node.changeName(node.name)
                        node.alias.append(node.unprocessed_name)
                        if layer.name == outputNodeName:
                            break
                    if isInnerLRN(layer):
                        network.attach(NetworkStage(layer.name + '_Square', top, StorageOrder.orderZYX, 0, 0, PadStyle.caffe, DataType.fp16, DataType.fp16, StageType.square, 1, 1, 1, 1, inshape[2], inshape[1], inshape[0], 0, 0, inshape[0], None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, myriad_conf, args=arguments))
                        network.attach(NetworkStage(layer.name + '_AvgPool', [layer.name + '_Square'], StorageOrder.orderZYX, (layer.lrn_param.local_size - 1) // 2, (layer.lrn_param.local_size - 1) // 2, PadStyle.caffe, DataType.fp16, DataType.fp16, StageType.average_pooling, layer.lrn_param.local_size, layer.lrn_param.local_size, 1, 1, inshape[2], inshape[1], inshape[0], layer.lrn_param.local_size, layer.lrn_param.local_size, inshape[0], None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, myriad_conf, args=arguments))
                        network.attach(NetworkStage(layer.name + '_InnerLRN', [layer.name + '_AvgPool'], StorageOrder.orderZYX, 0, 0, PadStyle.caffe, DataType.fp16, DataType.fp16, StageType.innerlrn, 1, 1, 1, 1, inshape[2], inshape[1], inshape[0], 1, 1, inshape[0], None, TapsOrder.orderKCHW, np.array([layer.lrn_param.k, layer.lrn_param.alpha, layer.lrn_param.beta, 0], dtype=data_type), None, StageType.none, None, 0, 0, myriad_conf, args=arguments))
                        if top is None:
                            top = [
                             layer.name + '_InnerLRN', None]
                        else:
                            top = [
                             top[0], layer.name + '_InnerLRN']
                        network.attach(NetworkStage(layer.name, top, StorageOrder.orderZYX, 0, 0, PadStyle.caffe, DataType.fp16, DataType.fp16, StageType.eltwise_prod, 1, 1, 1, 1, inshape[2], inshape[1], inshape[0], 1, 1, inshape[0], None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, myriad_conf, args=arguments))
                        last_layer = layer
                        if layer.name == outputNodeName:
                            break
                        if isReshape(layer.type):
                            new_shape = layer.reshape_param.shape
                            network.attach(NetworkStage(layer.name, top, StorageOrder.orderZYX, 0, 0, PadStyle.caffe, DataType.fp16, DataType.fp16, StageType.reshape, 1, 1, 1, 1, inshape[2], inshape[1], inshape[0], 1, 1, inshape[0], None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, None, myriad_conf, arguments, new_x=new_shape.dim[3], new_y=new_shape.dim[2], new_c=new_shape.dim[1]))
                            last_layer = layer
                            if layer.name == outputNodeName:
                                break
                            if isPower(layer.type):
                                power_params = np.array([
                                 (layer.power_param.shift,
                                  layer.power_param.scale, layer.power_param.power)], dtype=np.dtype('<f4'))
                                power_node = NetworkStage(layer.name, top, StorageOrder.orderZYX, 0, 0, PadStyle.none, DataType.fp16, DataType.fp16, get_caffe_op_type(layer), 1, 1, 1, 1, inshape[2], inshape[1], inshape[0], 0, 0, get_caffe_output_channels(layer, inshape, top, network), None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, 0, myriad_conf, args=arguments, opParams=power_params)
                                network.attach(power_node)
                                last_layer = layer
                                if layer.name == outputNodeName:
                                    break
                                if isCrop(layer.type):
                                    crop_axis = layer.crop_param.axis
                                    if crop_axis < 0:
                                        crop_axis += 4
                                    if crop_axis == 0:
                                        throw_error(ErrorTable.AttemptedBatchMode)
                                    crop_offset = np.array([0, 0, 0], np.dtype('<u4'))
                                    for offset_i in range(0, 3):
                                        if offset_i >= crop_axis - 1:
                                            if len(layer.crop_param.offset) == 1:
                                                crop_offset[offset_i] = layer.crop_param.offset[0]
                                        else:
                                            crop_offset[offset_i] = layer.crop_param.offset[offset_i - (crop_axis - 1)]

                                    crop_offset = np.array([crop_offset[2], crop_offset[1],
                                     crop_offset[0]], dtype=np.dtype('<u4'))
                                    ref_bottom = network.search_several(layer.bottom[1])
                                    ref_bottom_dimX = ref_bottom.outputDimX
                                    ref_bottom_dimY = ref_bottom.outputDimY
                                    ref_bottom_dimZ = ref_bottom.outputDimZ
                                    ref_dims = {0: (ref_bottom_dimX, ref_bottom_dimY, ref_bottom_dimZ),
                                     1: (ref_bottom_dimX, ref_bottom_dimY, inshape[0]),
                                     2: (ref_bottom_dimX, inshape[1], inshape[0])}
                                    new_x, new_y, new_c = ref_dims.get(crop_axis - 1)
                                    crop_node = NetworkStage(layer.name, top, StorageOrder.orderZYX, 0, 0, PadStyle.none, DataType.fp16, DataType.fp16, get_caffe_op_type(layer), 1, 1, 1, 1, inshape[2], inshape[1], inshape[0], 0, 0, get_caffe_output_channels(layer, inshape, top, network), None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, 0, myriad_conf, args=arguments, new_x=new_x, new_y=new_y, new_c=new_c, opParams=crop_offset)
                                    network.attach(crop_node)
                                    last_layer = layer
                                    if layer.name == outputNodeName:
                                        break
                                    if (isConcat(layer.type) or isConvolution(layer.type) and get_caffe_kernel_size(layer)[0] > 1 or isDeconvolution(layer.type) and get_caffe_kernel_size(layer)[0] > 1) and len(curslicing) > 0:
                                        for slice in curslicing:
                                            for i in range(len(top)):
                                                if top[i] == slice[0]:
                                                    slicename = layer.name + '_Slice' + str(slice[1]) + '_' + str(slice[2])
                                                    network.attach(NetworkStage(slicename, [slice[0]], StorageOrder.orderZYX, 0, 0, PadStyle.caffe, DataType.fp16, DataType.fp16, StageType.copy, 1, 1, 1, 1, inshape[2], inshape[1], inshape[0], 1, 1, slice[2] - slice[1], None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, curslicing, myriad_conf, args=arguments))
                                                    top[i] = slicename

                                    if arguments.explicit_concat and isConcat(layer.type):
                                        outstride = 2 * sum((prev_output_shape[idx][0] for idx in range(len(top))))
                                        for idx, prev in enumerate(top):
                                            if idx == 0:
                                                substagename = layer.name
                                            else:
                                                substagename = layer.name + '_' + ('input' if prev is None else prev)
                                            node = NetworkStage(substagename, top if idx == 0 else [top[idx]], StorageOrder.orderZYX, 0, 0, PadStyle.caffe, DataType.fp16, DataType.fp16, StageType.copy, 1, 1, 1, 1, prev_output_shape[idx][2], prev_output_shape[idx][1], prev_output_shape[idx][0], 1, 1, prev_output_shape[idx][0], None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, curslicing, myriad_conf, args=arguments)
                                            if idx == 0:
                                                firstnode = node
                                            network.attach(node)
                                            if idx == 0:
                                                if layer.name == outputNodeName:
                                                    outputPointer, outputIndex = node.setoutput(outstride, 0, MemoryIndex.output.value)
                                                else:
                                                    outputPointer, outputIndex = node.setoutput(outstride)
                                            else:
                                                node.setoutput(outstride, outputPointer, outputIndex)
                                            outputPointer = outputPointer + 2 * prev_output_shape[idx][0]

                                        if layer.name == outputNodeName:
                                            firstnode.isoutput = True
                                            break
                                        if not isReLU(layer.type) and not isConcat(layer.type) and not isSlice(layer.type) and not isELU(layer.type):
                                            ngroups = get_caffe_group(layer)
                                            addednodes = []
                                            addednames = []
                                            for group in range(ngroups):
                                                taps = get_caffe_params(layer, net.params)[0]
                                                bias = get_caffe_params(layer, net.params)[1]
                                                prev = top
                                                layername = layer.name
                                                if ngroups > 1:
                                                    curslicing = []
                                                    curslicing.append([top[0] if top is-not None else None, inshape[0] // ngroups * group, inshape[0] // ngroups * (group + 1)])
                                                    taps = taps[taps.shape[0] // ngroups * group:taps.shape[0] // ngroups * (group + 1)]
                                                    if bias is-not None:
                                                        bias = bias[bias.shape[0] // ngroups * group:bias.shape[0] // ngroups * (group + 1)]
                                                    if get_caffe_kernel_size(layer)[0] > 1:
                                                        if top is None:
                                                            slicename = 'input'
                                                        else:
                                                            slicename = top[0] if isinstance(top[0], str) else top[0][0]
                                                        slicename = slicename + '_s' + str(group)
                                                        network.attach(NetworkStage(slicename, top, StorageOrder.orderZYX, 0, 0, PadStyle.caffe, DataType.fp16, DataType.fp16, StageType.copy, 1, 1, 1, 1, inshape[2], inshape[1], inshape[0], 1, 1, inshape[0] // ngroups, None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, curslicing, myriad_conf, args=arguments))
                                                        prev = [
                                                         slicename]
                                                    addednames.append(layer.name + '_p' + str(group))
                                                    layername = layer.name + '_p' + str(group)
                                                node = NetworkStage(layername, prev, StorageOrder.orderZYX, get_caffe_op_padding(layer)[0], get_caffe_op_padding(layer)[1], PadStyle.caffe, DataType.fp16, DataType.fp16, get_caffe_op_type(layer), get_caffe_op_radix(layer)[0], get_caffe_op_radix(layer)[1], get_caffe_op_stride(layer)[0], get_caffe_op_stride(layer)[1], inshape[2], inshape[1], inshape[0], get_caffe_kernel_size(layer)[0], get_caffe_kernel_size(layer)[1], get_caffe_output_channels(layer, inshape, top, network) // ngroups, taps, TapsOrder.orderKCHW, bias, None, StageType.none, None, 0, 0, curslicing, myriad_conf, args=arguments)
                                                network.attach(node)
                                                addednodes.append(node)

                                            if ngroups > 1:
                                                if idx == nlayers - 1:
                                                    NetworkStage.concat(addednodes)
                                                else:
                                                    concat_tracker.append((layer.name, addednames))
                                        else:
                                            caffe_apply_minor_op(network, layer, top)
                                        last_layer = layer
                                        if layer.name == outputNodeName:
                                            break

        if last_layer.type == 'Concat':
            nodes = network.search_several(last_layer.bottom)
            NetworkStage.concat(nodes)
        if outputNodeName != None:
            if inputNodeName != None:
                net.blobs[input_bottom].data[...] = input_data
                try:
                    net.forward(start=startNodeName, end=outputNodeName)
                except:
                    throw_error(ErrorTable.NoOutputNode, outputNodeName + '/' + startNodeName)

            else:
                net.blobs['data'].data[...] = input_data
                try:
                    net.forward(end=outputNodeName)
                except:
                    throw_error(ErrorTable.NoOutputNode, outputNodeName)

        elif inputNodeName != None:
            net.blobs[input_bottom].data[...] = input_data
            net.forward(start=startNodeName)
        else:
            net.blobs['data'].data[...] = input_data
            net.forward()
    if file_gen:
        try:
            np.save(filename + '_expected.npy', net.blobs[outputNodeName].data[0].astype(dtype=np.float16))
        except:
            throw_error(ErrorTable.NoOutputNode, extra=net.blobs.keys())

        network.outputTensor = net.blobs[outputNodeName].data.shape
        if len(network.outputTensor) == 4:
            network.outputTensor = network.outputTensor[1:]
            network.outputTensor = zyx_to_yxz_Dimension_only(network.outputTensor)
    return network
# okay decompiling CaffeParser.pyc
