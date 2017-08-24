# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/TensorFlowParser.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 29265 bytes
import sys
import tensorflow as tf
import google.protobuf as proto
import numpy as np
import math
import re
from Models.Network import *
from Models.NetworkStage import *
from Models.EnumDeclarations import *
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops
placeholder_dict = {}
const_dict = {}
node_dict = {}
variable_dict = {}
concat_tracker = []
reshape_tracker = []
identity_tracker = []
inputnode = 'input'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def apply_padding(pad_type, in_dim, kernel_dim, stride_dim):
    if pad_type == b'SAME':
        return same_padding(in_dim, kernel_dim, stride_dim)
    if pad_type == b'VALID':
        return valid_padding(in_dim, kernel_dim, stride_dim)
    print('No Such Pad Type Supported.', pad_type)


def same_padding(in_dim, kernel_dim, stride_dim):
    output_dim = math.ceil(float(in_dim) / float(stride_dim))
    pad = ((output_dim - 1) * stride_dim + kernel_dim - in_dim) / 2
    return (
     output_dim, pad)


def valid_padding(in_dim, kernel_dim, stride_dim):
    output_dim = math.ceil(float(in_dim - kernel_dim + 1) / float(stride_dim))
    pad = 0
    return (
     output_dim, pad)


def get_input(name):
    global identity_tracker
    global concat_tracker
    global reshape_tracker
    global node_dict
    global inputnode
    global const_dict
    if len(concat_tracker) != 0:
        for concat in concat_tracker:
            if concat[0] == name:
                return [
                 concat[1]]

    if len(reshape_tracker) != 0:
        for reshape in reshape_tracker:
            if reshape[0] == name:
                return [reshape[1]]

    if len(identity_tracker) != 0:
        for idn in identity_tracker:
            if idn[0] == name:
                if idn[1] == inputnode:
                    return
                return [idn[1]]

    if name == inputnode:
        return
    if name in const_dict.keys():
        pass
    if name in node_dict.keys():
        return [node_dict[name].unprocessed_name]


def have_first_input(name):
    if name == inputnode:
        return True
    if len(identity_tracker) != 0:
        for idn in identity_tracker:
            if idn[0] == name and idn[1] == inputnode:
                return True

    return False


def strip_tensor_id(word):
    return re.sub(':\\d+', '', word)


def count_inputs(t):
    graph = tf.get_default_graph()
    count = 0
    for node in graph.get_operations():
        for a in node.inputs:
            if a.name == t:
                count = count + 1

    return count


def parse_tensor(arguments, myriad_conf, debug=False, file_gen=False):
    global placeholder_dict
    global identity_tracker
    global inputnode
    global concat_tracker
    global reshape_tracker
    path = arguments.net_description
    image = arguments.image
    output_node_name = arguments.output_node_name
    input_node_name = arguments.input_node_name
    filename = arguments.outputs_name
    if input_node_name != None:
        inputnode = input_node_name
    with tf.Session() as sess:
        filetype = path.split('.')[-1]
        if filetype == 'pb':
            graph_def = graph_pb2.GraphDef()
            with open(path, 'rb') as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)
        else:
            saver = tf.train.import_meta_graph(path)
            if saver != None:
                saver.restore(sess, path[:path.rfind('.')])
        graph = tf.get_default_graph()
        inputTensor = graph.get_tensor_by_name(inputnode + ':0')
        if output_node_name == None:
            output_node_name = 'output'
        outputTensor = graph.get_tensor_by_name(output_node_name + ':0')
        shape = inputTensor.get_shape()
        if shape.dims == None:
            shape = [
             1, 224, 224, 3]
            if arguments.input_size:
                shape = [
                 1, arguments.input_size[1], arguments.input_size[0], 3]
        if image is None or image == 'Debug':
            if type(shape) == tf.TensorShape:
                shape = shape.as_list()
                if None in shape:
                    throw_error(ErrorTable.TFNotEvaluated)
            input_data = np.random.uniform(0, 1, shape)
            if debug:
                print('Input image shape', shape)
        else:
            input_data = parse_img(image, [int(shape[0]), int(shape[3]), int(shape[1]), int(shape[2])], raw_scale=arguments.raw_scale, mean=arguments.mean, channel_swap=arguments.channel_swap)
            input_data = input_data.transpose([0, 2, 3, 1])
        network = Network('TensorFlow Network', input_data)
        res = outputTensor.eval(feed_dict={inputnode + ':0': input_data})
        prev_node = None
        prev_node_label = None
        cnt = 0
        inputfound = False
        for idx, node in enumerate(graph.get_operations()):
            if debug:
                print('       ', idx, node.type, node.name)
                for a in node.inputs:
                    print('           IN:', a.name)

                for a in node.outputs:
                    print('           OUT:', a.name)

            if not inputfound:
                if have_first_input(node.name):
                    inputfound = True
                    if debug:
                        print('Starting to process')
                continue
                if node.type == 'Const':
                    const_dict[node.name] = node.outputs[0].get_shape()
                elif node.type == 'Placeholder':
                    placeholder_dict[node.name] = node.outputs[0].get_shape()
                elif node.type == 'Variable' or node.type == 'VariableV2':
                    variable_dict[node.name] = node.outputs[0].get_shape()
                elif node.type == 'Conv2D':
                    if debug:
                        print('Conv2D')
                    inputs = node.inputs[0]
                    input_shape = inputs.get_shape()
                    if input_shape.dims == None and inputs.name == input_node_name + ':0':
                        input_shape = input_data.shape
                    taps = node.inputs[1]
                    taps_shape = node.inputs[1].get_shape()
                    outputs = node.outputs[0].get_shape()
                    ksize = taps_shape[0]
                    stride = node.get_attr('strides')
                    output_size = [
                     input_shape[0],
                     apply_padding(node.get_attr('padding'), int(input_shape[1]), int(ksize), stride[1])[0],
                     apply_padding(node.get_attr('padding'), int(input_shape[2]), int(ksize), stride[2])[0],
                     outputs[3]]
                    if debug:
                        print(output_size)
                    xyz = (int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
                    node.outputs[0].set_shape(output_size)
                    top = get_input(strip_tensor_id(inputs.name))
                    prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.tfsame if node.get_attr('padding') == b'SAME' else PadStyle.tfvalid, DataType.fp16, DataType.fp16, StageType.convolution, int(taps_shape[1]), int(taps_shape[0]), stride[2], stride[1], xyz[0], xyz[1], xyz[2], int(taps_shape[1]), int(taps_shape[0]), int(taps_shape[3]), np.array(taps.eval()), TapsOrder.orderHWCK, None, None, None, None, 0, 0, myriad_config=myriad_conf, args=arguments)
                    network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1
                elif node.type == 'BiasAdd':
                    if debug:
                        print('BiasAdd')
                    inputs = node.inputs[0].get_shape()
                    bias_data = node.inputs[1]
                    outputs = node.outputs[0].get_shape()
                    if len(inputs) == 4:
                        node.outputs[0].set_shape([inputs[0], inputs[1], inputs[2], outputs[3]])
                    else:
                        if len(inputs) == 2:
                            node.outputs[0].set_shape([inputs[0], inputs[1]])
                        else:
                            print('Unsupported Bias Dimensions')
                        prev_node.addBias(np.array(bias_data.eval()).astype(np.float16))
                        prev_node.changeName(node.name)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                elif node.type == 'MaxPool':
                    if debug:
                        print('MaxPool')
                    inputs = node.inputs[0]
                    input_shape = node.inputs[0].get_shape()
                    outputs = node.outputs[0].get_shape()
                    ksize = node.get_attr('ksize')
                    stride = node.get_attr('strides')
                    pad = 0
                    output_size = [
                     input_shape[0],
                     apply_padding(node.get_attr('padding'), int(input_shape[1]), int(ksize[1]), stride[1])[0],
                     apply_padding(node.get_attr('padding'), int(input_shape[2]), int(ksize[2]), stride[2])[0],
                     outputs[3]]
                    node.outputs[0].set_shape(output_size)
                    top = get_input(strip_tensor_id(inputs.name))
                    if len(input_shape) == 4:
                        xyz = (
                         int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
                    else:
                        xyz = (
                         1, 1, int(input_shape[1]))
                    prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.tfsame if node.get_attr('padding') == b'SAME' else PadStyle.tfvalid, DataType.fp16, DataType.fp16, StageType.max_pooling, ksize[1], ksize[2], stride[1], stride[2], xyz[0], xyz[1], xyz[2], ksize[1], ksize[2], int(output_size[3]), None, None, None, None, None, None, 0, 0, myriad_config=myriad_conf, args=arguments)
                    network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1
                elif node.type == 'Relu':
                    if debug:
                        print('ReLU')
                    inputs = node.inputs[0].get_shape()
                    outputs = node.outputs[0].get_shape()
                    if len(inputs) == 4:
                        node.outputs[0].set_shape([inputs[0], inputs[1], inputs[2], outputs[3]])
                    else:
                        if len(inputs) == 2:
                            node.outputs[0].set_shape([inputs[0], inputs[1]])
                        else:
                            print('Unsupported ReLU Dimensions')
                        prev_node.postOp = StageType.relu
                        prev_node.changeName(node.name)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                elif node.type == 'LRN':
                    if debug:
                        print('LRN')
                    inputs = node.inputs[0]
                    input_shape = node.inputs[0].get_shape()
                    outputs = node.outputs[0].get_shape()
                    node.outputs[0].set_shape([input_shape[0], input_shape[1], input_shape[2], outputs[3]])
                    top = get_input(strip_tensor_id(inputs.name))
                    xyz = (int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
                    bias = np.array([node.get_attr('bias'), node.get_attr('alpha') * (2 * node.get_attr('depth_radius') + 1), node.get_attr('beta'), 0], dtype=np.float16)
                    prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.none, DataType.fp16, DataType.fp16, StageType.LRN, 2 * node.get_attr('depth_radius') + 1, 0, 1, 1, xyz[0], xyz[1], xyz[2], 0, 0, xyz[2], None, None, bias, None, None, None, 0, 0, myriad_config=myriad_conf, args=arguments)
                    network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1
                elif node.type == 'MatMul':
                    if debug:
                        print('FCL / MatMul')
                    inputs = node.inputs
                    input_shape = node.inputs[0].get_shape()
                    taps = node.inputs[1]
                    taps_shape = node.inputs[1].get_shape()
                    outputs = node.outputs[0].get_shape()
                    node.outputs[0].set_shape([node.inputs[0].get_shape()[0],
                     node.inputs[1].get_shape()[1]])
                    top = get_input(strip_tensor_id(inputs[0].name))
                    xyz = (1, 1, int(input_shape[1]))
                    prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.none, DataType.fp16, DataType.fp16, StageType.fully_connected_layer, 1, 1, 1, 1, xyz[0], xyz[1], xyz[2], 1, 1, int(taps_shape[1]), np.array(taps.eval()).astype(np.float16), TapsOrder.orderHWCK, None, None, None, None, 0, 0, myriad_config=myriad_conf, args=arguments)
                    network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1
                elif node.type == 'Softmax' or node.type == 'Sigmoid' or node.type == 'Tanh':
                    if debug:
                        print(node.type)
                    inputs = node.inputs[0]
                    input_shape = node.inputs[0].get_shape()
                    outputs = node.outputs[0].get_shape()
                    if len(input_shape) == 4:
                        node.outputs[0].set_shape([input_shape[0], input_shape[1], input_shape[2], outputs[3]])
                    else:
                        if len(input_shape) == 2:
                            node.outputs[0].set_shape([input_shape[0], input_shape[1]])
                        else:
                            print('Unsupported ' + node.type + ' dimensions')
                        taps_shape = [
                         1, 1, 1, 1]
                        stride = [1, 1, 1, 1]
                        pad = 0
                        if node.type == 'Softmax':
                            stagetype = StageType.soft_max
                        else:
                            if node.type == 'Sigmoid':
                                stagetype = StageType.sigmoid
                            else:
                                stagetype = StageType.tanh
                            top = get_input(strip_tensor_id(inputs.name))
                            if len(input_shape) == 4:
                                xyz = (
                                 int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
                            else:
                                xyz = (
                                 1, 1, int(input_shape[1]))
                            prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.none, DataType.fp16, DataType.fp16, stagetype, int(taps_shape[0]), int(taps_shape[0]), stride[1], stride[2], xyz[0], xyz[1], xyz[2], int(taps_shape[0]), int(taps_shape[0]), int(input_shape[1]), None, None, None, None, None, None, 0, 0, myriad_config=myriad_conf, args=arguments)
                            network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1
                elif node.type == 'AvgPool':
                    if debug:
                        print('Avg Pool')
                    inputs = node.inputs[0]
                    input_shape = node.inputs[0].get_shape()
                    outputs = node.outputs[0].get_shape()
                    ksize = node.get_attr('ksize')
                    stride = node.get_attr('strides')
                    pad = 0
                    output_size = [
                     input_shape[0],
                     apply_padding(node.get_attr('padding'), int(input_shape[1]), int(ksize[1]), stride[1])[0],
                     apply_padding(node.get_attr('padding'), int(input_shape[2]), int(ksize[2]), stride[2])[0],
                     outputs[3]]
                    node.outputs[0].set_shape(output_size)
                    top = get_input(strip_tensor_id(inputs.name))
                    if len(input_shape) == 4:
                        xyz = (
                         int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
                    else:
                        xyz = (
                         1, 1, int(input_shape[1]))
                    prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.tfsame if node.get_attr('padding') == b'SAME' else PadStyle.tfvalid, DataType.fp16, DataType.fp16, StageType.average_pooling, ksize[1], ksize[2], stride[1], stride[2], xyz[0], xyz[1], xyz[2], ksize[1], ksize[2], int(output_size[3]), None, None, None, None, None, None, 0, 0, myriad_config=myriad_conf, args=arguments)
                    network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1
                elif node.type == 'Reshape':
                    if debug:
                        print('Reshape')
                    inputs = node.inputs
                    input_shape = node.inputs[0].get_shape()
                    desired_shape = node.inputs[1].eval()
                    if desired_shape[0] == -1:
                        desired_shape[0] = 1
                    node.outputs[0].set_shape(desired_shape)
                    reshape_tracker += [(strip_tensor_id(node.outputs[0].name), strip_tensor_id(inputs[0].name))]
                elif node.type == 'Shape':
                    if debug:
                        print(node.type, len(node.inputs[0].get_shape()))
                elif node.type == 'Identity':
                    if debug:
                        print('Identity')
                    inputs = node.inputs
                    input_shape = node.inputs[0].get_shape()
                    node.outputs[0].set_shape(node.inputs[0].get_shape())
                    identity_tracker += [(strip_tensor_id(node.outputs[0].name), strip_tensor_id(inputs[0].name))]
                elif node.type == 'NoOp':
                    if debug:
                        print('No OP')
                elif node.type == 'Concat' or node.type == 'ConcatV2':
                    if debug:
                        print('Concat')
                    concat_channel_size = 0
                    inputs = node.inputs
                    for src in inputs:
                        if len(src.get_shape()) >= 4:
                            concat_channel_size += int(src.get_shape()[3])

                    a_input = node.inputs[1].get_shape()
                    node.outputs[0].set_shape([a_input[0], a_input[1], a_input[2], concat_channel_size])
                    rep_arr = []
                    if node.type == 'Concat':
                        for inp in inputs[1:]:
                            rep_arr.append(strip_tensor_id(inp.name))

                    else:
                        for inp in inputs[:-1]:
                            rep_arr.append(strip_tensor_id(inp.name))

                    concat_tracker += [(strip_tensor_id(node.outputs[0].name), rep_arr)]
                elif (node.type == 'Add' or node.type == 'Mul' or node.type == 'Maximum') and strip_tensor_id(node.inputs[0].name) in node_dict.keys() and strip_tensor_id(node.inputs[1].name) in node_dict.keys():
                    if debug:
                        print(node.type)
                    top = [
                     strip_tensor_id(node.inputs[0].name), strip_tensor_id(node.inputs[1].name)]
                    input_shape = node.inputs[0].get_shape()
                    outputs = node.outputs[0].get_shape()
                    if len(input_shape) == 4:
                        xyz = (
                         int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
                    else:
                        xyz = (
                         1, 1, int(input_shape[1]))
                    if node.type == 'Add':
                        op = StageType.eltwise_sum
                    else:
                        if node.type == 'Mul':
                            op = StageType.eltwise_prod
                        else:
                            op = StageType.eltwise_max
                        prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.none, DataType.fp16, DataType.fp16, op, 1, 1, 1, 1, xyz[0], xyz[1], xyz[2], xyz[0], xyz[1], xyz[2], None, None, None, None, None, None, 0, 0, myriad_config=myriad_conf, args=arguments)
                        network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1
                elif node.type == 'Mul' and prev_node_label != None and (node.inputs[0].name == prev_node_label + ':0' or node.inputs[1].name == prev_node_label + ':0'):
                    iidx = 1 if node.inputs[1].name == prev_node_label + ':0' else 0
                    if prev_node.op == StageType.convolution and count_inputs(prev_node_label + ':0') == 1:
                        if debug:
                            print('Mul with constant absorbed into convolution')
                        prev_node.taps = np.multiply(prev_node.taps, node.inputs[1 - iidx].eval())
                        if prev_node.bias is not None:
                            prev_node.bias = np.multiply(prev_node.bias, node.inputs[1 - iidx].eval())
                        prev_node_label = node.name
                    else:
                        if debug:
                            print('Mul with constant')
                        inputs = node.inputs[iidx]
                        input_shape = node.inputs[iidx].get_shape()
                        top = get_input(strip_tensor_id(inputs.name))
                        if len(input_shape) == 4:
                            xyz = (
                             int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
                        else:
                            xyz = (
                             1, 1, int(input_shape[1]))
                        prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.none, DataType.fp16, DataType.fp16, StageType.scale, 0, 0, 1, 1, xyz[0], xyz[1], xyz[2], 0, 0, xyz[2], node.inputs[1 - iidx].eval(), TapsOrder.orderHWCK, None, None, None, None, 0, 0, myriad_config=myriad_conf, args=arguments)
                        network.attach(prev_node)
                        prev_node_label = strip_tensor_id(node.outputs[0].name)
                        node_dict[prev_node_label] = prev_node
                        cnt += 1
                elif node.type == 'Add' and prev_node_label != None and node.inputs[0].name == prev_node_label + ':0':
                    if debug:
                        print('Add (bias)')
                    inputs = node.inputs[0].get_shape()
                    bias_data = None
                    bias_data = node.inputs[1].eval()
                    outputs = node.outputs[0].get_shape()
                    if len(inputs) == 4:
                        node.outputs[0].set_shape([inputs[0], inputs[1], inputs[2], outputs[3]])
                    else:
                        if len(inputs) == 2:
                            node.outputs[0].set_shape([inputs[0], inputs[1]])
                        else:
                            throw_error(ErrorTable.StageDetailsNotSupported, 'Unsupported Bias Dimensions')
                        prev_node.addBias(np.array(bias_data).astype(np.float16))
                        prev_node.changeName(node.name)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                elif node.type == 'Slice':
                    if debug:
                        print('Slice')
                    input_shape = node.inputs[0].get_shape()
                    slicingbegin = node.inputs[1].eval()
                    slicingsize = node.inputs[2].eval()
                    if len(input_shape) != 4 or len(slicingbegin) != 4 or len(slicingsize) != 4 or slicingbegin[0] != 0 or slicingbegin[1] != 0 or slicingbegin[2] != 0 or input_shape[0] != slicingsize[0] or input_shape[1] != slicingsize[1] or input_shape[2] != slicingsize[2]:
                        throw_error(ErrorTable.StageDetailsNotSupported, 'Slice type not supported')
                    top = get_input(strip_tensor_id(node.inputs[0].name))
                    curslicing = []
                    curslicing.append((top, int(slicingbegin[3]), int(slicingbegin[3] + slicingsize[3])))
                    prev_node = NetworkStage(node.name, top, StorageOrder.orderYXZ, 0, 0, PadStyle.none, DataType.fp16, DataType.fp16, StageType.copy, 1, 1, 1, 1, int(input_shape[1]), int(input_shape[2]), int(input_shape[3]), 1, 1, slicingsize[3], None, TapsOrder.orderKCHW, None, None, StageType.none, None, 0, 0, curslicing, myriad_conf, args=arguments)
                    network.attach(prev_node)
                    prev_node_label = strip_tensor_id(node.outputs[0].name)
                    node_dict[prev_node_label] = prev_node
                    cnt += 1
                else:
                    if node.type == 'TruncatedNormal' or node.type == 'Assign' or node.type == 'RandomUniform' or node.type == 'Div' or node.type == 'Mul' or node.type == 'Floor' or node.type == 'Add' or node.type == 'Sub' or node.type == 'Rsqrt' or node.type == 'RandomStandardNormal' or node.type == 'L2Loss' or node.type == 'Pack':
                        pass
                    else:
                        throw_error(ErrorTable.StageDetailsNotSupported, node.type)
                    if node.name == output_node_name:
                        if node.type == 'Concat' or node.type == 'ConcatV2':
                            nodes = network.search_several(get_input(node.name)[0])
                            NetworkStage.concat(nodes)
                        break

        if len(res.shape) == 4:
            network.outputTensor = (
             res.shape[0], res.shape[1], res.shape[2], res.shape[3])
        else:
            network.outputTensor = (
             res.shape[0], 1, 1, res.shape[1])
        if file_gen:
            pass
        try:
            np.save(filename + '_expected.npy', res)
        except:
            throw_error(ErrorTable.NoOutputNode, extra=net.blob.keys())

    return network
# okay decompiling TensorFlowParser.pyc
