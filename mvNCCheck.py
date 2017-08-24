# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./mvNCCheck.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 5448 bytes
import os
import sys
import argparse
import numpy as np
from Controllers.EnumController import *
from Controllers.FileIO import *
from Models.Blob import *
from Models.EnumDeclarations import *
from Models.MyriadParam import *
from Views.Validate import *
major_version = np.uint32(2)
release_number = np.uint32(0)

def parse_args():
    parser = argparse.ArgumentParser(description='mvNCCheck.py validates a Caffe or Tensorflow network on the Movidius Neural Compute Stick\n')
    parser.add_argument('network', type=str, help='Network file (.prototxt, .meta, .pb, .protobuf)')
    parser.add_argument('-w', dest='weights', type=str, help='Weights file (override default same name of .protobuf)')
    parser.add_argument('-in', dest='inputnode', type=str, help='Input node name')
    parser.add_argument('-on', dest='outputnode', type=str, help='Output node name')
    parser.add_argument('-i', dest='image', type=str, default='Debug', help='Image to process')
    parser.add_argument('-s', dest='nshaves', type=int, default=1, help='Number of shaves (default 1)')
    parser.add_argument('-is', dest='inputsize', nargs=2, type=int, help="Input size for networks that don't provide an input shape, width and height expected")
    parser.add_argument('-S', dest='scale', type=float, help='Scale the input by this amount, before mean')
    parser.add_argument('-M', dest='mean', type=str, help='Numpy file or constant to subtract from the image, after scaling')
    parser.add_argument('-id', dest='expectedid', type=int, help='Expected output id for validation')
    args = parser.parse_args()
    return args


class Arguments:

    def __init__(self, network, image, inputnode, outputnode, inputsize, nshaves, weights, extargs):
        self.net_description = network
        filetype = network.split('.')[-1]
        self.parser = Parser.TensorFlow
        if filetype in ('prototxt', ):
            self.parser = Parser.Caffe
            if weights is None:
                weights = network[:-8] + 'caffemodel'
                if not os.path.isfile(weights):
                    weights = None
        self.conf_file = network[:-len(filetype)] + 'conf'
        if not os.path.isfile(self.conf_file):
            self.conf_file = None
        self.net_weights = weights
        self.input_node_name = inputnode
        self.output_node_name = outputnode
        self.input_size = inputsize
        self.number_of_shaves = nshaves
        self.image = image
        self.raw_scale = 1
        self.mean = None
        self.channel_swap = None
        self.explicit_concat = False
        self.acm = 0
        self.timer = None
        self.number_of_iterations = 2
        self.upper_temperature_limit = -1
        self.lower_temperature_limit = -1
        self.backoff_time_normal = -1
        self.backoff_time_high = -1
        self.backoff_time_critical = -1
        self.temperature_mode = 'Advanced'
        self.network_level_throttling = 1
        self.stress_full_run = 1
        self.stress_usblink_write = 1
        self.stress_usblink_read = 1
        self.debug_readX = 100
        self.mode = 'validation'
        self.outputs_name = 'output'
        self.exp_id = None
        if extargs is not None:
            if hasattr(extargs, 'mean') and extargs.mean is not None:
                self.mean = extargs.mean
            if hasattr(extargs, 'scale') and extargs.scale is not None:
                self.raw_scale = extargs.scale
            if hasattr(extargs, 'expectedid') and extargs.expectedid is not None:
                self.exp_id = extargs.expectedid


def check_net(network, image, inputnode=None, outputnode=None, nshaves=1, inputsize=None, weights=None, extargs=None):
    FileInit()
    args = Arguments(network, image, inputnode, outputnode, inputsize, nshaves, weights, extargs)
    myriad_config = MyriadParam(0, nshaves - 1)
    if args.conf_file is not None:
        get_myriad_info(args, myriad_config)
    filetype = network.split('.')[-1]
    if filetype in ('prototxt', ):
        from Controllers.CaffeParser import parse_caffe
        net = parse_caffe(args, myriad_config, file_gen=True)
    else:
        if filetype in ('pb', 'protobuf', 'meta'):
            from Controllers.TensorFlowParser import parse_tensor
            net = parse_tensor(args, myriad_config, file_gen=True)
        else:
            throw_error(ErrorTable.ParserNotSupported)
        net.finalize()
        net.optimize()
        graph_file = Blob(major_version, net.name, '', myriad_config, net, 'graph')
        graph_file.generate()
        timings, myriad_output = run_myriad(graph_file, args, file_gen=True)
        expected = np.load(args.outputs_name + '_expected.npy')
        result = np.load(args.outputs_name + '_result.npy')
        filename = str(args.outputs_name) + '_val.csv'
        if args.exp_id is not None:
            quit_code = validation(result, expected, args.exp_id, ValidationStatistic.top1, filename, args)
        else:
            quit_code = validation(result, expected, None, ValidationStatistic.top5, filename, args)
    return quit_code


if __name__ == '__main__':
    print('\x1b[1mmvNCCheck v' + '{0:02d}'.format(major_version) + '.' + '{0:02d}'.format(release_number) + ', Copyright @ Movidius Ltd 2016\x1b[0m\n')
    args = parse_args()
    quit_code = check_net(args.network, args.image, args.inputnode, args.outputnode, args.nshaves, args.inputsize, args.weights, args)
    quit(quit_code)
# okay decompiling mvNCCheck.pyc
