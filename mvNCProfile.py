# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./mvNCProfile.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 4234 bytes
import os
import sys
import argparse
import numpy as np
from Controllers.EnumController import *
from Controllers.FileIO import *
from Models.Blob import *
from Models.EnumDeclarations import *
from Models.MyriadParam import *
from Views.Summary import *
from Views.Graphs import *
major_version = np.uint32(2)
release_number = np.uint32(0)

def parse_args():
    parser = argparse.ArgumentParser(description='mvNCProfile.py profiles a Caffe or Tensorflow network on the Movidius Neural Computer Stick\n')
    parser.add_argument('network', type=str, help='Network file (.prototxt, .meta, .pb, .protobuf)')
    parser.add_argument('-w', dest='weights', type=str, help='Weights file (override default same name of .protobuf)')
    parser.add_argument('-in', dest='inputnode', type=str, help='Input node name')
    parser.add_argument('-on', dest='outputnode', type=str, help='Output node name')
    parser.add_argument('-s', dest='nshaves', type=int, default=1, help='Number of shaves (default 1)')
    parser.add_argument('-is', dest='inputsize', nargs=2, type=int, help="Input size for networks that don't provide an input shape, width and height expected")
    args = parser.parse_args()
    return args


class Arguments:

    def __init__(self, network, inputnode, outputnode, inputsize, nshaves, weights):
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
        self.image = 'Debug'
        self.raw_scale = 1
        self.mean = None
        self.channel_swap = None
        self.explicit_concat = False
        self.acm = 0
        self.timer = True
        self.number_of_iterations = 1
        self.upper_temperature_limit = -1
        self.lower_temperature_limit = -1
        self.backoff_time_normal = -1
        self.backoff_time_high = -1
        self.backoff_time_critical = -1
        self.temperature_mode = 'Advanced'
        self.network_level_throttling = 1
        self.stress_full_run = 2
        self.stress_usblink_write = 1
        self.stress_usblink_read = 1
        self.debug_readX = 100
        self.mode = 'profile'
        self.outputs_name = 'output'


def profile_net(network, inputnode=None, outputnode=None, nshaves=1, inputsize=None, weights=None):
    FileInit()
    args = Arguments(network, inputnode, outputnode, inputsize, nshaves, weights)
    myriad_config = MyriadParam(0, nshaves - 1)
    filetype = network.split('.')[-1]
    if args.conf_file is not None:
        get_myriad_info(args, myriad_config)
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
        timings, myriad_output = run_myriad(graph_file, args, file_gen=False)
        net.gather_metrics(timings)
        print_summary_of_network(graph_file)
        generate_graphviz(net, graph_file, filename=args.outputs_name)


if __name__ == '__main__':
    print('\x1b[1mmvNCProfile v' + '{0:02d}'.format(major_version) + '.' + '{0:02d}'.format(release_number) + ', Copyright @ Movidius Ltd 2016\x1b[0m\n')
    args = parse_args()
    profile_net(args.network, args.inputnode, args.outputnode, args.nshaves, args.inputsize, args.weights)
# okay decompiling mvNCProfile.pyc
