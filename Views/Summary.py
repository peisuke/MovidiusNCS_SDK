# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Views/Summary.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 1823 bytes
from Models.EnumDeclarations import *
g_total_time = 0
number = 0

def print_summary_of_nodes(node):
    global number
    global g_total_time
    print('%-11i%-28s%15.3f%18.2f%16.2f' % (number, node.unprocessed_name, node.flops, node.BWs, node.ms))
    number += 1
    g_total_time += node.ms


def print_summary_of_network(blob_file):
    print('Network Summary')
    print('\nDetailed Per Layer Profile')
    print('Layer      Name                                 MFLOPs    Bandwidth MB/s        time(ms)')
    print('========================================================================================')
    for stage in blob_file.network.stageslist:
        print_summary_of_nodes(stage)

    print('----------------------------------------------------------------------------------------')
    print('           Total inference time                                         %16.2f' % g_total_time)
    print('----------------------------------------------------------------------------------------\n')
# okay decompiling Summary.pyc
