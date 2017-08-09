# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Views/Graphs.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 6406 bytes
import datetime
from graphviz import Digraph
import math
import warnings
from Controllers.EnumController import *
import numpy as np

def get_normalized_color(start_color, end_color, start_no, end_no, value):
    a_r, a_g, a_b = int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:], 16)
    b_r, b_g, b_b = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:], 16)
    value = float(value)
    if end_no - start_no == 0:
        return '#FFFFFF'
    percentage = (value - start_no) / (end_no - start_no)
    r_diff = b_r - a_r
    g_diff = b_g - a_g
    b_diff = b_b - a_b
    adjusted_r = r_diff * percentage + a_r
    adjusted_g = g_diff * percentage + a_g
    adjusted_b = b_diff * percentage + a_b
    invalid_values = [
     float('NaN'), float('Inf')]
    if math.isnan(adjusted_r) or math.isnan(adjusted_b) or math.isnan(adjusted_g):
        warnings.warn('Non-Finite value detected', RuntimeWarning)
        return '#7A7A7A'
    return '#%X%X%X' % (int(adjusted_r), int(adjusted_g), int(adjusted_b))


def generate_graphviz(net, blob, filename='output'):
    print("Generating Profile Report '" + str(filename) + "_report.html'...")
    dot = Digraph(name=filename, format='svg')
    table = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">\n<TR><TD  BGCOLOR = "#E0E0E0" COLSPAN="3">Layer</TD></TR>\n<TR><TD BGCOLOR = "#88FFFF"> Complexity <br/> (MFLOPs) </TD>\n<TD BGCOLOR = "#FF88FF"> Bandwidth <br/> (MB/s) </TD>\n<TD BGCOLOR = "#FFFF88"> Time <br/> (ms)</TD></TR>\n</TABLE>>\n'
    dot.node('Legend', table, shape='plaintext')
    table = 'input: {}'.format(net.inputTensor.shape)
    dot.node('Input', table)
    ms_min, ms_max = net.head[0].minmax('ms', net.head[0].ms, net.head[0].ms)
    for stage in net.head:
        ms_min, ms_max = stage.minmax('ms', ms_min, ms_max)

    bws_min, bws_max = net.head[0].minmax('BWs', net.head[0].BWs, net.head[0].BWs)
    for stage in net.head:
        bws_min, bws_max = stage.minmax('BWs', bws_min, bws_max)

    flop_min, flop_max = net.head[0].minmax('flops', net.head[0].flops, net.head[0].flops)
    for stage in net.head:
        flop_min, flop_max = stage.minmax('flops', flop_min, flop_max)

    last_nodes = []
    for stage in net.head:
        dot, last = stage.graphviz(dot, ms_min, ms_max, bws_min, bws_max, flop_min, flop_max)
        last_nodes.extend(last)

    channels = 0
    for shape in net.outputInfo[0]:
        channels = channels + shape[2]

    table = 'output: {}'.format([net.outputInfo[0][0][0], net.outputInfo[0][0][1], channels])
    dot.node('Output', table)
    for node in last_nodes:
        if net.search(node).isoutput:
            dot.edge(node, 'Output')

    total_time = 0
    total_bw = 0
    for stage in net.head:
        time, bw = stage.summaryStats()
        total_time += time
        total_bw += bw

    table = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">\n<TR><TD  BGCOLOR = "#C60000" COLSPAN="3">Summary</TD></TR>\n<TR><TD  BGCOLOR = "#E2E2E2" COLSPAN="3">{0} SHV Processors</TD></TR>\n<TR><TD  BGCOLOR = "#DADADA" COLSPAN="3">Inference time {1} ms</TD></TR>\n<TR><TD  BGCOLOR = "#E2E2E2" COLSPAN="3">Bandwidth {2} MB/sec</TD></TR>\n<TR><TD  BGCOLOR = "#DADADA" COLSPAN="3">This network is Compute bound</TD></TR>\n</TABLE>>\n'.format(blob.myriad_params.lastShave.value - blob.myriad_params.firstShave.value + 1, format(total_time, '.2f'), format(total_bw / 1048576 / (total_time / 1000), '.2f'))
    dot.node('Summary', table, shape='plaintext')
    dot.render()
    generate_html_report(filename + '.gv.svg', net.name, filename=filename)


def generate_ete(blob):
    print('Currently does not work alongside caffe integration due to GTK conflicts')


def dataurl(file):
    import base64
    encoded = base64.b64encode(open(file, 'rb').read())
    return 'data:image/svg+xml;base64,' + str(encoded)[2:-1]


def generate_html_report(graph_filename, network_name, filename='output'):
    html_start = '\n<html>\n<head>\n'
    css = '\n<style>\n.container{\n   text-align: center;\n}\nh3{\n    font-weight: 100;\n    font-size: x-large;\n}\n#mvNCLogo, #ReportImage{\n   margin: auto;\n   display: block;\n}\n#mvNCLogo{\n   width: 300px;\n   padding-left: 50px;\n}\n#ReportImage{\n   width: 60%;\n}\n.infobox{\n    text-align: left;\n    margin-left: 2%;\n    font-family: monospace;\n}\n</style>\n'
    html_end = '\n</head>\n<body>\n\n<div class="container">\n    <img id="MovidiusLogo" src="MovidiusLogo.png" />\n    <hr />\n    <h3> Network Analysis </h3>\n    <div class="infobox">\n        <div> Network Model: <b> ' + network_name + ' </b> </div>\n        <div> Generated on: <b>  ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + ' </b> </div>\n    </div>\n    <img id="ReportImage" src=" ' + dataurl(graph_filename) + ' " />\n</div>\n\n</body>\n</html>\n    '
    document = html_start + css + html_end
    f = open(filename + '_report.html', 'w')
    f.write(document)
    f.close()


def generate_temperature_report(data, filename='output'):
    tempBuffer = np.trim_zeros(data)
    if tempBuffer.size == 0:
        throw_error(ErrorTable.NoTemperatureRecorded)
    print(tempBuffer)
    print('Average Temp', np.mean(tempBuffer))
    print('Peak Temp', np.amax(tempBuffer))
    try:
        import matplotlib.pyplot as plt
        plt.plot(tempBuffer)
        plt.ylabel('Temp')
        plt.savefig(filename + '_temperature.png')
    except:
        pass
# okay decompiling Graphs.pyc
