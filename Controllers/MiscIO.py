# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/MiscIO.py
# Compiled at: 2017-07-12 20:19:19
# Size of source mod 2**32: 19818 bytes
import ctypes
import sys
import struct
import numpy as np
import warnings
import os
import os.path
import time
from enum import Enum
if sys.version_info[:2] == (3, 4):
    sys.path.append('../bin/mvNCToolkit_p34')
elif sys.version_info[:2] == (3, 5):
    sys.path.append('../bin/mvNCToolkit_p35')
sys.path.append('../bin/')
sys.path.append('./')
from mvnc import mvncapi
from Controllers.FileIO import *
from Controllers.DataTransforms import *
import os.path
myriad_debug_size = 120
handler = None
no_conf_warning_thrown = False
device = None

def set_string_range(string, length):
    old_length = len(string)
    formatted_string = ('{:<' + str(length) + '}').format(string)
    list_str = list(formatted_string)
    for i in range(old_length, length):
        list_str[i] = '\x00'

    formatted_string = ''.join(list_str)
    return formatted_string


def get_myriad_info(arguments, myriad_param):
    global device
    if device is None:
        devices = mvncapi.EnumerateDevices()
        if len(devices) == 0:
            throw_error(ErrorTable.USBError, 'No devices found')
        device = mvncapi.Device(devices[0])
        try:
            device.OpenDevice()
        except:
            throw_error(ErrorTable.USBError, 'Error opening device')

        myriad_param.optimization_list = device.GetDeviceOption(mvncapi.DeviceOption.OPTIMISATIONLIST)


def run_myriad(blob, arguments, file_gen=False):
    global device
    net = blob.network
    f = open(blob.blob_name, 'rb')
    blob_file = f.read()
    if device is None:
        devices = mvncapi.EnumerateDevices()
        if len(devices) == 0:
            throw_error(ErrorTable.USBError, 'No devices found')
        device = mvncapi.Device(devices[0])
        try:
            device.OpenDevice()
        except:
            throw_error(ErrorTable.USBError, 'Error opening device')

        net.inputTensor = net.inputTensor.astype(dtype=np.float16)
        print('USB: Transferring Data...')
        if arguments.lower_temperature_limit != -1:
            device.SetDeviceOption(mvncapi.DeviceOption.TEMP_LIM_LOWER, arguments.lower_temperature_limit)
        if arguments.upper_temperature_limit != -1:
            device.SetDeviceOption(mvncapi.DeviceOption.TEMP_LIM_HIGHER, arguments.upper_temperature_limit)
        if arguments.backoff_time_normal != -1:
            device.SetDeviceOption(mvncapi.DeviceOption.BACKOFF_TIME_NORMAL, arguments.backoff_time_normal)
        if arguments.backoff_time_high != -1:
            device.SetDeviceOption(mvncapi.DeviceOption.BACKOFF_TIME_HIGH, arguments.backoff_time_high)
        if arguments.backoff_time_critical != -1:
            device.SetDeviceOption(mvncapi.DeviceOption.BACKOFF_TIME_CRITICAL, arguments.backoff_time_critical)
        device.SetDeviceOption(mvncapi.DeviceOption.TEMPERATURE_DEBUG, 1 if arguments.temperature_mode == 'Simple' else 0)
        graph = device.AllocateGraph(blob_file)
        graph.SetGraphOption(mvncapi.GraphOption.ITERATIONS, arguments.number_of_iterations)
        graph.SetGraphOption(mvncapi.GraphOption.NETWORK_THROTTLE, arguments.network_level_throttling)
        sz = net.outputTensor
        for y in range(arguments.stress_full_run):
            if arguments.timer:
                import time
                ts = time.time()
            graph.LoadTensor(net.inputTensor, None)
            try:
                myriad_output, userobj = graph.GetResult()
            except Exception as e:
                if e.args[0] == mvncapi.Status.MYRIADERROR:
                    debugmsg = graph.GetGraphOption(mvncapi.GraphOption.DEBUGINFO)
                    throw_error(ErrorTable.MyriadRuntimeIssue, debugmsg)
                else:
                    throw_error(ErrorTable.MyriadRuntimeIssue, e.args[0])

            if arguments.timer:
                ts2 = time.time()
                print('\x1b[94mTime to Execute : ', str(round((ts2 - ts) * 1000, 2)), ' ms\x1b[39m')
            print('USB: Myriad Execution Finished')

        timings = graph.GetGraphOption(mvncapi.GraphOption.TIMETAKEN)
        if arguments.mode in [OperationMode.temperature_profile]:
            tempBuffer = device.GetDeviceOption(mvncapi.DeviceOption.THERMALSTATS)
        throttling = device.GetDeviceOption(mvncapi.DeviceOption.THERMAL_THROTTLING_LEVEL)
        if throttling == 1:
            print('*********** THERMAL THROTTLING INITIATED ***********')
        if throttling == 2:
            print('************************ WARNING ************************')
            print('*           THERMAL THROTTLING LEVEL 2 REACHED          *')
            print('*********************************************************')
        myriad_output = myriad_output.reshape(sz)
        if arguments.mode in [OperationMode.temperature_profile]:
            net.temperature_buffer = tempBuffer
        if arguments.parser == Parser.Caffe:
            if net.outputNeedsTransforming and len(myriad_output.shape) > 2:
                if len(myriad_output.shape) == 4:
                    myriad_output = myriad_output.reshape(myriad_output.shape[1:])
                if file_gen:
                    np.save(arguments.outputs_name + '_result.npy', myriad_output)
                myriad_output = yxz_to_zyx(myriad_output)
        else:
            if arguments.parser == Parser.TensorFlow:
                myriad_output = myriad_output.reshape(myriad_output.shape[1:])
            else:
                throw_error(ErrorTable.ParserNotSupported, string)
            if file_gen:
                np.save(arguments.outputs_name + '_result.npy', myriad_output)
            print('USB: Myriad Connection Closing.')
            graph.DeallocateGraph()
            device.CloseDevice()
            print('USB: Myriad Connection Closed.')
    return (
     timings, myriad_output)


def run_emulation(blob):
    net = blob.network
    print(net.name)


def parse_img(path, new_size, raw_scale=1, mean=None, channel_swap=None):
    import PIL
    from PIL import Image
    import skimage
    import skimage.io
    import skimage.transform
    if path == 'None' or path == None:
        return np.ones(new_size)
    if path == 'None' or path is None:
        print('No Image Detected, Using Array of Ones')
        return np.ones(new_size)
    lenet_8_special_case = False
    if new_size == [1, 8, 28, 28]:
        lenet_8_special_case = True
    if path.split('.')[-1].lower() in ('png', 'jpeg', 'jpg', 'bmp', 'gif'):
        greyscale = True if new_size[2] == 1 else False
        data = skimage.img_as_float(skimage.io.imread(path, as_grey=greyscale)).astype(np.float32)
    elif path.split('.')[-1] in ('npy', ):
        im = np.load(path)
        if len(im.shape) == 2:
            if im.shape[0] != new_size[2] or im.shape[1] != new_size[3]:
                throw_error(ErrorTable.InvalidInputFile)
        elif len(im.shape) == 3:
            if im.shape[0] != new_size[2] or im.shape[1] != new_size[3]:
                throw_error(ErrorTable.InvalidInputFile)
        else:
            throw_error(ErrorTable.InvalidInputFile)
        data = np.asarray(im)
    else:
        if path.split('.')[-1] in ('mat', ):
            print('Filetype not officially supported use at your own peril: MAT File')
            import scipy.io
            im = scipy.io.loadmat(path)
            data = np.asarray(im)
        else:
            print('Unsupported')
            throw_error(ErrorTable.InputFileUnsupported)
        if len(data.shape) == 2:
            if lenet_8_special_case:
                tmp = np.zeros((1, 8, data.shape[0], data.shape[1]))
                tmp[0][:] = data
                return tmp
            data = data[:, :, np.newaxis]
        data = skimage.transform.resize(data, new_size[2:])
        data = np.transpose(data, (2, 0, 1))
        data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
        data *= raw_scale
    if mean is not None:
        try:
            mean = np.load(mean).mean(1).mean(1)
            mean_arr = np.zeros(data.shape[1:])
            for x in range(mean.shape[0]):
                mean_arr[x].fill(mean[x])

            data[0] -= mean_arr
        except:
            if mean[0] >= '0' and mean[0] <= '9':
                data = data - float(mean)
            else:
                raise

        if channel_swap != None:
            data[0] = data[0][np.argsort(channel_swap), :, :]
    return data


def predict_parser(net_desc):
    filetype = net_desc.split('.')[-1]
    if filetype in ('prototxt', ):
        return Parser.Caffe
    if filetype in ('pb', 'protobuf', 'txt', 'meta'):
        return Parser.TensorFlow
    throw_error(ErrorTable.UnrecognizedFileType)


def parse_optimization(line, stage):
    a = 'opt_'
    a += stage_as_label(stage.op) + '_'
    a += line
    b = 'opt_'
    b += stage_as_label(stage.op) + '_'
    b += str(stage.radixX) + '_'
    b += str(stage.radixY) + '_'
    b += line
    c = 'opt_'
    c += stage_as_label(stage.op) + '_'
    c += str(stage.radixX) + '_'
    c += str(stage.radixY) + '_'
    c += str(stage.strideX) + '_'
    c += str(stage.strideY) + '_'
    c += line
    d = 'opt_'
    d += stage_as_label(stage.op) + '_M_N_'
    d += str(stage.strideX) + '_'
    d += str(stage.strideY) + '_'
    d += line
    return [
     a, b, c, d]


def debug_label(s, line):
    if line == s:
        return True
    return False


def check_generic_label(line, stage):
    s = stage_as_label(stage.op) + ':'
    s += str(stage.radixX) + 'x'
    s += str(stage.radixY) + '_s'
    s += str(stage.strideX) + '_'
    s += str(stage.strideY)
    if debug_label(s, line):
        return True
    s = stage_as_label(stage.op) + ':'
    s += str(stage.radixX) + 'x'
    s += str(stage.radixY) + '_s'
    s += str(stage.strideX)
    if debug_label(s, line):
        return True
    s = stage_as_label(stage.op) + ':'
    s += str(stage.radixX) + 'x'
    s += str(stage.radixY)
    if debug_label(s, line):
        return True
    s = stage_as_label(stage.op)
    if debug_label(s, line):
        return True
    return False


def parseOptimizations(myriad_config, opt_controller):
    print(myriad_config.optimization_list)
    for opt in myriad_config.optimization_list:
        parts = opt.split('_')
        parts += [None] * (7 - len(parts))
        op_name = parts[1]
        conf = {'radixX': parts[2],
         'radixY': parts[3],
         'strideX': parts[4],
         'strideY': parts[5],
         'name_of_opt': parts[6]}
        opt_controller.add_available_optimization(op_name, conf)


def readOptimisationMask(name, stage, myriad_config, args):
    global no_conf_warning_thrown
    defaultOptimisation = 2147483648
    startDefault = defaultOptimisation
    if myriad_config.optimization_list == None or args.conf_file == 'optimisation.conf' and not os.path.isfile(args.conf_file):
        return defaultOptimisation
        try:
            with open(args.conf_file) as f:
                found = 0
                optimisations = 0
                opt_selected = False
                shv = 0
                for line in f:
                    line = line.rstrip()
                    if line in ('generic optimisations', 'generic'):
                        found = 2
                        optimisations = 0
                        opt_selected = False
                    elif line == name:
                        found = 1
                        optimisations = defaultOptimisation
                        shv = 0
                        opt_selected = False
                    elif line == '':
                        if found == 2 or found == 3 or found == 5:
                            if shv == 0:
                                defaultOptimisation = optimisations | defaultOptimisation
                            found = 0
                        elif found == 1 or found == 4:
                            if shv == 0:
                                optimisations = defaultOptimisation | optimisations
                            print('Layer (a)', name, 'use the optimisation mask which is: ', format(optimisations, '#0x'))
                            return optimisations
                    else:
                        if found == 1:
                            opt_lines = parse_optimization(line, stage)
                            for opt_line in opt_lines:
                                if opt_line in myriad_config.optimization_list and not opt_selected:
                                    print('Spec opt found', opt_line, ' 1<<', myriad_config.optimization_list.index(opt_line))
                                    if optimisations == defaultOptimisation:
                                        optimisations = 0
                                    defaultOptimisation = startDefault
                                    optimisations = optimisations | 1 << myriad_config.optimization_list.index(opt_line)
                                    opt_selected = True
                                    found = 4
                                    if shv == 0:
                                        defaultOptimisation = optimisations | defaultOptimisation

                        if len(line) >= 7 and line[0:7] == 'shaves=':
                            shv = min(int(line[7:]), args.number_of_shaves)
                            optimisations = optimisations | shv << 27
                            found = 6
                        elif found == 3:
                            opt_lines = parse_optimization(line, stage)
                            for opt_line in opt_lines:
                                if opt_line in myriad_config.optimization_list and not opt_selected:
                                    print('Generic Spec opt found', opt_line, ' 1<<', myriad_config.optimization_list.index(opt_line))
                                    optimisations = optimisations | 1 << myriad_config.optimization_list.index(opt_line)
                                    opt_selected = True
                                    found = 5

                        else:
                            if found == 4:
                                pass
                            if len(line) >= 7 and line[0:7] == 'shaves=':
                                shv = min(int(line[7:]), args.number_of_shaves)
                                optimisations = optimisations | shv << 27
                                found = 6
                            elif found == 2 and check_generic_label(line, stage):
                                shv = 0
                                found = 3

                print(found, format(defaultOptimisation, '#0x'))
                if found == 2 or found == 5:
                    defaultOptimisation = optimisations
                elif found == 6:
                    print('Layer (b)', name, 'use the optimisation mask which is: ', format(optimisations, '#0x'))
                    return optimisations
        except FileNotFoundError:
            if not no_conf_warning_thrown:
                throw_warning(ErrorTable.OptimizationParseError)
                no_conf_warning_thrown = True
            return defaultOptimisation

        print('Layer', name, 'use the generic optimisations which is: ', format(defaultOptimisation, '#0x'))
    return defaultOptimisation


# global myriad_debug_size ## Warning: Unused global
# okay decompiling MiscIO.pyc
