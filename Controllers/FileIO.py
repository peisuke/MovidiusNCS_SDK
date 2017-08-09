# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/FileIO.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 8183 bytes
import warnings
import ctypes
import numpy as np
from Models.EnumDeclarations import *
from Controllers.EnumController import *
data_offset = 0
zero_data_offset = 0
buffer = []
bss_buffer = []
buffer_index = MemoryIndex.workbuffer.value - 1

def FileInit():
    global buffer_index
    global data_offset
    global zero_data_offset
    global bss_buffer
    global buffer
    data_offset = 0
    zero_data_offset = 0
    buffer = []
    bss_buffer = []
    buffer_index = MemoryIndex.workbuffer.value - 1


def get_numpy_element_byte_size(array):
    if array.dtype == np.float32 or array.dtype == np.int32 or array.dtype == np.uint32:
        warnings.warn('\x1b[93mYou are using a large type. ' + 'Consider reducing your data sizes for best performance\x1b[0m')
        return 4
    if array.dtype == np.float16 or array.dtype == np.int16 or array.dtype == np.uint16:
        return 2
    if array.dtype == np.uint8 or array.dtype == np.int8:
        warnings.warn('\x1b[93mYou are using an experimental type. May not be fully functional\x1b[0m')
        return 1
    throw_error(ErrorTable.DataTypeNotSupported, array.dtype)


def align(offset, data, align_to=64):
    rem = offset % align_to
    new_offset = offset if rem == 0 else offset + (align_to - rem)
    if data is not None:
        new_data = np.pad(data.flatten(), (0, int((new_offset - offset) / data.dtype.itemsize)), mode='constant')
    else:
        new_data = None
    return (new_offset, new_data)


def get_buffer(for_data, datatype):
    global data_offset
    buffer_size = len(for_data.flatten()) * dtype_size(datatype)
    buffer_size, for_data = align(buffer_size, for_data, 64)
    buffer.append(for_data)
    data_offset += buffer_size
    return (
     data_offset - buffer_size, len(buffer))


def get_zero_buffer(for_data, datatype):
    global zero_data_offset
    global buffer_index
    RADIX_MAX = 5
    width = for_data.shape[2]
    channels = for_data.shape[0]
    pad = RADIX_MAX // 2 * (width + 1) * channels * dtype_size(datatype)
    buffer_size = len(for_data.flatten()) * dtype_size(datatype) + 2 * pad
    buffer_size, for_data = align(buffer_size, for_data, 64)
    bss_buffer.append(for_data)
    zero_data_offset += buffer_size
    buffer_index += 1
    if zero_data_offset - buffer_size + pad + buffer_index > 41943040:
        throw_error(ErrorTable.NoResources)
    return (
     zero_data_offset - buffer_size + pad, buffer_index)


def replace_buffer(new_data, offset, datatype):
    offset = offset - 1
    if offset < 0:
        return
    buffer_size = len(new_data.flatten()) * dtype_size(datatype)
    buffer_size, new_data = align(buffer_size, new_data)
    buffer[offset] = new_data


def write_data(f):
    for data in buffer:
        f.write(data)


def data_size():
    byte_count = sum([a.flatten().shape[0] * get_numpy_element_byte_size(a) for a in buffer])
    return byte_count


def byte_size(item):
    if type(item) is bytes:
        return len(item)
    return ctypes.sizeof(item)


def get_buffer_start(blob):
    file_size = 0
    if blob.VCS_Fix:
        file_size = byte_size(ctypes.c_uint64(0)) * 4
    file_size += byte_size(blob.filesize)
    file_size += byte_size(blob.version)
    file_size += byte_size(blob.name)
    file_size += byte_size(blob.report_dir)
    file_size += byte_size(blob.stage_count)
    file_size += byte_size(ctypes.c_uint32(0))
    file_size += blob.myriad_params.binary_size()
    file_size += blob.network.head[0].binary_size() * blob.network.count
    file_size += align(file_size, np.zeros(1), align_to=8)[0] - file_size
    return ctypes.c_uint32(file_size)


def estimate_file_size(blob):
    file_size = 0
    if blob.VCS_Fix:
        file_size = byte_size(ctypes.c_uint64(0)) * 4
    file_size += byte_size(blob.filesize)
    file_size += byte_size(blob.version)
    file_size += byte_size(blob.name)
    file_size += byte_size(blob.report_dir)
    file_size += byte_size(blob.stage_count)
    file_size += byte_size(ctypes.c_uint32(0))
    file_size += blob.myriad_params.binary_size()
    file_size += blob.network.head[0].binary_size() * blob.network.count
    file_size += align(file_size, np.zeros(1), align_to=8)[0] - file_size
    file_size += data_size()
    if file_size > 335544320:
        throw_error(ErrorTable.NoResources)
    return ctypes.c_uint32(file_size)
# okay decompiling FileIO.pyc
