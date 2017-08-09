# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/DataTransforms.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 3657 bytes
import numpy as np
from Models.EnumDeclarations import *
from Controllers.EnumController import *
from ctypes import *
import math

def zyx_to_yxz_Dimension_only(data, z=0, y=0, x=0):
    z = data[0] if z == 0 else z
    y = data[1] if y == 0 else y
    x = data[2] if x == 0 else x
    return (
     y, x, z)


def zyx_to_yxz(data, data_type):
    return np.moveaxis(data, 0, 2).ravel().reshape(data.shape[1], data.shape[2], data.shape[0])


def yxz_to_zyx(data, y=0, x=0, z=0):
    y = data.shape[0] if y == 0 else y
    x = data.shape[1] if x == 0 else x
    z = data.shape[2] if z == 0 else z
    yxz = data.reshape((y * x, z))
    trans = yxz.transpose()
    trans = np.reshape(trans, (z * x, y))
    trans = np.reshape(trans, (z, y, x))
    return trans


def xyz_to_zyx(data, x=0, y=0, z=0):
    x = data.shape[0] if x == 0 else x
    y = data.shape[1] if y == 0 else y
    z = data.shape[2] if z == 0 else z
    trans = data.swapaxes(0, 2)
    return trans


def xyz_to_yxz(data, y=0, x=0, z=0):
    x = data.shape[0] if x == 0 else x
    y = data.shape[1] if y == 0 else y
    z = data.shape[2] if z == 0 else z
    trans = data.swapaxes(0, 1)
    return trans


def yxz_to_xyz(data, y=0, x=0, z=0):
    y = data.shape[0] if y == 0 else y
    x = data.shape[1] if x == 0 else x
    z = data.shape[2] if z == 0 else z
    trans = data.swapaxes(0, 1)
    return trans


def kchw_to_hwck(data, k=0, c=0, fh=0, fw=0):
    k = data.shape[0] if k == 0 else k
    c = data.shape[1] if c == 0 else c
    fh = data.shape[2] if fh == 0 else fh
    fw = data.shape[3] if fw == 0 else fw
    data = data.reshape((k, c, fh, fw))
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 2, 3)
    return data


def hwck_transpose_correction(data, fh=0, fw=0, c=0, k=0):
    fw = data.shape[0] if fw == 0 else fw
    fh = data.shape[1] if fh == 0 else fh
    c = data.shape[2] if c == 0 else c
    k = data.shape[3] if k == 0 else k
    return data


def merge_buffers_zyz(data):
    pass
# okay decompiling DataTransforms.pyc
