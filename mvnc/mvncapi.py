# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./mvnc/mvncapi.py
# Compiled at: 2017-07-12 09:45:26
# Size of source mod 2**32: 6360 bytes
import sys
import numpy
from enum import Enum
from ctypes import *
try:
    f = CDLL('./libmvnc.so')
except:
    f = CDLL('libmvnc.so')

class Status(Enum):
    OK = 0
    BUSY = -1
    ERROR = -2
    OUT_OF_MEMORY = -3
    DEVICE_NOT_FOUND = -4
    INVALID_PARAMETERS = -5
    TIMEOUT = -6
    MVCMDNOTFOUND = -7
    NODATA = -8
    GONE = -9
    UNSUPPORTEDGRAPHFILE = -10
    MYRIADERROR = -11


class GlobalOption(Enum):
    LOGLEVEL = 0


class DeviceOption(Enum):
    TEMP_LIM_LOWER = 1
    TEMP_LIM_HIGHER = 2
    BACKOFF_TIME_NORMAL = 3
    BACKOFF_TIME_HIGH = 4
    BACKOFF_TIME_CRITICAL = 5
    TEMPERATURE_DEBUG = 6
    THERMALSTATS = 1000
    OPTIMISATIONLIST = 1001
    THERMAL_THROTTLING_LEVEL = 1002


class GraphOption(Enum):
    ITERATIONS = 0
    NETWORK_THROTTLE = 1
    DONTBLOCK = 2
    TIMETAKEN = 1000
    DEBUGINFO = 1001


def EnumerateDevices():
    name = create_string_buffer(28)
    i = 0
    devices = []
    while True:
        if f.mvncGetDeviceName(i, name, 28) != 0:
            break
        devices.append(name.value.decode('utf-8'))
        i = i + 1

    return devices


def SetGlobalOption(opt, data):
    data = c_int(data)
    status = f.mvncSetDeviceOption(0, opt.value, pointer(data), sizeof(data))
    if status != Status.OK.value:
        raise Exception(Status(status))


def GetGlobalOption(opt):
    if opt == GlobalOption.LOGLEVEL:
        optsize = c_uint()
        optvalue = c_uint()
        status = f.mvncGetDeviceOption(0, opt.value, byref(optvalue), byref(optsize))
        if status != Status.OK.value:
            raise Exception(Status(status))
        return optvalue.value
    optsize = c_uint()
    optdata = POINTER(c_byte)()
    status = f.mvncGetDeviceOption(0, opt.value, byref(optdata), byref(optsize))
    if status != Status.OK.value:
        raise Exception(Status(status))
    v = create_string_buffer(optsize.value)
    memmove(v, optdata, optsize.value)
    return v.raw


class Device:

    def __init__(self, name):
        self.handle = c_void_p()
        self.name = name

    def OpenDevice(self):
        status = f.mvncOpenDevice(bytes(bytearray(self.name, 'utf-8')), byref(self.handle))
        if status != Status.OK.value:
            raise Exception(Status(status))

    def CloseDevice(self):
        status = f.mvncCloseDevice(self.handle)
        self.handle = c_void_p()
        if status != Status.OK.value:
            raise Exception(Status(status))

    def SetDeviceOption(self, opt, data):
        if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
            data = c_float(data)
        else:
            data = c_int(data)
        status = f.mvncSetDeviceOption(self.handle, opt.value, pointer(data), sizeof(data))
        if status != Status.OK.value:
            raise Exception(Status(status))

    def GetDeviceOption(self, opt):
        if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
            optdata = c_float()
        else:
            if opt == DeviceOption.BACKOFF_TIME_NORMAL or opt == DeviceOption.BACKOFF_TIME_HIGH or opt == DeviceOption.BACKOFF_TIME_CRITICAL or opt == DeviceOption.TEMPERATURE_DEBUG or opt == DeviceOption.THERMAL_THROTTLING_LEVEL:
                optdata = c_int()
            else:
                optdata = POINTER(c_byte)()
            optsize = c_uint()
            status = f.mvncGetDeviceOption(self.handle, opt.value, byref(optdata), byref(optsize))
            if status != Status.OK.value:
                raise Exception(Status(status))
            if opt == DeviceOption.TEMP_LIM_HIGHER or opt == DeviceOption.TEMP_LIM_LOWER:
                return optdata.value
            if opt == DeviceOption.BACKOFF_TIME_NORMAL or opt == DeviceOption.BACKOFF_TIME_HIGH or opt == DeviceOption.BACKOFF_TIME_CRITICAL or opt == DeviceOption.TEMPERATURE_DEBUG or opt == DeviceOption.THERMAL_THROTTLING_LEVEL:
                return optdata.value
            v = create_string_buffer(optsize.value)
            memmove(v, optdata, optsize.value)
        if opt == DeviceOption.OPTIMISATIONLIST:
            l = []
            for i in range(40):
                if v.raw[i * 50] != 0:
                    ss = v.raw[i * 50:]
                    end = ss.find(0)
                    l.append(ss[0:end].decode())

            return l
        if opt == DeviceOption.THERMALSTATS:
            return numpy.frombuffer(v.raw, dtype=numpy.float32)
        return int.from_bytes(v.raw, byteorder='little')

    def AllocateGraph(self, graphfile):
        hgraph = c_void_p()
        status = f.mvncAllocateGraph(self.handle, byref(hgraph), graphfile, len(graphfile))
        if status != Status.OK.value:
            raise Exception(Status(status))
        return Graph(hgraph)


class Graph:

    def __init__(self, handle):
        self.handle = handle
        self.userobjs = {}

    def SetGraphOption(self, opt, data):
        data = c_int(data)
        status = f.mvncSetGraphOption(self.handle, opt.value, pointer(data), sizeof(data))
        if status != Status.OK.value:
            raise Exception(Status(status))

    def GetGraphOption(self, opt):
        if opt == GraphOption.ITERATIONS or opt == GraphOption.NETWORK_THROTTLE or opt == GraphOption.DONTBLOCK:
            optdata = c_int()
        else:
            optdata = POINTER(c_byte)()
        optsize = c_uint()
        status = f.mvncGetGraphOption(self.handle, opt.value, byref(optdata), byref(optsize))
        if status != Status.OK.value:
            raise Exception(Status(status))
        if opt == GraphOption.ITERATIONS or opt == GraphOption.NETWORK_THROTTLE or opt == GraphOption.DONTBLOCK:
            return optdata.value
        v = create_string_buffer(optsize.value)
        memmove(v, optdata, optsize.value)
        if opt == GraphOption.TIMETAKEN:
            return numpy.frombuffer(v.raw, dtype=numpy.float32)
        if opt == GraphOption.DEBUGINFO:
            return v.raw[0:v.raw.find(0)].decode()
        return int.from_bytes(v.raw, byteorder='little')

    def DeallocateGraph(self):
        status = f.mvncDeallocateGraph(self.handle)
        self.handle = 0
        if status != Status.OK.value:
            raise Exception(Status(status))

    def LoadTensor(self, tensor, userobj):
        tensor = tensor.tostring()
        userobj = py_object(userobj)
        key = c_long(addressof(userobj))
        self.userobjs[key.value] = userobj
        status = f.mvncLoadTensor(self.handle, tensor, len(tensor), key)
        if status == Status.BUSY.value:
            return False
        if status != Status.OK.value:
            del self.userobjs[key.value]
            raise Exception(Status(status))
        return True

    def GetResult(self):
        tensor = c_void_p()
        tensorlen = c_uint()
        userobj = c_long()
        status = f.mvncGetResult(self.handle, byref(tensor), byref(tensorlen), byref(userobj))
        if status == Status.NODATA.value:
            return (None, None)
        if status != Status.OK.value:
            raise Exception(Status(status))
        v = create_string_buffer(tensorlen.value)
        memmove(v, tensor, tensorlen.value)
        tensor = numpy.frombuffer(v.raw, dtype=numpy.float16)
        retuserobj = self.userobjs[userobj.value]
        del self.userobjs[userobj.value]
        return (
         tensor, retuserobj.value)
# okay decompiling mvncapi.pyc
