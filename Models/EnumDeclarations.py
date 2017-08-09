# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Models/EnumDeclarations.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 4543 bytes
import numpy as np
from enum import Enum
import warnings

class OperationMode(Enum):
    generation = 0
    validation = 1
    test_validation = 2
    test_generation = 3
    invalid = 5
    profile = 6
    demo = 7
    testTensorFlow = 8
    temperature_profile = 9
    optimization_list = 10


class ValidationStatistic(Enum):
    top1 = 0
    top5 = 1
    accuracy_metrics = 2
    invalid = 3
    class_check_exact = 4
    class_check_broad = 5


class MemoryIndex(Enum):
    none = 0
    input = 1
    output = 2
    blob = 3
    workbuffer = 4


class NetworkLimitation(Enum):
    DDR_Speed_Bound = 0
    DDR_Space_Bound = 1
    Compute_Speed_Bound = 2
    Unsupported_Functions = 3


class StageType(Enum):
    convolution = 0
    max_pooling = 1
    average_pooling = 2
    soft_max = 3
    fully_connected_layer = 4
    none = 5
    relu = 6
    relu_x = 7
    depth_convolution = 8
    bias = 9
    prelu = 10
    LRN = 11
    eltwise_sum = 12
    eltwise_prod = 13
    eltwise_max = 14
    scale = 15
    relayout = 16
    square = 17
    innerlrn = 18
    copy = 19
    sigmoid = 20
    tanh = 21
    deconvolution = 22
    elu = 23
    reshape = 24
    toplanemajor = 25
    power = 26
    crop = 27
    dropout = 28
    maxout = 29
    normalization = 30
    leaky_relu = 31
    r_relu = 32
    BNLL = 33
    abs = 34
    stochastic_pooling = 35
    convolution_HW = 36
    max_pooling_HW = 37
    average_pooling_HW = 38
    fully_connected_layer_HW = 39
    convolution_pooling = 40


class StorageOrder(Enum):
    orderXYZ = 0
    orderXZY = 1
    orderYXZ = 2
    orderYZX = 3
    orderZYX = 4
    orderZXY = 5


class TapsOrder(Enum):
    orderHWCK = 0
    orderKCHW = 1


class PadStyle(Enum):
    none = 0
    tfvalid = 1
    caffe = 2
    tfsame = 3


class DataType(Enum):
    fp64 = 0
    fp32 = 1
    fp16 = 2
    fp8 = 3
    int64 = 4
    int32 = 5
    int16 = 6
    int8 = 7
    int4 = 8
    int2 = 9
    bit = 10


class ErrorTable(Enum):
    Unknown = 0
    CaffeImportError = 1
    PythonVersionError = 2
    CaffeSyntaxError = 3
    StageTypeNotSupported = 4
    StageDetailsNotSupported = 5
    MyriadExeNotPresent = 6
    USBError = 7
    ArgumentErrorDescription = 8
    ArgumentErrorWeights = 9
    ModeSelectionError = 10
    ArgumentErrorExpID = 11
    ArgumentErrorImage = 12
    NoOutputNode = 13
    DataTypeNotSupported = 14
    ParserNotSupported = 15
    InputNotFirstLayer = 16
    GraphConstructionFailure = 17
    ConversionNotSupported = 18
    ArgumentErrorRequired = 19
    InputSyntaxNotSupported = 20
    ValidationSelectionError = 21
    UnrecognizedFileType = 22
    InvalidInputFile = 23
    AttemptedBatchMode = 24
    MyriadRuntimeIssue = 25
    NoUSBBinary = 26
    InvalidNumberOfShaves = 27
    CaffeMemoryError = 28
    TupleSyntaxWrong = 29
    InputFileUnsupported = 30
    USBDataTransferError = 31
    OptimizationParseError = 32
    NoTemperatureRecorded = 33
    TFNotEvaluated = 34
    NoResources = 35


class Parser(Enum):
    TensorFlow = 0
    Caffe = 1
    Torch = 2
    Theano = 3
    Debug = 4
# okay decompiling EnumDeclarations.pyc
