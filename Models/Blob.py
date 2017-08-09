# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Models/Blob.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 2133 bytes
import numpy as np
from ctypes import *
from Controllers.MiscIO import *
from Controllers.FileIO import *
import sys

class Blob:

    def __init__(self, version, name, report_dir, myriad_params, network, blob_name):
        self.version = c_uint32(version)
        self.filesize = c_uint32(16)
        self.name = set_string_range(name, 100).encode('ascii')
        self.report_dir = set_string_range(report_dir, 100).encode('ascii')
        self.myriad_params = myriad_params
        self.network = network
        self.stage_count = c_uint32(self.network.count)
        self.VCS_Fix = True
        self.blob_name = blob_name

    def generate(self):
        with open(self.blob_name, 'wb') as f:
            if self.VCS_Fix:
                f.write(c_uint64(0))
                f.write(c_uint64(0))
                f.write(c_uint64(0))
                f.write(c_uint64(0))
            f.write(estimate_file_size(self))
            f.write(self.version)
            f.write(self.name)
            f.write(self.report_dir)
            f.write(self.stage_count)
            f.write(get_buffer_start(self))
            self.myriad_params.generate(f)
            self.network.generate_info(f)
            self.network.generate_data(f)
# okay decompiling Blob.pyc
