# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/Init.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 388 bytes
import warnings

def setup_warnings():
    formatwarning_orig = warnings.formatwarning
    warnings.formatwarning = lambda message, category, filename, lineno, line=None: formatwarning_orig(message, category, filename, lineno, line='')
# okay decompiling Init.pyc
