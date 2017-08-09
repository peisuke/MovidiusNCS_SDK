# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Controllers/Metrics.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 6059 bytes
import numpy as np
import yaml
from csv import writer
import os
import sys
mdk_root = os.environ['HOME']
OKBLUE = '\x1b[94m'
OKGREEN = '\x1b[92m'
WARNING = '\x1b[93m'
FAIL = '\x1b[91m'
NORMAL = '\x1b[0m'
BOLD = '\x1b[1m'
PURPLE = '\x1b[95m'
NUM_OF_ATTR = 5
THRESHOLDS = [
 2, 1, 0, 1]

class CompareTestOutput:

    def __init__(self):
        line = OKBLUE + BOLD + 'TEST COMPARE: ' + NORMAL
        self.NEW_REPORT = True
        self.test_index = 0

    def debug_prints(self, matrix, fname):
        with open(fname + '.yaml', 'w') as temp:
            yaml.dump(matrix.tolist(), temp)
        with open(fname + '.yaml') as temp:
            loaded = yaml.load(temp)
        loaded = np.array(loaded)

    def metrics(self, a, b):
        ref = np.max(np.abs(b))
        total_values = int(len(b.flatten()))
        diff = np.abs(a - b)
        max_error = np.max(np.abs(a - b))
        mean_error = np.mean(np.abs(a - b))
        l2_error = np.sqrt(np.sum(np.square(a - b)) / total_values)
        if ref == 0:
            max_error = 0 if max_error == 0 else np.inf
            mean_error = 0 if mean_error == 0 else np.inf
            l2_error = 0 if l2_error == 0 else np.inf
        else:
            max_error = max_error / ref * 100
            mean_error = mean_error / ref * 100
            l2_error = l2_error / ref * 100
        percentage_wrong = len(np.extract(diff > 0.02 * ref, diff)) / total_values * 100
        sum_diff = np.sum(np.abs(a - b))
        return [
         max_error, mean_error, percentage_wrong, l2_error, sum_diff]

    def generate_report(self, report_file_obj, result, reference):
        obtained_val = [
         None] * NUM_OF_ATTR
        threshold_val = [None] * NUM_OF_ATTR
        attr_obtained = [None] * NUM_OF_ATTR
        attr_threshold = [None] * NUM_OF_ATTR
        out_array = [[], []]
        attr = [
         None] * NUM_OF_ATTR
        if self.NEW_REPORT == True:
            attr[0] = 'min pixel accuracy'
            attr[1] = 'average pixel accuracy'
            attr[2] = 'percentage of correct values'
            attr[3] = 'pixel-wise l2 error'
            attr[4] = 'global sum difference'
            test_string = 'test index'
            out_array[0].append(test_string.upper())
            for attr_idx in range(0, len(attr)):
                attr_obtained[attr_idx] = 'M' + str(attr_idx + 1) + ' Obtained ' + attr[attr_idx]
                attr_obtained[attr_idx] = attr_obtained[attr_idx].upper()
                out_array[0].append(attr_obtained[attr_idx])

            for attr_idx in range(0, len(attr)):
                attr_threshold[attr_idx] = 'M' + str(attr_idx + 1) + ' threshold ' + attr[attr_idx]
                attr_threshold[attr_idx] = attr_threshold[attr_idx].upper()
                out_array[0].append(attr_threshold[attr_idx])

            string = 'Pass / Fail'
            out_array[0].append(string.upper())
            report_file_obj.writerow(out_array[0])
            self.NEW_REPORT = False
            if result is None and reference is None:
                return True
        obtained_val = self.metrics(result.astype(np.float32), reference)
        test_status = self.matrix_comparison(obtained_val)
        threshold_val[0] = THRESHOLDS[0]
        threshold_val[1] = THRESHOLDS[1]
        threshold_val[2] = THRESHOLDS[2]
        threshold_val[3] = THRESHOLDS[3]
        threshold_val[4] = 'Inf'
        if self.NEW_REPORT == False:
            out_array[1].append(self.test_index)
            for attr_idx in range(0, len(attr)):
                out_array[1].append(obtained_val[attr_idx])

            for attr_idx in range(0, len(attr)):
                out_array[1].append(threshold_val[attr_idx])

            if test_status == True:
                test_status_str = 'Pass'
            else:
                test_status_str = 'Fail'
            out_array[1].append(test_status_str)
            report_file_obj.writerow(out_array[1])
        return test_status

    def matrix_comparison(self, results):
        status = []
        for i in range(4):
            if results[i] > THRESHOLDS[i] or np.isnan(results[0]).any():
                status.append(FAIL + 'Fail' + NORMAL)
            else:
                status.append('Pass')

        print('------------------------------------------------------------')
        print(' Obtained values ')
        print('------------------------------------------------------------')
        print(' Obtained Min Pixel Accuracy: {}% (max allowed={}%), {}'.format(results[0], THRESHOLDS[0], status[0]))
        print(' Obtained Average Pixel Accuracy: {}% (max allowed={}%), {}'.format(results[1], THRESHOLDS[1], status[1]))
        print(' Obtained Percentage of wrong values: {}% (max allowed={}%), {}'.format(results[2], THRESHOLDS[2], status[2]))
        print(' Obtained Pixel-wise L2 error: {}% (max allowed={}%), {}'.format(results[3], THRESHOLDS[3], status[3]))
        print(' Obtained Global Sum Difference: {}'.format(results[4]))
        print('------------------------------------------------------------')
        return status[0] == 'Pass' and status[1] == 'Pass' and status[2] == 'Pass' and status[3] == 'Pass'


def compare_matricies(result, expected, filename=None, common_dimension=4):
    compare_obj = CompareTestOutput()
    f = open(filename, 'w+')
    csv = writer(f)
    compare_obj.generate_report(csv, result, expected)


def check_match(output, expected):
    data = output.flatten()
    sorted = np.argsort(data)
    top1 = True if expected == sorted[0] else False
    top5 = True if expected in [sorted[0], sorted[1], sorted[2], sorted[3], sorted[4]] else False
    return (
     top1, top5)
# okay decompiling Metrics.pyc
