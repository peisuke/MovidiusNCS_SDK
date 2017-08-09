# uncompyle6 version 2.11.2
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
# [GCC 5.4.0 20160609]
# Embedded file name: ../../src/./Views/Validate.py
# Compiled at: 2017-07-10 20:21:07
# Size of source mod 2**32: 5923 bytes
from Models.EnumDeclarations import *
from Controllers.Metrics import *
from Controllers.EnumController import *

def top_test(result, exp_id, tolerance):
    data = result.flatten()
    ordered = np.argsort(data)
    ordered = ordered[::-1]
    if int(exp_id) in ordered[:tolerance]:
        print('\nResult: Validation Pass\n')
        top = 0
    else:
        print('\nResult: Validation Fail\n')
        top = 1
    return top


def significant_classification_check(a, b, threshold, classification_type):
    significant_classifications = 0
    significant_matches = np.zeros(1)
    data = a.flatten()
    ordered = np.argsort(data)[::-1]
    for x in ordered:
        if data[x] > threshold:
            significant_classifications += 1

    ordered = ordered[:significant_classifications]
    data2 = b.flatten()
    ordered2 = np.argsort(data2)[::-1]
    ordered2 = ordered2[:significant_classifications]
    ordered_percentages = []
    for x in ordered[:significant_classifications]:
        ordered_percentages.append(data[x])

    ordered_percentages2 = []
    for x in ordered2[:significant_classifications]:
        ordered_percentages2.append(data[x])

    match_percentage = np.sum(ordered == ordered2)
    match_percentage /= ordered.flatten().shape[0]
    match_percentage *= 100
    match_percentage2 = np.in1d(ordered, ordered2)
    match_percentage2 = np.sum(match_percentage2 == True)
    match_percentage2 /= ordered.flatten().shape[0]
    match_percentage2 *= 100
    test_status = 'NO RESULT'
    if classification_type == ValidationStatistic.class_check_exact:
        test_status = 'PASS' if np.all(match_percentage > 90) else 'FAIL'
    if classification_type == ValidationStatistic.class_check_broad:
        test_status = 'PASS' if np.all(match_percentage2 == 100) else 'FAIL'
    print('------------------------------------------------------------')
    print(' Class Validation')
    print(' ------------------------------------------------------------')
    print(' Number of Significant Classifications: {}'.format(significant_classifications))
    print(' Framework S-Classes:      {}'.format(ordered))
    print(' Framework S-%:            {}'.format(ordered_percentages))
    print(' Myriad S-Classifications: {}'.format(ordered2))
    print(' Myriad S-%:               {}'.format(ordered_percentages2))
    print(' Precise Ordering Match: {}%'.format(match_percentage))
    print(' Broad Ordering Match: {}%'.format(match_percentage2))
    print(' Result: {}'.format(test_status))
    print('------------------------------------------------------------')


def top_classifications(values, amount=5):
    data = values.flatten()
    ordered = np.argsort(data)
    ordered = ordered[::-1]
    for i, x in enumerate(ordered[:amount]):
        print(str(i + 1) + ')', x, data[x])


def validation(result, expected, expected_index, validation_type, filename, arguments):
    if validation_type == ValidationStatistic.accuracy_metrics:
        np.set_printoptions(precision=4, suppress=True)
        print('Result: ', result.shape)
        print(result)
        print('Expected: ', expected.shape)
        print(expected)
        compare_matricies(result, expected, filename)
        return 0
    if validation_type == ValidationStatistic.top1:
        exit_code = top_test(result, expected_index, 1)
        print('Result: ', result.shape)
        top_classifications(result, 1)
        print('Expected: ', expected.shape)
        top_classifications(expected, 1)
        compare_matricies(result, expected, filename)
        return exit_code
    if validation_type == ValidationStatistic.top5:
        print('Result: ', result.shape)
        top_classifications(result)
        print('Expected: ', expected.shape)
        top_classifications(expected)
        compare_matricies(result, expected, filename)
        return 0
    if validation_type == ValidationStatistic.class_check_exact:
        exit_code = significant_classification_check(result, expected, arguments.class_test_threshold, validation_type)
        return exit_code
    if validation_type == ValidationStatistic.class_check_broad:
        exit_code = significant_classification_check(result, expected, arguments.class_test_threshold, validation_type)
        return exit_code
    throw_error(ErrorTable.ValidationSelectionError)
# okay decompiling Validate.pyc
