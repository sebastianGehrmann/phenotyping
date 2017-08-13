
'''
Miscellaneous utility functions
'''
import os
import operator
import sys

def is_python3():
    '''
    Return True if the python interpreter is 3
    '''
    return sys.version_info > (3, 0)

def get_basename_without_extension(filepath):
    '''
    Getting the basename of the filepath without the extension
    E.g. 'data/formatted/dischargeSummariesClean.csv' -> 'dischargeSummariesClean'
    '''
    return os.path.basename(os.path.splitext(filepath)[0])


def reverse_dictionary(dict):
    return {v: k for k, v in dict.items()}


def unique_sorted(my_list):
    return sorted(set(my_list))


def sort_dictionary_by_value(my_dict):
    '''
    Example:
    {1: 2, 3: 4, 4: 3, 2: 1, 0: 0} -> [(0, 0), (2, 1), (1, 2), (4, 3), (3, 4)]
    '''
    sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(1))
    return sorted_dict

def sort_dictionary_keys_by_value(my_dict):
    sorted_dict = sort_dictionary_by_value(my_dict)
    output = []
    for kv in sorted_dict :
        output.append(kv[0])
    return output


def create_folder_if_not_exists(dir):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)