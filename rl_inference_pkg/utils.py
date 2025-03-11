""" Utility functions"""
from typing import Any, List
import numpy as np

def dict_to_array(data: dict, keys: List):
    """construct an array of values from dictionary in the order specified by keys

    Arguments:
    dict -- dictionary to look up keys in
    keys -- list of keys to retrieve in orer

    Output: 
    array of values from dict in same order as keys
    """
    return np.array([data.get(key) for key in keys])