""" Utility functions"""
from typing import Any, List
import numpy as np

def dict_to_array(data: dict, keys: List):
    """construct an array of values from dictionary in the order specified by keys

    Args:
        data (dict): dictionary to look up keys in
        keys (list): list of keys to retrieve in order

    Returns: 
        np.array: array of values from dict in same order as keys
    """
    return np.array([data.get(key) for key in keys])

def wrap_to_pi(values: list):
    """
    Wraps values in list to [-pi, pi]

    Args:
        values (list): list of values

    Returns:
        list: list with wrapped values

    """
    return list((np.array(values) + np.pi) % (2 * np.pi) - np.pi)