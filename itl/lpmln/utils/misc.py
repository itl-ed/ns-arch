"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import numpy as np


def wrap_args(*args):
    """
    Wrap list of arguments, adding whether each arg is variable or not by looking at
    if the first letter is uppercased
    """
    return [(a, type(a)==str and a[0].isupper()) for a in args]

def logit(p, large=float("inf")):
    """ Compute logit of the probability value p """
    if p == 1:
        return large
    elif p == 0:
        return -large
    else:
        return np.log(p/(1-p))

def sigmoid(l):
    """ Compute probability of the logit value l """
    if l == float("inf"):
        return 1
    elif l == float("-inf"):
        return 0
    else:
        return 1 / (1 + np.exp(-l))
