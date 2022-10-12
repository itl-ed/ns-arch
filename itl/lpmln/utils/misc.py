"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import numpy as np


def wrap_args(*args):
    """
    Wrap list of arguments, adding whether each arg is variable or not by looking at
    if the first letter is uppercased
    """
    wrapped = []
    for a in args:
        if type(a) == str:
            # Non-function term
            wrapped.append((a, a[0].isupper()))
        elif type(a) == tuple:
            # Function term
            _, f_args = a
            wrapped.append((a, all(fa[0].isupper() for fa in f_args)))
        elif type(a) == int or type(a) == float:
            # Number, definitely not a variable
            wrapped.append((a, False))
        else:
            raise NotImplementedError

    return wrapped

def logit(p, large=float("inf")):
    """ Compute logit of the probability value p """
    if p == 1:
        return large
    elif p == 0:
        if type(large)==str:
            return f"-{large}"
        else:
            assert type(large)==float
            return -large
    else:
        return float(np.log(p/(1-p)))

def sigmoid(l):
    """ Compute probability of the logit value l """
    if l == float("inf") or l == "a":
        return 1
    elif l == float("-inf") or l == "-a":
        return 0
    else:
        return float(1 / (1 + np.exp(-l)))

def cacheable(fn):
    """ Class method decorator that caches output values by input params """
    fn_name = fn.__name__

    def wrapper(*args, **kwargs):
        instance = args[0]
        if args[1:] in instance.cache[fn_name]:
            return instance.cache[fn_name][args[1:]]
        else:
            result = fn(*args, **kwargs)
            instance.cache[fn_name][args[1:]] = result
            return result

    return wrapper
