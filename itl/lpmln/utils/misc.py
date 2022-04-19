"""
Miscellaneous utility methods that don't classify into other files in utils
"""
def wrap_args(*args):
    """
    Wrap list of arguments, adding whether each arg is variable or not by looking at
    if the first letter is uppercased
    """
    return [(a, a[0].isupper()) for a in args]
