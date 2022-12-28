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

def flatten_head_body(head, body):
    """
    Rearrange until any nested conjunctions are all properly flattened out,
    so that rule can be translated into appropriate ASP clause
    """
    from .. import Literal

    head = list(head) if head is not None else []
    body = list(body) if body is not None else []
    conjuncts = head + body
    while any(not isinstance(c, Literal) for c in conjuncts):
        # Migrate any negated conjunctions in head to body
        conjuncts_p = [h for h in head if isinstance(h, Literal)]
        conjuncts_n = [h for h in head if not isinstance(h, Literal)]

        head = conjuncts_p
        body = body + sum(conjuncts_n, [])

        if any(not isinstance(c, Literal) for c in body):
            # Introduce auxiliary literals that are derived when
            # each conjunction in body is satisfied
            # (Not needed, not implemented yet :p)
            raise NotImplementedError

        conjuncts = head + body
    
    return head, body

def unify_mappings(mappings):
    """
    Find the unification of a sequence of mappings; return None if not unifiable
    """
    unified = {}
    for mapping in mappings:
        for k, v in mapping.items():
            if k in unified:
                if unified[k] != v:
                    return None
            else:
                unified[k] = v

    return unified
