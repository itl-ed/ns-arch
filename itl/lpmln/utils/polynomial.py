"""
Utility methods for binary arithmetics of polynomials of exp("a"), represented as
dicts from term degree to (log-)coefficient
"""
from itertools import product
from collections import defaultdict

import numpy as np


def poly_sum(poly1, poly2):
    """ Sum of two polynomials expressed as degree-cofficient dict """
    all_degrees = set(poly1) | set(poly2)
    poly_sum = {}
    for deg in all_degrees:
        poly_sum[deg] = np.logaddexp(
            poly1.get(deg, float("-inf")), poly2.get(deg, float("-inf"))
        )
    return poly_sum

def poly_prod(poly1, poly2):
    """ Multiplication of two polynomials expressed as degree-coefficient dict """
    expanded_terms = [
        (deg1+deg2, coeff1+coeff2)
        for (deg1, coeff1), (deg2, coeff2) in product(poly1.items(), poly2.items())
    ]
    poly_product = defaultdict(lambda: float("-inf"))
    for deg, coeff in expanded_terms:
        poly_product[deg] = np.logaddexp(poly_product[deg], coeff)
    return dict(poly_product)

def poly_ratio_at_limit(poly1, poly2):
    """
    Dividng a polynomial by another to find the ratio poly1/poly2, applying the
    asymptotic limit of alpha (hard weight term) to infinity where appropriate.
    Does not intend to cover cases where poly1 has higher degree than poly2, raise
    error for such inputs.
    """
    if len(poly2) == 0:
        # Denominator is zero, shouldn't happen
        raise ValueError
    if len(poly1) == 0:
        # Numerator is zero, just return poly1
        return 0.0

    # Maximum degree of exp(a) for both poly1 and poly2
    max_degree = max(max(poly1), max(poly2))
    assert max_degree == max(poly2)

    if max(poly1) < max_degree:
        # At the limit of alpha, if the maximum degree in poly1 is lower than the
        # maximum degree for both, the ratio becomes zero
        return 0.0
    else:
        # If poly1 is not dominated by higher degree of poly2, return the ratio of
        # coefficients for the max degree terms, as lower degree terms can safely
        # be disregarded at the limit of alpha
        assert max(poly1) == max_degree
        return float(np.exp(poly1[max_degree] - poly2[max_degree]))
