"""
Implements polynomial class. In this codebase polynomials are used to represent
unnormalized probability masses, where exp("a") is considered to be variable
("a" represents a 'hard' weight that asymptotically approaches infinity in the
limit).
"""
import re
from itertools import product
from collections import defaultdict

import numpy as np


class Polynomial:
    """
    Being unnormalized probability masses, values of polynomials are interpreted to
    be nonnegative. Each polynomial is represented by a set of degree-coefficient
    pairs. Coefficients are in log space. Empty polynomial instance represents value
    of zero.
    """
    def __init__(self, terms=None, float_val=None):
        assert terms is None or float_val is None
        if float_val is not None:
            assert float_val >= 0.0
            if float_val == 0.0:
                self.terms = {}
            else:
                self.terms = { 0: float(np.log(float_val)) }
        else:
            if terms is None:
                self.terms = {}
            else:
                self.terms = dict(terms)

    def __repr__(self):
        if len(self.terms) == 0: return "Poly(0.0)"

        degree_repr = { 0: "", 1: "exp(a)", -1: "exp(-a)" }
        terms_repr = [
            (float(np.exp(self.terms[deg])), deg)
            for deg in sorted(self.terms, reverse=True)
        ]
        terms_repr = [
            (
                f"{coeff:.1e}" if abs(coeff)>1e3 else f"{coeff:.1f}",
                degree_repr.get(deg, f"exp({int(deg)}a)")
            )
            for coeff, deg in terms_repr
        ]
        terms_repr = [
            f"{coeff_str}*{deg_str}" if deg_str!="" else coeff_str
            for coeff_str, deg_str in terms_repr
        ]
        return f"Poly({'+'.join(terms_repr)})"

    def __hash__(self):
        return hash(str(sorted(self.terms.items())))

    def __eq__(self, other):
        """ Equal if same value """
        return self.terms == other.terms

    def __add__(self, other):
        """ Sum of two polynomials """
        assert isinstance(other, Polynomial)
        all_degrees = set(self.terms) | set(other.terms)
        poly_sum_terms = {}

        for deg in all_degrees:
            poly_sum_terms[deg] = float(np.logaddexp(
                self.terms.get(deg, float("-inf")),
                other.terms.get(deg, float("-inf"))
            ))

        return Polynomial(terms=poly_sum_terms)

    def __mul__(self, other):
        """ Multiplication of two polynomials """
        assert isinstance(other, Polynomial)
        term_products = product(self.terms.items(), other.terms.items())
        expanded_terms = [
            (deg1+deg2, coeff1+coeff2)
            for (deg1, coeff1), (deg2, coeff2) in term_products
        ]

        poly_product_terms = defaultdict(lambda: float("-inf"))
        for deg, coeff in expanded_terms:
            poly_product_terms[deg] = float(np.logaddexp(poly_product_terms[deg], coeff))

        return Polynomial(terms=poly_product_terms)

    def is_zero(self):
        """ Test if the value of self is zero (empty self.terms) """
        return len(self.terms) == 0

    def primitivize(self):
        """
        Convert to appropriate value in primitive type; float for soft weight, str
        for hard weight. Intended only for single-term polynomials.
        """
        assert len(self.terms)==1
        deg, coeff = list(self.terms.items())[0]
        if deg == 0:
            # Return as float
            return coeff
        else:
            # Return as str
            sign_str = "-" if coeff < 0 else ""
            deg_str = "" if abs(deg)==1 else str(deg)
            return f"{sign_str}{deg_str}a"

    def ratio_at_limit(self, other):
        """
        Dividng a polynomial by another to find the ratio, applying the asymptotic
        limit of alpha (hard weight term) to infinity where appropriate. Does not
        intend to cover cases where self has higher degree than other, raise error
        for such inputs.
        """
        assert isinstance(other, Polynomial)
        if len(other.terms) == 0:
            # Denominator is zero, shouldn't happen
            raise ValueError
        if len(self.terms) == 0:
            # Numerator is zero, just return poly1
            return 0.0

        # Maximum degree of exp(a) for both poly1 and poly2
        max_degree = max(max(self.terms), max(other.terms))
        assert max_degree == max(other.terms)

        if max(self.terms) < max_degree:
            # At the limit of alpha, if the maximum degree in poly1 is lower than the
            # maximum degree for both, the ratio becomes zero
            return 0.0
        else:
            # If poly1 is not dominated by higher degree of poly2, return the ratio of
            # coefficients for the max degree terms, as lower degree terms can safely
            # be disregarded at the limit of alpha
            assert max(self.terms) == max_degree
            return float(np.exp(self.terms[max_degree] - other.terms[max_degree]))

    @staticmethod
    def from_primitive(prm):
        """
        Convert from appropriate value in primitive type; float for soft weight, str
        for hard weight.
        """
        assert type(prm) == float or type(prm) == str

        if type(prm) == float:
            return Polynomial(terms={ 0: prm })
        else:
            w_parse = re.match("(-?)(\d*)(.+)", prm)
            w_sign, w_mult, w_hardRep = w_parse.groups()
            assert w_hardRep == "a"

            w_mult = 1 if w_mult == "" else int(w_mult)
            deg = -w_mult if w_sign == "-" else w_mult

            return Polynomial(terms={ deg: 0.0 })
