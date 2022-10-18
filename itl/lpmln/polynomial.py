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

    def __str__(self):
        if len(self.terms) == 0: return "0.0"

        degree_str = { 0: "", 1: "exp(a)", -1: "exp(-a)" }
        terms_str = [
            (float(np.exp(self.terms[deg])), deg)
            for deg in sorted(self.terms, reverse=True)
        ]
        terms_str = [
            (
                f"{coeff:.1e}" if abs(coeff)>1e3 else f"{coeff:.1f}",
                degree_str.get(deg, f"exp({int(deg)}a)")
            )
            for coeff, deg in terms_str
        ]
        terms_str = [
            f"{coeff_str}*{deg_str}" if deg_str!="" else coeff_str
            for coeff_str, deg_str in terms_str
        ]

        if hasattr(self, "denom_terms"):
            self_denom_poly = Polynomial(self.denom_terms)
            denom_str = str(self_denom_poly)
            return f"({'+'.join(terms_str)})/({denom_str})"
        else:
            return f"{'+'.join(terms_str)}"

    def __repr__(self):
        return f"Poly({str(self)})"

    def __hash__(self):
        return hash(str(sorted(self.terms.items())))

    def __eq__(self, other):
        """ Equal if same value """
        return self.terms == other.terms

    def __add__(self, other):
        """ Sum of two polynomials """
        assert isinstance(other, Polynomial)

        if self.is_zero(): return other
        if other.is_zero(): return self

        if hasattr(self, "denom_terms") or hasattr(other, "denom_terms"):
            # If fractions are involved, need recursive calls
            common_denom = Polynomial(float_val=1.0)
            self_num_poly = Polynomial(terms=self.terms)
            other_num_poly = Polynomial(terms=other.terms)

            if hasattr(self, "denom_terms"):
                # self is a fraction
                common_denom = common_denom * Polynomial(terms=self.denom_terms)
                other_num_poly = other_num_poly * Polynomial(terms=self.denom_terms)
            if hasattr(other, "denom_terms"):
                # other is a fraction
                common_denom = common_denom * Polynomial(terms=other.denom_terms)
                self_num_poly = self_num_poly * Polynomial(terms=other.denom_terms)

            return (self_num_poly + other_num_poly) / common_denom

        else:
            # Base case; sum of two polys without denominators
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

        if self.is_zero() or other.is_zero():
            # Multiplication by zero
            return Polynomial(float_val=0.0)

        if hasattr(self, "denom_terms") or hasattr(other, "denom_terms"):
            # If fractions are involved, need recursive calls
            # and denominators
            self_num_poly = Polynomial(terms=self.terms)
            other_num_poly = Polynomial(terms=other.terms)
            multiplied = self_num_poly * other_num_poly

            new_denom_poly = Polynomial(float_val=1.0)
            if hasattr(self, "denom_terms"):
                # self is a fraction
                new_denom_poly = new_denom_poly * Polynomial(terms=self.denom_terms)
            if hasattr(other, "denom_terms"):
                # other is a fraction
                new_denom_poly = new_denom_poly * Polynomial(terms=other.denom_terms)

            if new_denom_poly.terms != { 0: 0.0 }:
                setattr(multiplied, "denom_terms", new_denom_poly.terms)

            return multiplied

        else:
            # Base case; multiplication of two polys without denominators
            term_products = product(self.terms.items(), other.terms.items())
            expanded_terms = [
                (deg1+deg2, coeff1+coeff2)
                for (deg1, coeff1), (deg2, coeff2) in term_products
            ]

            poly_product_terms = defaultdict(lambda: float("-inf"))
            for deg, coeff in expanded_terms:
                poly_product_terms[deg] = float(np.logaddexp(poly_product_terms[deg], coeff))

            return Polynomial(terms=poly_product_terms)

    def __truediv__(self, other):
        """ Division of self by some other polynomial """
        assert isinstance(other, Polynomial)
        assert not other.is_zero(), "Cannot divide by zero"

        if len(other.terms) == 1:
            # Division by single-term poly is simple
            o_deg = list(other.terms)[0]
            o_coeff = other.terms[o_deg]
            poly_div_terms = {
                deg-o_deg: coeff-o_coeff for deg, coeff in self.terms.items()
            }
            return Polynomial(terms=poly_div_terms)
        else:
            # Division by multi-term poly generally has to involve fractions...
            if hasattr(other, "denom_terms"):
                other_inverted = Polynomial(terms=other.denom_terms)
            else:
                other_inverted = Polynomial(float_val=1.0)
            setattr(other_inverted, "denom_terms", other.terms)

            return self * other_inverted

    def is_zero(self):
        """ Test if the value of self is zero (empty self.terms) """
        return len(self.terms) == 0

    def at_limit(self):
        """
        Return value when applying the asymptotic limit of alpha (hard weight term)
        to infinity where appropriate.
        """
        if self.is_zero():
            # Numerator is zero, just return zero
            return 0.0

        if hasattr(self, "denom_terms"):
            # Has denominator, need to consider both numerator and denominator
            if max(self.terms) > max(self.denom_terms):
                return float("inf")
            elif max(self.terms) < max(self.denom_terms):
                return 0.0
            else:
                max_degree = max(self.terms)
                return float(np.exp(self.terms[max_degree] - self.denom_terms[max_degree]))
        else:
            # Doesn't have denominator
            if max(self.terms) > 0:
                return float("inf")
            elif max(self.terms) < 0:
                return 0.0
            else:
                return float(np.exp(self.terms[0]))

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

    @staticmethod
    def from_primitive(prm):
        """
        Convert from appropriate value (already in log-space) in primitive type;
        float for soft weight, str for hard weight.
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
