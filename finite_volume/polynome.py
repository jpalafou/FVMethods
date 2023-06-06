import dataclasses
from finite_volume.mathematiques import Fraction, LinearCombination


@dataclasses.dataclass
class Polynome(LinearCombination):
    """
    express a polynomial
    a*x^{n} + b*x^{n-1} + c*x^{n-2} + ...
    as a dictionary
    {n: a, n-1: b, n-2: c, ...}
    """

    def __post_init__(self):
        """
        linear combinations should be sorted and should not contain
        coefficients of 0 unless they are the zero instance {0: 0}
        """
        if self.coeffs == {0: 0}:
            # self is the zero instance, do nothing
            pass
        elif self.coeffs == {} or (
            all(j == 0 for j in self.coeffs.values())
            and list(self.coeffs.keys()) != [0]
        ):
            # if coeffs is empty or if all its values are zero and 0 is not
            # the only index
            object.__setattr__(self, "coeffs", {0: 0})
        else:  # coeffs is nonempty and contains at least one nonzero item
            # sort by degree
            sorted_coeffs = dict(sorted(self.coeffs.items(), reverse=True))
            # remove 0 coefficients if they unless 0x^0 is the only term
            new_coeffs = {}
            for deg, coeff in sorted_coeffs.items():
                if coeff != 0:
                    new_coeffs[deg] = sorted_coeffs[deg]
            object.__setattr__(self, "coeffs", new_coeffs)

    def __str__(self):
        """
        print Polynomial as a function of x
        """
        string = ""
        for deg, coeff in self.coeffs.items():
            if string:
                if coeff < 0:
                    operator = " - "
                else:
                    operator = " + "
            else:
                if coeff < 0:
                    operator = "-"
                else:
                    operator = ""
            if abs(coeff) == 1 and deg != 0:
                value = ""
            else:
                value = str(abs(coeff))
            if deg != 0:
                string = string + f"{operator}{value}x^{deg}"
            else:
                string = string + f"{operator}{value}"
        return string

    def __repr__(self):
        return str(self)

    def one():
        return Polynome({0: 1})

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            product_coeffs = {}
            for i in self.coeffs.keys():
                for j in other.coeffs.keys():
                    coeff = self.coeffs[i] * other.coeffs[j]
                    if i + j not in product_coeffs.keys():
                        product_coeffs[i + j] = coeff
                    else:
                        product_coeffs[i + j] = product_coeffs[i + j] + coeff
            return self.__class__(product_coeffs)
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(
                dict([(i, other * j) for i, j in self.coeffs.items()])
            )
        raise TypeError(
            f"Cannot multiply a {self.__class__.__name__} with" + f"a {type(other)}"
        )

    __rmul__ = __mul__

    def differentiate(self):
        """
        returns the first derivative of a polynomial
        """
        derivative_coeffs = {}
        for deg, coeff in self.coeffs.items():
            if deg != 0:
                derivative_coeffs[deg - 1] = deg * coeff
        return self.__class__(derivative_coeffs)

    def antidifferentiate(self, constant=0):
        integral_coeffs = {0: constant}
        for deg, coeff in self.coeffs.items():
            integral_coeffs[deg + 1] = coeff / (deg + 1)
        return self.__class__(integral_coeffs)

    def eval(self, x: float, as_fraction: bool = False) -> float:
        """
        returns p(x) as an int/float
        """
        if as_fraction:
            out = Fraction.zero()
            if all(isinstance(coeff, Fraction) for deg, coeff in self.coeffs.items()):
                for deg, coeff in self.coeffs.items():
                    out += coeff * x**deg
            elif all(isinstance(coeff, int) for deg, coeff in self.coeffs.items()):
                for deg, coeff in self.coeffs.items():
                    out += Fraction(coeff, 1) * x**deg
        else:
            out = 0
            if all(isinstance(coeff, Fraction) for deg, coeff in self.coeffs.items()):
                for deg, coeff in self.coeffs.items():
                    out += coeff.real * x**deg
            else:
                for deg, coeff in self.coeffs.items():
                    out += coeff * x**deg
        return out

    @classmethod
    def lagrange(cls, x: list, i: int):
        polynome = cls({0: Fraction.one()})
        for j in range(len(x)):
            if j != i:
                denom = x[i] - x[j]
                polynome *= cls({1: Fraction(1, denom), 0: Fraction(-x[j], denom)})
        return polynome
