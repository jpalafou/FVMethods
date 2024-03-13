"""
verbose module which defines various useful functions and classes
"""

import dataclasses
import numpy as np


def gauss_lobatto(n: int):
    """
    args:
        n   int, number of quadrature points
    returns:
        nodes, weights
            nodes       array of the n quadrature points on [-1, 1]
            weights     array (size n)
    """
    if n < 2:
        raise BaseException("A Gauss-Lobatto quadrature has at least 2 points.")
    # find ndoes
    p = np.polynomial.legendre.Legendre.basis(n - 1)  # Legendre polynomial
    raw_interior_nodes = p.deriv().roots()  # roots of derivative
    # ensure nodes array is symmetric about a center of 0
    if n % 2 == 0:
        first_half = raw_interior_nodes[: (n - 2) // 2]
        flipped_half = -first_half[::-1]
        nodes = np.concatenate(([-1], first_half, flipped_half, [1]))
    else:
        first_half = raw_interior_nodes[: (n - 2) // 2]
        flipped_half = -first_half[::-1]
        nodes = np.concatenate(([-1], first_half, [0], flipped_half, [1]))
    # find weights
    weights = np.empty(n)
    weights[1:-1] = 2 / (n * (n - 1) * (p(nodes[1:-1]) ** 2))
    weights[0] = 2 / (n * (n - 1))
    weights[-1] = weights[0]

    return nodes, weights


def gcf(mylist: list[int]) -> int:
    """
    returns gcf of the absolute value of a list
    """
    if all(isinstance(i, int) for i in mylist):
        if 0 in mylist:
            raise BaseException("0 has no greatest factor.")
        else:
            cf = 1
            for i in range(2, min(abs(j) for j in mylist) + 1):
                if all(abs(k) % i == 0 for k in mylist):
                    cf = i
            return cf
    raise TypeError("Input is not a list of integers.")


def lcm(a: int, b: int) -> int:
    """
    returns signed lcm of two integers
    """
    if all(isinstance(i, int) for i in [a, b]):
        return a * b // gcf([a, b])
    raise TypeError("Input is not a pair of integers.")


@dataclasses.dataclass
class Fraction:
    """
    a real number consisiting of an integer numerator divded by an
    integer denominator
    """

    def __init__(self, numerator: int, denominator: int):
        if denominator == 0:
            raise BaseException("Invalid case: zero denominator.")
        self.numerator = numerator
        self.denominator = denominator
        # convert rational number to real number
        # return int if possible, otherwise float
        if numerator % denominator == 0:
            real = numerator // denominator
        else:
            real = numerator / denominator
        self.real = real
        self.__post_init__()

    def __post_init__(self):
        if self.denominator == 1:
            # already simplified, do nothing
            return
        if self.numerator == 0:
            # the denominator must be nonzero at this point
            # if the numerator is zero, assign the zero instance
            object.__setattr__(self, "denominator", 1)
            return
        if self.denominator < 0:
            # move negative sign from denominator
            object.__setattr__(self, "numerator", -self.numerator)
            object.__setattr__(self, "denominator", abs(self.denominator))
        # reduce fraction if possible
        factor = gcf([self.numerator, self.denominator])
        if factor > 1:  # reduce fraction if possible
            object.__setattr__(self, "numerator", self.numerator // factor)
            object.__setattr__(self, "denominator", self.denominator // factor)

    def __str__(self):
        if self.numerator == 1 and self.denominator == 1:
            return "1"
        elif self.numerator == -1 and self.denominator == 1:
            return "-1"
        else:
            return str(f"{self.numerator}/{self.denominator}")

    def __repr__(self):
        return str(self)

    @classmethod
    def zero(cls):
        return cls(0, 1)

    @classmethod
    def one(cls):
        return cls(1, 1)

    def __int__(self):
        if isinstance(self.real, int):
            return self.real
        raise ValueError(f"{str(self)} cannot be converted to type int")

    def __float__(self):
        return float(self.real)

    def __eq__(self, other):
        if isinstance(other, Fraction):
            return (self.numerator, self.denominator) == (
                other.numerator,
                other.denominator,
            )
        if isinstance(other, int) or isinstance(other, float):
            return self.real == other

    def __gt__(self, other):
        if isinstance(other, Fraction):
            return self.real > other.real
        if isinstance(other, int) or isinstance(other, float):
            return self.real > other

    def __lt__(self, other):
        if isinstance(other, Fraction):
            return self.real < other.real
        if isinstance(other, int) or isinstance(other, float):
            return self.real < other

    def __ge__(self, other):
        if isinstance(other, Fraction):
            return self.real >= other.real
        if isinstance(other, int) or isinstance(other, float):
            return self.real >= other

    def __le__(self, other):
        if isinstance(other, Fraction):
            return self.real <= other.real
        if isinstance(other, int) or isinstance(other, float):
            return self.real <= other

    def __abs__(self):
        return self.__class__(abs(self.numerator), abs(self.denominator))

    def __add__(self, other):
        if isinstance(other, Fraction):
            denominator = lcm(self.denominator, other.denominator)
            numerator = (self.numerator * (denominator // self.denominator)) + (
                other.numerator * (denominator // other.denominator)
            )
            return self.__class__(numerator, denominator)
        if isinstance(other, int) or isinstance(other, float):
            return self.real + other

    def __neg__(self):
        return self.__class__(-self.numerator, self.denominator)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, Fraction):
            return self.__class__(
                self.numerator * other.numerator,
                self.denominator * other.denominator,
            )
        if isinstance(other, int):
            return self.__class__(other * self.numerator, self.denominator)
        if isinstance(other, float):
            return self.real * other
        NotImplementedError(
            f"Undefined multiplication between types {self.__class__.__name__}"
            f" and {other.__class__.__name__}."
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Fraction):
            return self.__class__(
                self.numerator * other.denominator,
                self.denominator * other.numerator,
            )
        if isinstance(other, int):
            return self.__class__(self.numerator, other * self.denominator)
        if isinstance(other, float):
            return self.real / other
        NotImplementedError(
            f"Undefined division between types {self.__class__.__name__}"
            f" and {other.__class__.__name__}."
        )


@dataclasses.dataclass
class LinearCombination:
    """
    class that describes linear combinations of terms
    a u_0 + b u_1 + c u_2 + ...
    as a dictionary
    {0: a, 1: b, 2: c, ...}
    enables addition and subtraction between linear combinations
    """

    coeffs: dict[int:float]

    def __post_init__(self):
        """
        linear combinations should be sorted and should not contain
        coefficients of 0 unless they are the zero instance {0: 0}
        """
        if self.coeffs == {0: 0}:
            # self is the zero instance, do nothing
            return
        if self.coeffs == {} or all(j == 0 for j in self.coeffs.values()):
            # if coeffs is empty or if all its values are zero and 0 is not
            # the only index
            object.__setattr__(self, "coeffs", {0: 0})
        else:  # coeffs is nonempty and contains at least one nonzero item
            # sort by degree
            sorted_coeffs = dict(sorted(self.coeffs.items()))
            # remove 0 coefficients if they unless 0x^0 is the only term
            new_coeffs = {}
            for deg, coeff in sorted_coeffs.items():
                if coeff != 0:
                    new_coeffs[deg] = sorted_coeffs[deg]
            object.__setattr__(self, "coeffs", new_coeffs)

    def __str__(self):
        strings = [f"{coeff} u_{ind}" for (ind, coeff) in self.coeffs.items()]
        return " + ".join(strings) if strings else str(self.zero())

    def __repr__(self):
        return str(self)

    @classmethod
    def zero(cls):
        return cls({0: 0})

    def __add__(self, other):
        coeffs_sum = {}
        for i in self.coeffs.keys():
            if i in other.coeffs.keys():
                coeffs_sum[i] = self.coeffs[i] + other.coeffs[i]
            else:
                coeffs_sum[i] = self.coeffs[i]
        for i in other.coeffs.keys():
            if i not in self.coeffs.keys():
                coeffs_sum[i] = other.coeffs[i]
        return self.__class__(coeffs_sum)

    def __neg__(self):
        return self.__class__(dict([(i, -j) for i, j in self.coeffs.items()]))

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(
                dict([(i, other * j) for i, j in self.coeffs.items()])
            )
        raise TypeError(
            f"Cannot multiply type {self.__class__.__name__} with type"
            f" {other.__class__.__name__}."
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.__class__(
                dict([(i, j / other) for i, j in self.coeffs.items()])
            )
        raise TypeError(
            f"Cannot divide type {self.__class__.__name__} by type"
            f" {other.__class__.__name__}."
        )


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
