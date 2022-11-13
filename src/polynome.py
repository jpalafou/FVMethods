import dataclasses
import numpy as np
from lincom import LinearCombination

# helper functions
def gcf(mylist):
    """
    returns gcf of an absolute value list as an int
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
    raise TypeError(f"Input is not a list of integers.")


def lcm(a, b):
    """
    returns signed lcm of two integers a and b as an int
    """
    if all(isinstance(i, int) for i in [a, b]):
        return a * b // gcf([a, b])
    raise TypeError(f"Input is not a pair of integers.")


@dataclasses.dataclass
class Polynome(LinearCombination):
    """
    express a polynomial
    a*x^{n} + b*x^{n-1} + c*x^{n-2} + ...
    as a dictionary
    {n: a, n-1: b, n-2: c, ...}
    """

    coeffs: dict

    def __post_init__(self):
        """
        Polynomial dictionaries should be reverse sorted and should not contain
        coefficients of 0 unless they are the zero polynomial {0: 0}
        """
        # if coeffs is empty or if all its values are zero and 0 is not the only
        # index
        if self.coeffs == {} or (all(j == 0 for j in self.coeffs.values()) and \
        list(self.coeffs.keys()) != [0]):
            object.__setattr__(self, "coeffs", {0: 0})
        else: # coeffs is nonempty and contains at least one nonzero item
            # sort by degree
            sorted_coeffs = dict(sorted(self.coeffs.items(), reverse=True))
            # remove 0 coefficients if they unless 0x^0 is the only term
            new_coeffs = {}
            for deg, coeff in sorted_coeffs.items():
                if coeff != 0 or (deg == 0 and len(new_coeffs) == 0):
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

    def one():
        return Polynome({0: 1})

    def __mul__(self, other):
        """
        Polynomial * Polynomial --> Polynomial
        or
        Polynomial * int --> Polynomial
        """
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
        elif isinstance(other, int):
            return self.__class__(dict([(i, other * j) for i, j in \
            self.coeffs.items()]))
        raise TypeError(f"Cannot multiply a {self.__class__.__name__} with" + \
        f"a {type(other)}")

    __rmul__ = __mul__

    def __floordiv__(self, other):
        """
        Polynomial // int --> Polynomial
        """
        quotient_coeffs = {}
        if isinstance(other, int):
            if other == 0:
                raise BaseException(f"Cannot divide a {self.__class__.__name__} by 0.")
            return self.__class__(dict([(i, j // other) for i, j in self.coeffs.items()]))
        else:
            raise TypeError(f"Cannot divide a {self.__class__.__name__} by a {type(other)}")

    def prime(self):
        """
        returns the first derivative of a polynomial
        """
        derivative_coeffs = {}
        for deg, coeff in self.coeffs.items():
            if deg != 0:
                derivative_coeffs[deg - 1] = deg * coeff
        return self.__class__(derivative_coeffs)

    def eval(self, x):
        """
        returns p(x) as an int/float
        """
        return sum([coeff * x**deg for deg, coeff in self.coeffs.items()])


@dataclasses.dataclass
class Lagrange:
    """
    Lagrange() := Polynome()/int
    allow addition/subtraction between Lagrange polynomials
    """
    numerator: Polynome
    denominator: int

    @classmethod
    def Lagrange_i(cls, x_values, i):
        """
        find the ith Lagrange polynomial from a set of x points
        depends on Polynome class
        """
        numerator = Polynome.one()
        denominator = 1
        for j in range(len(x_values)):
            if j != i:
                numerator *= Polynome({1: 1, 0: -x_values[j]})
                denominator *= x_values[i] - x_values[j]
        return cls(numerator, denominator)

    def __post_init__(self):
        """
        denominator should not be negative or zero. gcf of numerator and
        denominator should be factored out of both when applicable
        """
        # check if types are correct
        if not isinstance(self.numerator, Polynome) or \
        not isinstance(self.denominator, int):
            raise BaseException("Invalid numerator or denominator type. " + \
            "Did you mean to use Lagrange.Lagrange_i()?")
        # denominator cannot be zero
        if self.denominator == 0:
            raise BaseException('Lagrange instance with 0 denominator.')
        # reformat lagrange
        if self.denominator != 1 and self.numerator != \
        self.numerator.__class__.zero():
            # redistribute negative sign if denominator is negative
            if self.denominator < 0:
                new_numerator = -self.numerator
                new_denominator = abs(self.denominator)
            else:
                new_numerator = self.numerator
                new_denominator = self.denominator
            # factor gcf out of numerator and denominator if it is > 1
            gcf_fraction = gcf(list(new_numerator.coeffs.values()) + \
            [new_denominator])
            if gcf_fraction > 1:
                new_numerator = new_numerator // gcf_fraction
                new_denominator = new_denominator // gcf_fraction
            assert new_denominator > 0
            object.__setattr__(self, "numerator", new_numerator)
            object.__setattr__(self, "denominator", new_denominator)

    def __str__(self):
        return f"({self.numerator})/{self.denominator}"

    def zero(self):
        return self.__class__(self.numerator.__class__.zero(), self.denominator)

    def __add__(self, other):
        """
        Lagrange + Lagrange --> Lagrange
        """
        denominator = lcm(self.denominator, other.denominator)
        numerator = self.numerator * (denominator // self.denominator) \
        + other.numerator * (denominator // other.denominator)
        return self.__class__(numerator, denominator)

    def __neg__(self):
        """
        -Lagrange --> Lagrange
        """
        return self.__class__(-self.numerator, self.denominator)

    def __sub__(self, other):
        """
        Lagrange - Lagrange --> Lagrange
        """
        return self + -other

    def prime(self):
        """
        d/dx{Lagrange} --> Lagrange
        """
        return self.__class__(self.numerator.prime(), self.denominator)

    def eval(self, x, div = 'true'):
        """
        Lagrange(int/float) --> int/float
        """
        if div == 'true':
            return self.numerator.eval(x) / self.denominator
        elif div == 'floor':
            return self.numerator.eval(x) // self.denominator
        else:
            raise BaseException('Invalid division type.')

    def tuple_eval(self, x):
        """
        Lagrange(int) --> tuple (numerator int, denominator int)
        """
        quotient = (self.numerator.eval(x), self.denominator)
        if quotient[0] == 0:
            factor = 1
        else:
            factor = gcf(quotient)
        return (quotient[0] // factor, quotient[1] // factor)


a = LinearCombination({0: 5, -5: 10, 5: 100})
b = LinearCombination({0: 5, -5: 10, 5: 100})
print(a)
print(b)
print(a + b)
