import dataclasses
import numpy as np
from util.mathbasic import fact, gcf, lcm, Fraction
from util.lincom import LinearCombination


@dataclasses.dataclass
class Polynome(LinearCombination):
    """
    express a polynomial
    a*x^{n} + b*x^{n-1} + c*x^{n-2} + ...
    as a dictionary
    {n: a, n-1: b, n-2: c, ...}
    """

    coeffs: dict[int:int]

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
            f"Cannot multiply a {self.__class__.__name__} with"
            + f"a {type(other)}"
        )

    __rmul__ = __mul__

    def __floordiv__(self, other: int):
        if isinstance(other, int):
            if other == 0:
                raise BaseException(
                    f"Cannot divide a {self.__class__.__name__} by 0."
                )
            return self.__class__(
                dict([(i, j // other) for i, j in self.coeffs.items()])
            )
        else:
            raise TypeError(
                f"Cannot divide a {self.__class__.__name__} by a {type(other)}"
            )

    def prime(self):
        """
        returns the first derivative of a polynomial
        """
        derivative_coeffs = {}
        for deg, coeff in self.coeffs.items():
            if deg != 0:
                derivative_coeffs[deg - 1] = deg * coeff
        return self.__class__(derivative_coeffs)

    def eval(self, x: float) -> float:
        """
        returns p(x) as an int/float
        """
        return sum([coeff * x**deg for deg, coeff in self.coeffs.items()])

    @classmethod
    def legendre(cls, n):
        """
        generate the nth Legendre polynomial
        https://mathworld.wolfram.com/LegendrePolynomial.html
        """
        square_dif = Polynome({2: 1, 0: -1})  # x^2 - 1
        derive_me = Polynome({0: 1})  # 1
        for _ in range(n):
            derive_me = derive_me * square_dif
        for _ in range(n):
            derive_me = derive_me.prime()
        denom = 2**n * fact(n)
        return (1 / denom) * derive_me

    def find_zeros(
        self, x_bounds: list[float], epsilon: float = 1e-8
    ) -> list[float]:
        """
        generate a list of approximate zeros for a polynomial within a given
        range.
        WARNING: does NOT work for zeros with even multiplicity
        """
        if len(x_bounds) != 2:
            raise BaseException("Two x boundaries are required")
        if x_bounds[0] > x_bounds[1]:
            raise BaseException("x values should be in increasing order")
        h = (x_bounds[1] - x_bounds[0]) / 1000  # scanning resolution
        x_scan = np.arange(
            x_bounds[0], x_bounds[1] + h, h
        )  # x values to check for sign changes
        list_of_zeros = []
        # don't forget to check the left bound
        if self.eval(x_bounds[0]) == 0:
            list_of_zeros.append(x_bounds[0])
        # the for loop starts at x_bounds[0] + h
        for i in range(1, len(x_scan)):
            if self.eval(x_scan[i]) == 0:  # is the current x value a zero?
                list_of_zeros.append(x_scan[i])
            elif (
                self.eval(x_scan[i]) * self.eval(x_scan[i - 1]) < 0
            ):  # sign change indicates we skipped over a zero
                # find a zero in a range assuming the zero exists
                # use a simple midpoint approximation
                a = x_scan[i - 1]
                b = x_scan[i]
                middle = (1 / 2) * (a + b)
                while abs(self.eval(middle)) > epsilon:
                    # using same sign change principle
                    if self.eval(a) * self.eval(middle) < 0:
                        b = middle
                    else:
                        a = middle
                    middle = (1 / 2) * (a + b)
                list_of_zeros.append(middle)
        return list_of_zeros


@dataclasses.dataclass
class Lagrange(Polynome):
    """
    polynomial numerator with an integer denominator
    allow addition/subtraction between Lagrange polynomials
    """

    coeffs: dict[int:int]
    denominator: int

    def __post_init__(self):
        """
        denominator should not be negative or zero. gcf of numerator and
        denominator should be factored out of both when applicable
        """

        # make sure there is a new_numerator
        self.numerator = Polynome(self.coeffs)

        # check if denominator type is correct
        if not isinstance(self.denominator, int):
            raise BaseException("Invalid denominator type.")
        # denominator cannot be zero
        if self.denominator == 0:
            raise BaseException("Lagrange instance with 0 denominator.")
        # reformat lagrange
        if (
            self.denominator != 1
            and self.numerator != self.numerator.__class__.zero()
        ):
            # redistribute negative sign if denominator is negative
            if self.denominator < 0:
                new_numerator = -self.numerator
                new_denominator = abs(self.denominator)
            else:
                new_numerator = self.numerator
                new_denominator = self.denominator
            # factor gcf out of numerator and denominator if it is > 1
            gcf_fraction = gcf(
                list(new_numerator.coeffs.values()) + [new_denominator]
            )
            if gcf_fraction > 1:
                new_numerator = new_numerator // gcf_fraction
                new_denominator = new_denominator // gcf_fraction
            assert new_denominator > 0
            new_coeffs = new_numerator.coeffs
            object.__setattr__(self, "numerator", new_numerator)
            object.__setattr__(self, "coeffs", new_coeffs)
            object.__setattr__(self, "denominator", new_denominator)

    def __str__(self):
        return f"({self.numerator})/{self.denominator}"

    @classmethod
    def Lagrange_i(cls, x_values: list[int], i: int):
        """
        find the ith Lagrange polynomial from a set of x points
        """
        numerator = Polynome.one()
        denominator = 1
        for j in range(len(x_values)):
            if j != i:
                numerator *= Polynome({1: 1, 0: -x_values[j]})
                denominator *= x_values[i] - x_values[j]
        return cls(numerator.coeffs, denominator)

    def zero(self):
        return self.__class__(
            self.numerator.__class__.zero().coeffs, self.denominator
        )

    def __add__(self, other):
        denominator = lcm(self.denominator, other.denominator)
        numerator = (self.numerator * (denominator // self.denominator)) + (
            other.numerator * (denominator // other.denominator)
        )
        return self.__class__(numerator.coeffs, denominator)

    def __neg__(self):
        return self.__class__((-self.numerator).coeffs, self.denominator)

    def __sub__(self, other):
        return self + -other

    def prime(self):
        return self.__class__(self.numerator.prime().coeffs, self.denominator)

    def eval(self, x: float, div: str = "true") -> float:
        """
        returns the result of the polynomial fraction evaluated at x
        """
        if div == "true":
            return self.numerator.eval(x) / self.denominator
        elif div == "floor":
            return self.numerator.eval(x) // self.denominator
        elif div == "fraction":
            return Fraction(self.numerator.eval(x), self.denominator)
        else:
            raise BaseException("Invalid division type.")
