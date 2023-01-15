import dataclasses


def fact(n):
    """
    factorial n! of an integer n
    """
    if not isinstance(n, int):
        raise TypeError("fact() only accept integers")
    if n < 0:
        raise BaseException("fact() is not definted for negative integers")
    product = 1
    for i in range(1, n + 1):
        product *= i
    return product


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

    numerator: int
    denominator: int

    def __post_init__(self):
        if not isinstance(self.numerator, int) or not isinstance(
            self.denominator, int
        ):
            raise TypeError("Input is not an int tuple.")
        if self.denominator == 0:
            raise BaseException("Invalid case: zero denominator.")
        if (self.numerator, self.denominator) == (
            0,
            1,
        ):  # zero instance, do nothing
            pass
        elif self.numerator == 0 and self.denominator != 1:
            # if the numerator is zero, assign the zero instance
            object.__setattr__(self, "numerator", 0)
            object.__setattr__(self, "denominator", 1)
        else:  # reduce fraction if possible
            factor = gcf([self.numerator, self.denominator])
            if factor > 1:
                object.__setattr__(self, "numerator", self.numerator // factor)
                object.__setattr__(
                    self, "denominator", self.denominator // factor
                )
            # move negative sign from denominator
            if self.denominator < 0:
                object.__setattr__(self, "numerator", -self.numerator)
                object.__setattr__(self, "denominator", abs(self.denominator))

    def __str__(self):
        if self.numerator == 1 and self.denominator == 1:
            return "1"
        elif self.numerator == -1 and self.denominator == 1:
            return "-1"
        else:
            return str(f"{self.numerator}/{self.denominator}")

    @classmethod
    def zero(cls):
        return cls(0, 1)

    def __add__(self, other):
        denominator = lcm(self.denominator, other.denominator)
        numerator = (self.numerator * (denominator // self.denominator)) + (
            other.numerator * (denominator // other.denominator)
        )
        return self.__class__(numerator, denominator)

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
        elif isinstance(other, int):
            return self.__class__(other * self.numerator, self.denominator)
        TypeError(
            f"Illegal multiplication between types {self.__class__.__name__}"
            f" and {other.__class__.__name__}."
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Fraction):
            return self.__class__(
                self.numerator * other.denominator,
                self.denominator * other.numerator,
            )
        elif isinstance(other, int):
            return self.__class__(self.numerator, other * self.denominator)
        else:
            TypeError(
                f"Illegal division between types {self.__class__.__name__}"
                f" and {other.__class__.__name__}."
            )
