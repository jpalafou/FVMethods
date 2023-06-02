from abc import abstractmethod
import dataclasses
import numpy as np
import csv
import os.path
from finite_volume.mathematiques import (
    lcm,
    Fraction,
    LinearCombinationOfFractions,
    LinearCombination,
)
from finite_volume.polynome import Lagrange


stensil_path = "stensils/"


class Kernel:
    """
    provide information about a finite volume scheme based based on a kernel
    definition
    """

    def __init__(self, left: int, right: int, adj_index_at_center: int = 0):
        """
        left:   number of cells left of center in the kernel
        right:  number of cells right of center
        u_index_at_center:      what is the subscript of u at the central cell
        x_cell_centers and x_cell_faces are lists of integers
        """
        self.index_at_center = left  # index of the center cell
        self.size = left + right + 1  # number of cells in the kernel
        self.adj_index_at_center = adj_index_at_center

        # kernel step size
        h = 2
        self.h = h  # must be an even integer

        # generate an array of x-values for the cell centers
        x_cell_centers = list(range(-h * left, h * (right + 1), h))
        self.x_cell_centers = x_cell_centers

        # generate an array of x-values for the cell faces
        x_cell_faces = [i - h // 2 for i in x_cell_centers]
        x_cell_faces.append(x_cell_centers[-1] + h // 2)
        self.x_cell_faces = x_cell_faces

        # true indices
        self.indices = [
            i - self.index_at_center + adj_index_at_center for i in range(self.size)
        ]

    def __str__(self):
        string = "|"
        for i in range(self.size):
            string = string + " " + str(self.indices[i]) + " |"
        return string


@dataclasses.dataclass
class Stensil(LinearCombinationOfFractions):
    """
    Parent class for stensils which in general can be
        read from csv
        written to csv
        converted to an np array
        constructed from a kernel
        constructed from a specified order
    """

    @classmethod
    def read_from_csv(cls, path: str):
        """
        args:
            path    str, location of csv
        returns:
            Stensil instance
        """
        coeffs = {}
        with open(path, mode="r") as the_file:
            for row in csv.reader(the_file):
                if len(row) == 2:  # float weights
                    coeffs[int(row[0])] = float(row[1])
                elif len(row) == 3:  # fraction weights
                    coeffs[int(row[0])] = Fraction(int(row[1]), int(row[2]))
        return cls(coeffs)

    def write_to_csv(self, path: str):
        """
        args:
            path    str, location of csv
        returns:
            writes stensil instance as a csv to path
        """
        with open(path, "w+") as the_file:
            writer = csv.writer(the_file)
            for key, val in self.coeffs.items():
                if isinstance(val, int) or isinstance(val, float):
                    writer.writerow([key, val])
                elif isinstance(val, Fraction):
                    writer.writerow([key, val.numerator, val.denominator])

    def nparray(self):
        """
        convert a reconstruction scheme to an array of weights
        """
        if all(isinstance(i, Fraction) for i in self.coeffs.values()):
            denoms = [frac.denominator for frac in self.coeffs.values()]
            denom_lcm = 1
            for i in denoms:
                denom_lcm = lcm(denom_lcm, i)
            mylist = []
            for i in range(min(self.coeffs.keys()), max(self.coeffs.keys()) + 1):
                if i in self.coeffs.keys():
                    mylist.append(
                        self.coeffs[i].numerator
                        * denom_lcm
                        // self.coeffs[i].denominator
                    )
                else:
                    mylist.append(0)
            return np.array(mylist)
        elif all(isinstance(i, float) for i in self.coeffs.values()):
            return np.array([i for i in self.coeffs.values()])

    @classmethod
    @abstractmethod
    def construct_from_kernel(cls, kernel: Kernel):
        ...

    @classmethod
    @abstractmethod
    def construct_from_order(cls, order: int):
        ...


@dataclasses.dataclass
class ConservativeInterpolation(Stensil):
    """
    find the polynomial reconstruction evaluated at a point from a kernel of
    cell averages which conserves u inside the kernel
    """

    @classmethod
    def construct_from_kernel(cls, kernel: Kernel, reconstruct_here: str = "right"):
        """
        generate a stensil to evaluate a kernel at x = reconstruct_here
        if reconstruct_here = left, right, or center, the stensil will
        be in terms of integer fractions. otherwise, floating point
        values will be used for weights.
        """
        x = kernel.x_cell_faces
        if reconstruct_here == "right" or reconstruct_here == "r":
            x_eval = x[kernel.index_at_center + 1]
        elif reconstruct_here == "left" or reconstruct_here == "l":
            x_eval = x[kernel.index_at_center]
        elif reconstruct_here == "center" or reconstruct_here == "c":
            x_eval = kernel.x_cell_centers[kernel.index_at_center]
        elif isinstance(reconstruct_here, float):
            if reconstruct_here < -0.5 or reconstruct_here > 0.5:
                BaseException("Enter an x value to evaluate between -0.5 and 0.5.")
            x_eval = kernel.h * reconstruct_here
        else:
            BaseException("Must provide an x value for polynomial reconstruction.")

        # find the polynomial expression being multiplied to each cell value
        polynomial_weights = {}

        # skip first cell wall (coming from the left) because the cumulative
        # quantity is 0 there
        for i in range(1, len(kernel.x_cell_faces)):
            for j in kernel.indices[:i]:
                if j in polynomial_weights.keys():
                    polynomial_weights[j] = polynomial_weights[j] + Lagrange.Lagrange_i(
                        kernel.x_cell_faces, i
                    )
                else:
                    polynomial_weights[j] = Lagrange.Lagrange_i(kernel.x_cell_faces, i)

        # take the derivative of the polynomials
        polynomial_weights_prime = dict(
            [(i, polynome.prime()) for i, polynome in polynomial_weights.items()]
        )

        # evaluate them at the cell face, multiply by h
        if isinstance(x_eval, int):
            coeffs = (
                kernel.h
                * LinearCombinationOfFractions(
                    dict(
                        [
                            (i, polynome.eval(x_eval, div="fraction"))
                            for i, polynome in polynomial_weights_prime.items()
                        ]
                    )
                )
            ).coeffs
        elif isinstance(x_eval, float):
            coeffs = (
                kernel.h
                * LinearCombination(
                    dict(
                        [
                            (i, polynome.eval(x_eval, div="true"))
                            for i, polynome in polynomial_weights_prime.items()
                        ]
                    )
                )
            ).coeffs
        return cls(coeffs)

    @classmethod
    def construct_from_order(
        cls,
        order: int = 1,
        reconstruct_here: str = "right",
    ):
        """
        generate a stensil to evaluate a kernel at x = reconstruct_here
        from a given order of accuracy. asymmetric kernels will be
        averaged with their inverses to form a symmetric kernel
        """
        if reconstruct_here == "r" or reconstruct_here == 0.5:
            reconstruct_here = "right"
        if reconstruct_here == "l" or reconstruct_here == -0.5:
            reconstruct_here = "left"
        if reconstruct_here == "c" or reconstruct_here == 0:
            reconstruct_here = "center"
        save_path = (
            stensil_path + "conservative/" + f"order{order}_{reconstruct_here}.csv"
        )
        if os.path.isfile(save_path):
            interface_scheme = cls.read_from_csv(save_path)
        else:
            if order % 2 != 0:  # odd order
                kern = Kernel(order // 2, order // 2)
                interface_scheme = cls.construct_from_kernel(kern, reconstruct_here)
            else:  # even order
                long_length = order // 2  # long length
                short_length = order // 2 - 1  # short length
                interface_scheme = (
                    cls.construct_from_kernel(
                        Kernel(long_length, short_length), reconstruct_here
                    )
                    + cls.construct_from_kernel(
                        Kernel(short_length, long_length), reconstruct_here
                    )
                ) / 2
            interface_scheme.write_to_csv(save_path)
            if isinstance(reconstruct_here, str):
                print(
                    f"Wrote a {reconstruct_here} interface reconstruction "
                    f"scheme of order {order} to {save_path}\n"
                )
            else:
                print(
                    f"Wrote a reconstruction scheme at x = {reconstruct_here} "
                    f"of order {order} to {save_path}\n"
                )
        return interface_scheme
