import dataclasses
from src.polynome import Lagrange


class Kernel:
    """
    provide information about a finite volume scheme based based on a kernel
    definition
    """

    def __init__(self, left, right, adj_index_at_center = 0):
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
        h = 2 # must be an even integer
        self.h = h

        # generate an array of x-values for the cell centers
        x_cell_centers = list(range(-h * left, h * (right + 1), h))
        self.x_cell_centers = x_cell_centers

        # generate an array of x-values for the cell faces
        x_cell_faces = [i - h // 2 for i in x_cell_centers]
        x_cell_faces.append(x_cell_centers[-1] + h // 2)
        self.x_cell_faces = x_cell_faces

        # true indices
        self.indices = [i - self.index_at_center + adj_index_at_center for i \
        in range(self.size)]

    def __str__(self):
        string = "|"
        for i in range(self.size):
            string = string + " " + str(self.indices[i]) + " |"
        return string


@dataclasses.dataclass
class Interpolation:
    """
    find the interpolation scheme of a kernel at the right or left face
    enable addition and subtraction of interface interpolation schemes
    """

    def __init__(self, kernel, face="right"):
        """
        kernel: object
        face:   which face of the central cell is the scheme evaluating
        """
        x = kernel.x_cell_faces
        if face == "right" or face == "r":
            x_eval = x[kernel.index_at_center + 1]
        elif face == "left" or face == "l":
            x_eval = x[kernel.index_at_center]
        else:
            fprintf("ERROR! No interface x-value provided.")
            return

        # find the polynomial expression being multiplied to each cell value
        polynomial_weights = {}

        # skip first cell wall (coming from the left) because the cumulative
        # quantity is 0 there
        for i in range(1, len(kern.x_cell_faces)):
            for j in kern.indices[:i]:
                if polynomial_weights[j]:
                    polynomial_weights[j] = polynomial_weights[j] + \
                    Lagrange.Lagrange_i(kern.x_cell_faces, i)
                else:
                    polynomial_weights[j] = \
                    Lagrange.Lagrange_i(kern.x_cell_faces, i)

        # take the derivative of the polynomials
        polynomial_weights_prime = dict([(i, polynome.prime()) for i, polynome \
        in polynomial_weights.items()])

        # evaluate them at the cell face, multiply by h
        weights = dict([(i, polynome.tuple_eval(x_eval)) for i, polynome in \
        polynomial_weights_prime])

        # the Polynome class should actually be a subclass of expressions that
        # are linear combinations of terms. the Lagrange class should be a
        # subclass of expressions that are linear combinations of terms divided
        # by a common integer. finally, Interpolation should be a subclass of
        # FractionLinearCombination

        # IntegerLinearCombination
        # FractionalLinearCombination
