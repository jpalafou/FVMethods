# advection solution schems
import numpy as np
import matplotlib.pyplot as plt
from util.initial_condition import initial_condition2d
from util.integrate import Integrator, rk4_Dt_adjust
from util.fvscheme import ConservativeInterpolation
from util.david.simple_trouble_detection import trouble_detection


class AdvectionSolver(Integrator):
    """
    uses polynomial reconstruction of arbitrary order
    no limiter
    """

    def __init__(
        self,
        u0: np.ndarray = None,
        u0_preset: str = "square",
        n: tuple = (32, 32),
        x: tuple = (0, 1),
        y: tuple = (0, 1),
        T: float = 2,
        a: tuple = (1, 1),
        courant: float = 0.5,
        order: int = 1,
        bc_type: str = "periodic",
        posteriori: bool = False,
    ):
        self.a = a[0]
        self.b = a[1]
        self.order = order

        # spatial discretization
        if isinstance(n, int):
            self.nx = n
            self.ny = n
        elif isinstance(n, tuple):
            self.nx = n[0]
            self.ny = n[1]
        self.x_interface = np.linspace(x[0], x[1], num=self.nx + 1)
        self.x = 0.5 * (
            self.x_interface[:-1] + self.x_interface[1:]
        )  # x at cell centers
        self.hx = (x[1] - x[0]) / self.nx
        self.y_interface = np.linspace(y[0], y[1], num=self.ny + 1)
        self.y = 0.5 * (
            self.y_interface[:-1] + self.y_interface[1:]
        )  # y at cell centers
        self.hy = (x[1] - x[0]) / self.ny

        # time discretization
        self.courant = courant
        if self.a:  # nonzero horizontal advection velocity
            Dtx = courant * self.hx / np.abs(self.a)
        else:
            Dtx = courant * self.hx
        if self.b:  # nonzero vertical advection velocity
            Dty = courant * self.hy / np.abs(self.b)
        else:
            Dty = courant * self.hy
        Dt = min(Dtx, Dty)
        time_step_adjustment = 1
        if order > 4:
            time_step_adjustmentx = rk4_Dt_adjust(self.hx, x[1] - x[0], order)
            time_step_adjustmenty = rk4_Dt_adjust(self.hy, y[1] - y[0], order)
            time_step_adjustment = min(
                time_step_adjustmentx, time_step_adjustmenty
            )
            print(
                "Decreasing timestep by a factor of"
                f" {time_step_adjustment} to maintain order {order} with rk4"
            )
            Dt = Dt * time_step_adjustment
        n_time = int(np.ceil(T / Dt))
        self.Dt = T / n_time
        self.time_step_adjustment = time_step_adjustment
        self.t = np.linspace(0, T, num=n_time)

        # initial condition
        if not u0:
            self.u0 = initial_condition2d(self.x, self.y, u0_preset)
        else:
            if u0.shape != (self.ny, self.nx):
                raise BaseException(f"Invalid initial condition for size {n}")
            self.u0 = u0

        # initialize integrator
        super().__init__(self.u0, self.t)

        # stensil for reconstructed values at cell interfaces
        right_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "right")
        )
        left_interface_stensil_original = (
            ConservativeInterpolation.construct_from_order(order, "left")
        )
        self.right_interface_stensil = (
            right_interface_stensil_original.nparray().reshape(-1, 1, 1)
        )
        self.left_interface_stensil = (
            left_interface_stensil_original.nparray().reshape(-1, 1, 1)
        )
        assert sum(self.left_interface_stensil) == sum(
            self.right_interface_stensil
        )
        self._stensil_sum = sum(self.left_interface_stensil)
        # number of kernel cells from center referenced by a scheme
        # symmetry is assumed
        self._k = max(right_interface_stensil_original.coeffs.keys())
        # number of ghost cells on either side of the extended state vector
        self._gw = self._k + 1
        self.bc_type = bc_type  # type of boudnary condition
        # trouble detection
        self.trouble = np.zeros(self.u0.shape)  # troubled cells

    def apply_bc(
        self, without_ghost_cells: np.ndarray, num_ghost_cells: int
    ) -> np.ndarray:
        # construct an extended array
        ny, nx = without_ghost_cells.shape
        with_ghost_cells = np.zeros(
            (ny + 2 * num_ghost_cells, nx + 2 * num_ghost_cells)
        )
        gw = num_ghost_cells
        negative_gw = -gw
        with_ghost_cells[gw:negative_gw, gw:negative_gw] = without_ghost_cells
        if self.bc_type == "periodic":
            left_index = -2 * gw
            right_index = 2 * gw
            # left/right
            with_ghost_cells[:, :gw] = with_ghost_cells[
                :, left_index:negative_gw
            ]
            with_ghost_cells[:, negative_gw:] = with_ghost_cells[
                :, gw:right_index
            ]
            # up/down
            with_ghost_cells[:gw, :] = with_ghost_cells[
                left_index:negative_gw, :
            ]
            with_ghost_cells[negative_gw:, :] = with_ghost_cells[
                gw:right_index, :
            ]
            return with_ghost_cells

    def reimann(
        self,
        a_at_boundary,
        value_to_the_left_of_boundary: float,
        value_to_the_right_of_boundary: float,
    ) -> float:
        if a_at_boundary > 0:
            return value_to_the_left_of_boundary
        elif a_at_boundary < 0:
            return value_to_the_right_of_boundary
        else:
            return 0

    def udot(self, u: np.ndarray, t_i: float) -> np.ndarray:
        # construct an extended array with ghost zones and apply bc
        ny, nx = u.shape
        uwithghosts = self.apply_bc(u, self._gw)
        # construct an array of staggered state arrays such that a 3D
        # matrix operation with a stensil multiplies weights by their
        # referenced cells
        # north/south east/west
        ns = []
        ew = []
        stensil_length = 2 * self._k + 1
        for i in range(stensil_length):
            top_ind = i + ny + 2
            right_ind = i + nx + 2
            ns.append(uwithghosts[i:top_ind, :])
            ew.append(uwithghosts[:, i:right_ind])
        NS = np.asarray(ns)
        EW = np.asarray(ew)
        ubar_north = sum(NS * self.left_interface_stensil) / self._stensil_sum
        ubar_south = sum(NS * self.right_interface_stensil) / self._stensil_sum
        ubar_east = sum(EW * self.right_interface_stensil) / self._stensil_sum
        ubar_west = sum(EW * self.left_interface_stensil) / self._stensil_sum
        # evaluation arrays have excess ghost information and must be
        # chopped
        chop = self._gw - 1
        anti_chop = -chop
        if chop > 0:
            ubar_north = ubar_north[:, chop:anti_chop]
            ubar_south = ubar_south[:, chop:anti_chop]
            ubar_east = ubar_east[chop:anti_chop, :]
            ubar_west = ubar_west[chop:anti_chop, :]
        # initialize empty difference arrays
        Delta_ubar_NS = np.zeros((ny, nx))
        Delta_ubar_EW = np.zeros((ny, nx))
        for i in range(1, ny + 1):
            for j in range(1, nx + 1):
                Delta_ubar_NS[i - 1, j - 1] = self.reimann(
                    self.b,
                    ubar_north[i, j],
                    ubar_south[i - 1, j],
                ) - self.reimann(
                    self.b,
                    ubar_north[i + 1, j],
                    ubar_south[i, j],
                )
                Delta_ubar_EW[i - 1, j - 1] = self.reimann(
                    self.a,
                    ubar_east[i, j],
                    ubar_west[i, j + 1],
                ) - self.reimann(
                    self.a,
                    ubar_east[i, j - 1],
                    ubar_west[i, j],
                )
        return (
            -(self.a / self.hx) * Delta_ubar_EW
            - (self.b / self.hy) * Delta_ubar_NS
        )

    def posteriori_revision(self, previous_solution, candidate_solution):
        """
        u1 is now the candidate solution. check if it is (what are we
        checking?) and apply fallback scheme if necessary
        """
        # david usesa gw of 2
        ny, nx = candidate_solution.shape
        previous_solution = self.apply_bc(
            without_ghost_cells=self.u0, num_ghost_cells=2
        )
        previous_solution = previous_solution.reshape(1, ny, nx)
        candidate_solution = self.apply_bc(
            without_ghost_cells=self.u1, num_ghost_cells=2
        )
        candidate_solution = candidate_solution.reshape(1, ny, nx)

        trouble_detection(self.trouble, previous_solution, candidate_solution)

        print(self.trouble)

    def plot_error(self):
        fig, ((ax0, ax1, ax2)) = plt.subplots(1, 3, figsize=(12, 10))
        im0 = ax0.imshow(
            self.u[0], extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]]
        )
        ax0.figure.colorbar(im0, ax=ax0, shrink=0.5)
        ax0.set_title("initial condition")
        im1 = ax1.imshow(
            self.u[-1], extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]]
        )
        ax1.figure.colorbar(im1, ax=ax1, shrink=0.5)
        ax1.set_title("final time step")
        im2 = ax2.imshow(
            self.u[-1] - self.u[0],
            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
        )
        ax2.figure.colorbar(im2, ax=ax2, shrink=0.5)
        ax2.set_title("initial condition - final time step")
        fig.tight_layout()
        plt.show()
