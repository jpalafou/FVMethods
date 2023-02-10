from util.integrate import Integrator
import numpy as np
import matplotlib.pyplot as plt


def run2d():
    class Advection2D_order1(Integrator):
        def __init__(self, h, T, a, b):
            self.h = h
            self.a = a
            self.b = b
            self.Dt = 0.5 * h / max(a, b)
            self.t = np.arange(0, T + self.Dt, self.Dt)
            self.x = np.arange(0, 1 + h, h)
            self.y = np.arange(0, 1 + h, h)
            self.u0 = np.array(
                [
                    [
                        np.exp(-20 * ((i - 0.5) ** 2 + (j - 0.5) ** 2))
                        for i in self.y
                    ]
                    for j in self.x
                ]
            )
            super().__init__(self.u0, self.t)

        def central_diff(self, left, right):
            return (right - left) / (2 * self.h)

        def forward_diff(self, center, right):
            return (right - center) / self.h

        def udot(self, u, t_i: float):
            # apply boundaries
            n = len(self.x)
            dudt = np.zeros(u.shape)
            # left, right
            u_extended = np.concatenate(
                (np.zeros(n).reshape(n, 1), u, np.zeros(n).reshape(n, 1)),
                axis=1,
            )
            # up, down
            u_extended = np.concatenate(
                (
                    np.zeros(n + 2).reshape(1, n + 2),
                    u_extended,
                    np.zeros(n + 2).reshape(1, n + 2),
                )
            )
            # periodic box
            u_extended[:, 0] = u_extended[:, -2]
            u_extended[:, -1] = u_extended[:, 1]
            u_extended[0, :] = u_extended[-2, :]
            u_extended[-1, :] = u_extended[1, :]

            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    dudx = self.central_diff(
                        u_extended[i, j - 1], u_extended[i, j + 1]
                    )
                    dudy = self.central_diff(
                        u_extended[i + 1, j], u_extended[i - 1, j]
                    )
                    # dudx = self.forward_diff(
                    #     u_extended[i, j], u_extended[i, j + 1]
                    # )
                    # dudy = self.forward_diff(
                    #     u_extended[i, j], u_extended[i - 1, j]
                    # )
                    dudt[i - 1, j - 1] = -self.a * dudx - self.b * dudy

            return dudt

    solution2D = Advection2D_order1(0.02, 1, 1, 1)
    solution2D.rk2()
    return solution2D


solution2D = run2d()

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(solution2D.u[-1] - solution2D.u[0])
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5)
fig.tight_layout()
plt.show()
