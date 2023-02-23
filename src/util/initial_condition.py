import numpy as np
import math


def initial_condition1d(x, ic_type):
    # initial values of u
    if ic_type == "sinus":
        u0 = np.cos(2 * math.pi * x)
    elif ic_type == "heavi":
        u0 = np.heaviside(x - 0.5, 1)
    elif ic_type == "square":
        u0 = np.array(
            [np.heaviside(i - 0.25, 1) - np.heaviside(i - 0.75, 1) for i in x]
        )
    elif ic_type == "composite":
        u0 = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] >= 0.1 and x[i] <= 0.2:
                u0[i] = (
                    1
                    / 6
                    * (
                        np.exp(
                            -np.log(2)
                            / 36
                            / 0.0025**2
                            * (x[i] - 0.0025 - 0.15) ** 2
                        )
                        + np.exp(
                            -np.log(2)
                            / 36
                            / 0.0025**2
                            * (x[i] + 0.0025 - 0.15) ** 2
                        )
                        + 4
                        * np.exp(
                            -np.log(2) / 36 / 0.0025**2 * (x[i] - 0.15) ** 2
                        )
                    )
                )
            if x[i] >= 0.3 and x[i] <= 0.4:
                u0[i] = 0.75
            if x[i] >= 0.5 and x[i] <= 0.6:
                u0[i] = 1 - abs(20 * (x[i] - 0.55))
            if x[i] >= 0.7 and x[i] <= 0.8:
                u0[i] = (
                    1
                    / 6
                    * (
                        np.sqrt(max(1 - (20 * (x[i] - 0.75 - 0.0025)) ** 2, 0))
                        + np.sqrt(
                            max(1 - (20 * (x[i] - 0.75 + 0.0025)) ** 2, 0)
                        )
                        + 4 * np.sqrt(max(1 - (20 * (x[i] - 0.75)) ** 2, 0))
                    )
                )
    return u0


def initial_condition2d(x, y, ic_type):
    if ic_type == "sinus":
        return -np.array(
            [
                [np.cos(2 * math.pi * i) + np.cos(2 * math.pi * j) for i in x]
                for j in y
            ]
        )
    elif ic_type == "square":
        return np.array(
            [
                [
                    1
                    if (i > 0.25 and i < 0.75) and (j > 0.25 and j < 0.75)
                    else 0
                    for i in x
                ]
                for j in y
            ]
        )

