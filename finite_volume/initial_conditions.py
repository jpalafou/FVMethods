"""
defines commonly used initial conditions
"""


import numpy as np


def generate_ic(type: str, x: np.ndarray, y: np.ndarray = None):
    """
    args:
        type    'sinus', 'square', etc
        x       1d np array (n, )
        y       1d np array (m, )
    returns:
        initial condition defined on xy mesh (m, n)
    """

    if type == "composite":
        return composite(x)
    if type == "-composite":
        return -composite(x)
    if type == "disk":
        return disk(x, y)
    if type == "gauss":
        return gauss(x, y)
    if type == "sinus":
        return sinus(x, y)
    if type == "square":
        return square(x, y)
    if type == "-square":
        return -square(x, y)
    if type == "disk plus hill":
        return disk_plus_hill(x, y)
    if type == "just the gauss":
        return just_the_gauss(x)
    if type == "four friends":
        return four_friends(x)
    if type == "composite only gauss":
        return composite_only_gauss(x)


def composite(x):
    # only defined for 1d
    u = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] >= 0.1 and x[i] <= 0.2:
            u[i] = (
                1
                / 6
                * (
                    np.exp(-np.log(2) / 36 / 0.0025**2 * (x[i] - 0.0025 - 0.15) ** 2)
                    + np.exp(
                        -np.log(2) / 36 / 0.0025**2 * (x[i] + 0.0025 - 0.15) ** 2
                    )
                    + 4 * np.exp(-np.log(2) / 36 / 0.0025**2 * (x[i] - 0.15) ** 2)
                )
            )
        if x[i] >= 0.3 and x[i] <= 0.4:
            u[i] = 0.75
        if x[i] >= 0.5 and x[i] <= 0.6:
            u[i] = 1 - abs(20 * (x[i] - 0.55))
        if x[i] >= 0.7 and x[i] <= 0.8:
            u[i] = (
                1
                / 6
                * (
                    np.sqrt(max(1 - (20 * (x[i] - 0.75 - 0.0025)) ** 2, 0))
                    + np.sqrt(max(1 - (20 * (x[i] - 0.75 + 0.0025)) ** 2, 0))
                    + 4 * np.sqrt(max(1 - (20 * (x[i] - 0.75)) ** 2, 0))
                )
            )
    return u


def disk(x, y):
    # only defined for 2d
    xx, yy = np.meshgrid(x, y)
    r = 0.3
    slotw = 0.05
    slotl = 0.5
    disp = 0.5
    disk_idx = xx**2 + (yy - disp) ** 2 < r**2
    slot_idx = np.logical_and(
        np.abs(xx) < slotw / 2,
        np.logical_and(disp - r + slotl > yy, yy > disp - r),
    )
    u = np.zeros(x.shape)
    u = np.where(np.logical_and(disk_idx, np.logical_not(slot_idx)), 1, u)
    return u


def disk_plus_hill(x, y):
    # only defined for 2d
    xx, yy = np.meshgrid(x, y)
    r = 0.3
    disp = -0.5
    rr = np.sqrt(xx**2 + (yy - disp) ** 2)
    hill = np.where(rr < r, 0.5 * np.cos((np.pi / r) * rr) + 0.5, 0)
    return disk(x, y) + hill


def sinus(x, y):
    if y is None:
        return np.cos(2 * np.pi * x)
    return np.sin(2 * np.pi * (x + y[:, np.newaxis]))


def square(x, y):
    if y is None:
        return np.heaviside(x - 0.25, 1.0) - np.heaviside(x - 0.75, 1.0)
    X, Y = np.meshgrid(x, y)
    xcondition = np.logical_and(X > 0.25, X < 0.75)
    ycondition = np.logical_and(Y > 0.25, Y < 0.75)
    return np.where(np.logical_and(xcondition, ycondition), 1.0, 0.0)


def gauss(x, y, sigma=1 / 10):
    center = 0.5
    if y is None:
        xc = x - center
        return np.exp(-(1 / 2) * (xc / sigma) ** 2)
    xx, yy = np.meshgrid(x, y)
    xxc, yyc = xx - center, yy - center
    r_sq = xxc**2 + yyc**2
    return np.exp(-(1 / 2) * (r_sq / sigma**2))


def just_the_gauss(x):
    initial_domain = [0.7, 0.8]
    transformed_domain = [0.25, 0.75]

    def transform(x):
        m = (initial_domain[1] - initial_domain[0]) / (
            transformed_domain[1] - transformed_domain[0]
        )
        return m * (x - transformed_domain[0]) + initial_domain[0]

    y = transform(x)
    u = (
        1
        / 6
        * (
            np.sqrt(np.maximum(1 - (20 * (y - 0.75 - 0.0025)) ** 2, 0))
            + np.sqrt(np.maximum(1 - (20 * (y - 0.75 + 0.0025)) ** 2, 0))
            + 4 * np.sqrt(np.maximum(1 - (20 * (y - 0.75)) ** 2, 0))
        )
    )
    return u


def four_friends(x):
    initial_domain = [0.7, 0.8]
    transformed_domain = [0.2, 0.8]

    def transform(x):
        m = (initial_domain[1] - initial_domain[0]) / (
            transformed_domain[1] - transformed_domain[0]
        )
        return m * (x - transformed_domain[0]) + initial_domain[0]

    def pulse(x):
        u = (
            1
            / 6
            * (
                np.sqrt(np.maximum(1 - (20 * (x - 0.75 - 0.0025)) ** 2, 0))
                + np.sqrt(np.maximum(1 - (20 * (x - 0.75 + 0.0025)) ** 2, 0))
                + 4 * np.sqrt(np.maximum(1 - (20 * (x - 0.75)) ** 2, 0))
            )
        )
        return u

    u = (
        just_the_gauss(transform(x, [0, 1], [0, 0.25 * x[-1]]))
        + just_the_gauss(transform(x, [0, 1], [0.25 * x[-1], 0.5 * x[-1]]))
        + just_the_gauss(transform(x, [0, 1], [0.5 * x[-1], 0.75 * x[-1]]))
        + just_the_gauss(transform(x, [0, 1], [0.75 * x[-1], 1 * x[-1]]))
    )

    return u


def composite_only_gauss(x):
    # only defined for 1d
    u = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] >= 0.7 and x[i] <= 0.8:
            u[i] = (
                1
                / 6
                * (
                    np.sqrt(max(1 - (20 * (x[i] - 0.75 - 0.0025)) ** 2, 0))
                    + np.sqrt(max(1 - (20 * (x[i] - 0.75 + 0.0025)) ** 2, 0))
                    + 4 * np.sqrt(max(1 - (20 * (x[i] - 0.75)) ** 2, 0))
                )
            )
    return u
