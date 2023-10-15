import numpy as np

# This is the advection speed used throughout this notebook
a = 1


# This function sets the initial conditions
def set_ic(x, type="sinus"):
    u = np.zeros([x.size])
    if type == "sinus":
        for i in range(0, x.size):
            u[i] = np.cos(2 * np.pi * x[i])
    elif type == "gaussian":
        for i in range(0, x.size):
            u[i] = np.exp(-256.0 * (x[i] - 0.5) ** 2)
    elif type == "square":
        for i in range(int(x.size / 4), int(3 * x.size / 4)):
            u[i] = 1
    elif type == "composite":
        for i in range(0, x.size):
            if x[i] >= 0.1 and x[i] <= 0.2:
                u[i] = (
                    1
                    / 6
                    * (
                        np.exp(
                            -np.log(2) / 36 / 0.0025**2 * (x[i] - 0.0025 - 0.15) ** 2
                        )
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
    elif type == "hammett":
        for i in range(0, x.size):
            u[i] = np.exp(-4 * np.log(2) * (x[i] - 0.3) ** 2 / 0.1**2)
            if x[i] >= 0.6 and x[i] <= 0.8:
                u[i] = 1
    elif callable(type):
        u = type(x)
    else:
        print("Unkown IC type")
    return u


# This function sets the boundary conditions
# Note that it only works up to fifth-order because we have
# only 4 ghost zones in each direction.
def set_bc_high_order(u):
    u[0] = u[-10]
    u[1] = u[-9]
    u[2] = u[-8]
    u[3] = u[-7]
    u[4] = u[-6]
    u[-1] = u[9]
    u[-2] = u[8]
    u[-3] = u[7]
    u[-4] = u[6]
    u[-5] = u[5]


def smooth_extrema(u):
    # compute central first derivative
    du = 0.5 * (u[2:] - u[:-2])
    uprime = np.pad(du, [(1, 1)])
    set_bc_high_order(uprime)

    # compute left, right and central second derivative
    dlft = uprime[1:-1] - uprime[:-2]
    drgt = uprime[2:] - uprime[1:-1]
    dmid = 0.5 * (dlft + drgt)

    # detect discontinuity on the left (alpha_left<1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alfp = np.minimum(1, np.maximum(2 * dlft, 0) / dmid)
        alfm = np.minimum(1, np.minimum(2 * dlft, 0) / dmid)
    alfl = np.where(dmid > 0, alfp, alfm)
    alfl = np.where(dmid == 0, 1, alfl)

    # detect discontinuity on the right (alpha_right<1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alfp = np.minimum(1, np.maximum(2 * drgt, 0) / dmid)
        alfm = np.minimum(1, np.minimum(2 * drgt, 0) / dmid)
    alfr = np.where(dmid > 0, alfp, alfm)
    alfr = np.where(dmid == 0, 1, alfr)

    # finalize smooth extrema marker (alpha=1)
    alf = np.minimum(alfl, alfr)
    alpha = np.pad(alf, [(1, 1)])
    set_bc_high_order(alpha)

    return alpha


def trace(u, alpha, space=1, limiter=True, smooth_extrema_detection=True):
    if space == 1:
        uleft = u[4:-4]
        uright = u[4:-4]

    if space == 2:
        uleft = (-u[3:-5] + 4 * u[4:-4] + u[5:-3]) / 4
        uright = (u[3:-5] + 4 * u[4:-4] - u[5:-3]) / 4

    if space == 3:
        uleft = (-u[3:-5] + 5 * u[4:-4] + 2 * u[5:-3]) / 6
        uright = (2 * u[3:-5] + 5 * u[4:-4] - u[5:-3]) / 6
        umiddle = (-u[3:-5] + 26 * u[4:-4] - u[5:-3]) / 24

    if space == 4:
        uleft = (u[2:-6] - 6 * u[3:-5] + 20 * u[4:-4] + 10 * u[5:-3] - u[6:-2]) / 24
        uright = (-u[2:-6] + 10 * u[3:-5] + 20 * u[4:-4] - 6 * u[5:-3] + u[6:-2]) / 24
        umiddle = (-u[3:-5] + 26 * u[4:-4] - u[5:-3]) / 24

    if space == 5:
        uleft = (
            2 * u[2:-6] - 13 * u[3:-5] + 47 * u[4:-4] + 27 * u[5:-3] - 3 * u[6:-2]
        ) / 60
        uright = (
            -3 * u[2:-6] + 27 * u[3:-5] + 47 * u[4:-4] - 13 * u[5:-3] + 2 * u[6:-2]
        ) / 60
        umiddle = (
            27 * u[2:-6] - 348 * u[3:-5] + 6402 * u[4:-4] - 348 * u[5:-3] + 27 * u[6:-2]
        ) / 5760

    if space == 6:
        uleft = (
            -1 * u[1:-7]
            + 8 * u[2:-6]
            - 31 * u[3:-5]
            + 94 * u[4:-4]
            + 59 * u[5:-3]
            - 10 * u[6:-2]
            + 1 * u[7:-1]
        ) / 120
        uright = (
            1 * u[1:-7]
            - 10 * u[2:-6]
            + 59 * u[3:-5]
            + 94 * u[4:-4]
            - 31 * u[5:-3]
            + 8 * u[6:-2]
            - 1 * u[7:-1]
        ) / 120
        umiddle = (
            (3 / 640) * u[2:-6]
            - (29 / 480) * u[3:-5]
            + (1067 / 960) * u[4:-4]
            - (29 / 480) * u[5:-3]
            + (3 / 640) * u[6:-2]
        )

    if space > 1:
        bigm = np.maximum(u[3:-5], np.maximum(u[4:-4], u[5:-3]))
        smallm = np.minimum(u[3:-5], np.minimum(u[4:-4], u[5:-3]))

        if space > 2:
            bigmj = np.maximum(umiddle, np.maximum(uleft, uright)) - u[4:-4]
            smallmj = np.minimum(umiddle, np.minimum(uleft, uright)) - u[4:-4]
        else:
            bigmj = np.maximum(uleft, uright) - u[4:-4]
            smallmj = np.minimum(uleft, uright) - u[4:-4]

        # compute limiter
        theta = np.minimum(
            1,
            np.minimum(
                abs(bigm - u[4:-4]) / (abs(bigmj) + 1e-15),
                abs(smallm - u[4:-4]) / (abs(smallmj) + 1e-15),
            ),
        )
        # apply smooth extrema detection
        if smooth_extrema_detection:
            aslp = np.minimum(np.minimum(alpha[3:-5], alpha[5:-3]), alpha[4:-4])
            theta = np.where(aslp < 1, theta, 1)
        # apply limiter
        if limiter:
            uleft = theta * (uleft - u[4:-4]) + u[4:-4]
            uright = theta * (uright - u[4:-4]) + u[4:-4]

    return uleft, uright


def moncen(u, smooth_extrema_detection):
    dlft = u[1:-1] - u[:-2]
    drgt = u[2:] - u[1:-1]
    dcen = 0.5 * (dlft + drgt)
    dsgn = np.sign(dcen)
    du = dsgn * np.minimum(np.minimum(abs(2 * dlft), abs(2 * drgt)), abs(dcen))
    du = np.where(dlft * drgt <= 0, 0, du)
    return du


def MUSCLhancock(u, h, limit=False, smooth_extrema_detection=False, limiter=moncen):
    dlft = u[1:-1] - u[:-2]
    drgt = u[2:] - u[1:-1]
    duc = 0.5 * (dlft + drgt)
    du = duc
    if limit:
        du = limiter(u, smooth_extrema_detection)
    if smooth_extrema_detection:
        alpha = smooth_extrema(u)
        aslp = np.minimum(np.minimum(alpha[:-2], alpha[2:]), alpha[1:-1])
        du = np.where(aslp < 1, du, duc)
    return du


def solve_high_order(
    tend=1,
    n=100,
    cfl=0.4,
    ic_type="square",
    time=1,
    space=1,
    limiter=True,
    smooth_extrema_detection=True,
):
    # set run parameters
    h = 1 / n
    dt = cfl * h / abs(a)
    print(dt)
    nitertot = int(tend / dt)
    print("cell=", n, " iter=", nitertot)

    # set grid geometry
    xf = np.linspace(0, 1, n + 1)
    x = 0.5 * (xf[1:] + xf[:-1])

    # allocate permanent storage
    u = np.zeros([nitertot + 1, n])

    # set initial conditions
    u[0] = set_ic(x, ic_type)

    # allocate temporary workspace
    u1 = np.zeros([n + 10])
    u2 = np.zeros([n + 10])
    u3 = np.zeros([n + 10])
    u4 = np.zeros([n + 10])

    # init time and iteration counter
    t = 0
    niter = 1

    # main time loop
    while t < tend and niter <= nitertot:
        u1[5:-5] = u[niter - 1]  # copy old solution

        if time == 1:
            set_bc_high_order(u1)
            alpha = u1 * 0.0 + 1.0
            uleft, uright = trace(
                u1,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k1 = -a * (uleft[1:-1] - uleft[:-2]) / h
            unew = u1[5:-5] + k1 * dt

        if time == 2:
            set_bc_high_order(u1)
            alpha = smooth_extrema(u1)
            uleft, uright = trace(
                u1,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k1 = -a * (uleft[1:-1] - uleft[:-2]) / h
            u2[5:-5] = u1[5:-5] + k1 * dt / 2
            # u2[5:-5] = u1[5:-5] + k1 * dt

            set_bc_high_order(u2)
            alpha = smooth_extrema(u2)
            uleft, uright = trace(
                u2,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k2 = -a * (uleft[1:-1] - uleft[:-2]) / h
            unew = u1[5:-5] + k2 * dt
            # unew = u1[5:-5]/2 + (u2[5:-5] + k2 * dt)/2

        if time == 7:
            """
            second-order MUSCL-Hancock scheme
            """
            # prediction
            set_bc_high_order(u1)
            du = MUSCLhancock(
                u1,
                h,
                limit=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
                limiter=moncen,
            )
            uleft = u1[4:-4] + (1 - (a * dt / h)) * du[3:-3] / 2
            uright = u1[4:-4] - (1 + (a * dt / h)) * du[3:-3] / 2
            k = -a * (uleft[1:-1] - uleft[:-2]) / h
            unew = u1[5:-5] + k * dt

        if time == 3:
            set_bc_high_order(u1)
            alpha = smooth_extrema(u1)
            uleft, uright = trace(
                u1,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k1 = -a * (uleft[1:-1] - uleft[:-2]) / h
            u2[5:-5] = u1[5:-5] + k1 * dt / 3
            #            u2[5:-5] = u1[5:-5] + k1 * dt
            set_bc_high_order(u2)
            alpha = smooth_extrema(u2)
            uleft, uright = trace(
                u2,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k2 = -a * (uleft[1:-1] - uleft[:-2]) / h
            u3[5:-5] = u1[5:-5] + k2 * 2 * dt / 3
            #            u3[5:-5] = 3/4*u1[5:-5] + 1/4*(u2[5:-5]+k2 * dt)
            set_bc_high_order(u3)
            alpha = smooth_extrema(u3)
            uleft, uright = trace(
                u3,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k3 = -a * (uleft[1:-1] - uleft[:-2]) / h
            unew = u1[5:-5] + (k1 + 3 * k3) * dt / 4
        #            unew = 1/3*u1[5:-5] + 2/3*(u3[5:-5]+k3 * dt)

        if time == 4:
            set_bc_high_order(u1)
            alpha = smooth_extrema(u1)
            uleft, uright = trace(
                u1,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k1 = -a * (uleft[1:-1] - uleft[:-2]) / h
            u2[5:-5] = u1[5:-5] + k1 * dt / 2
            set_bc_high_order(u2)
            alpha = smooth_extrema(u2)
            uleft, uright = trace(
                u2,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k2 = -a * (uleft[1:-1] - uleft[:-2]) / h
            u3[5:-5] = u1[5:-5] + k2 * dt / 2
            set_bc_high_order(u3)
            alpha = smooth_extrema(u3)
            uleft, uright = trace(
                u3,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k3 = -a * (uleft[1:-1] - uleft[:-2]) / h
            u4[5:-5] = u1[5:-5] + k3 * dt
            set_bc_high_order(u4)
            alpha = smooth_extrema(u4)
            uleft, uright = trace(
                u4,
                alpha,
                space=space,
                limiter=limiter,
                smooth_extrema_detection=smooth_extrema_detection,
            )
            k4 = -a * (uleft[1:-1] - uleft[:-2]) / h
            unew = u1[5:-5] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        u[niter] = unew  # store new solution
        t = t + dt  # update time
        niter = niter + 1  # update iteration count
    print("Done ", niter - 1, t)
    print("Maximum principle:", np.min(u[-1]), np.max(u[-1]))
    return u
