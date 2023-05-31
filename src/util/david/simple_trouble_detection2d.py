import numpy as np
import warnings

warnings.filterwarnings("ignore")


def trouble_detection2d(u0, unew, hx, hy):
    """
    args:
        u0      previous state array (1, ny + 4, nx + 4)
        unew    candidate state array (1, ny + 4, nx + 4)
    returns:
        trouble binary array indicating troubled cells (1, ny, nx)
    """
    tolerance_ptge = 0

    W_max = compute_W_max(u0, "xy")[:, 2:-2, 2:-2]
    W_min = compute_W_min(u0, "xy")[:, 2:-2, 2:-2]

    W_min -= np.abs(W_min) * tolerance_ptge
    W_max += np.abs(W_max) * tolerance_ptge

    possible_trouble = np.where(unew[:, 2:-2, 2:-2] >= W_min, 0, 1)
    possible_trouble = np.where(unew[:, 2:-2, 2:-2] <= W_max, possible_trouble, 1)

    # Now check for smooth extrema and relax the criteria for such cases
    trouble = np.zeros(possible_trouble.shape)
    if np.any(possible_trouble):
        alphax = compute_smooth_extrema(unew, "x", hx)[:, 2:-2, :]
        alphay = compute_smooth_extrema(unew, "y", hy)[:, :, 2:-2]
        alpha = np.where(alphax < alphay, alphax, alphay)
        trouble = np.where(
            possible_trouble == 1, np.where(alpha < 1, 1, trouble), trouble
        )
    # switch for later
    # trouble = np.ones(possible_trouble.shape)
    return trouble


def compute_W_ex(W, dim, case):
    """
    args:
        W   array (1,m,n)
        dim     'x', 'y', or 'xy'
        case    'min' or 'max'
    returns:
        W_f max/min of each cell and its neighbors (1,m,n)
    """
    if case == "max":
        f = np.maximum
    elif case == "min":
        f = np.minimum
    W_f = W.copy()
    if dim == "x" or dim == "xy":
        # W_max(i) = max(W(i-1),W(i),W(i+1))
        # First comparing W(i) and W(i+1)
        W_f[:, :, :-1] = f(W[:, :, :-1], W[:, :, 1:])
        # Now comparing W_max(i) and W_(i-1)
        W_f[:, :, 1:] = f(W_f[:, :, 1:], W[:, :, :-1])

    if dim == "y" or dim == "xy":
        # W_max(j) = max(W(j-1),W(j),W(j+1))
        # First comparing W(j) and W(j+1)
        if dim == "xy":
            # W_max(j) = max(W_max(j-1),W_max(j),W_max(j+1))
            W_f[:, :-1, :] = f(W_f[:, :-1, :], W_f[:, 1:, :])
            W_f[:, 1:, :] = f(W_f[:, 1:, :], W_f[:, :-1, :])
        else:
            W_f[:, :-1, :] = f(W[:, :-1, :], W[:, 1:, :])
            # Now comparing W_max(j) and W_(j-1)
            W_f[:, 1:, :] = f(W_f[:, 1:, :], W[:, :-1, :])
    return W_f


def compute_W_max(W, dim):
    return compute_W_ex(W, dim, "max")


def compute_W_min(W, dim):
    return compute_W_ex(W, dim, "min")


def compute_min(A, Amin, dim):
    """
    args:
        A       (..., m, n)
        Amin    (..., m, n)
        dim     0 or 1
    overwrites:
        Amin    minimum of each cell and it's neighbors  
    """
    if dim == 0:
        Amin[..., :-1] = np.where(
            A[..., :-1] < A[..., 1:], A[..., :-1], A[..., 1:]
        )
        Amin[..., 1:] = np.where(
            A[..., :-1] < Amin[..., 1:], A[..., :-1], Amin[..., 1:]
        )
        # if s.BC[0] == "periodic":
        #    Amin[...,-1] = np.where(A[..., 0]<Amin[...,-1],A[..., 0],Amin[...,-1])
        #    Amin[..., 0] = np.where(A[...,-1]<Amin[..., 0],A[...,-1],Amin[..., 0])
    elif dim == 1:
        Amin[..., :-1, :] = np.where(
            A[..., :-1, :] < A[..., 1:, :], A[..., :-1, :], A[..., 1:, :]
        )
        Amin[..., 1:, :] = np.where(
            A[..., :-1, :] < Amin[..., 1:, :], A[..., :-1, :], Amin[..., 1:, :]
        )
        # if s.BC[1] == "periodic":
        #    Amin[...,-1,:] = np.where(A[..., 0,:]<Amin[...,-1,:],A[..., 0,:],Amin[...,-1,:])
        #    Amin[..., 0,:] = np.where(A[...,-1,:]<Amin[..., 0,:],A[...,-1,:],Amin[..., 0,:])


def first_order_derivative(U, dim, h):
    """
    args:
        U   (1, m, n)
        dim 0 or 1
        h   uniform mesh size
    returns:
        dU  (1, m, n - 2) if dim = 0
            (1, m - 2, n) if dim = 1
    """
    na = np.newaxis
    if dim == 0:
        # dUdx(i) = [U(i+1)-U(i-1)]/[x_cv(i+1)-x_cv(i-1)]
        dU = (U[:, :, 2:] - U[:, :, :-2]) / (2 * h)

    elif dim == 1:
        # dUdy(j) = [U(j+1)-U(j-1)]/[y_cv(j+1)-y_cv(j-1)]
        dU = (U[:, 2:, :] - U[:, :-2, :]) / (2 * h)
    return dU


def compute_smooth_extrema(U, dim, h):
    """
    args:
        U   (1, ny + 4, nx + 4)
        dim 'x' or 'y'
        h   uniform mesh size
    returns
        alpha   indicator for whether a cell is a smooth extremum
                (1, ny + 4, nx) if dim = 'x'
                (1, ny, nx + 4) if dim = 'y'
    """
    na = np.newaxis
    eps = 0
    if dim == "x":
        # First derivative dUdx(i) = [U(i+1)-U(i-1)]/[x_cv(i+1)-x_cv(i-1)]
        dU = first_order_derivative(U, 0, h)
        # Second derivative d2Udx2(i) = [dU(i+1)-dU(i-1)]/[x_cv(i+1)-x_cv(i-1)]
        d2U = first_order_derivative(dU, 0, h)

        dv = 0.5 * h * d2U
        # vL = dU(i-1)-dU(i)
        vL = dU[:, :, :-2] - dU[:, :, 1:-1]
        # alphaL = min(1,max(vL,0)/(-dv)),1,min(1,min(vL,0)/(-dv)) for dv<0,dv=0,dv>0
        alphaL = (
            -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0))
            / dv
        )
        alphaL = np.where(np.abs(dv) <= eps, 1, alphaL)
        alphaL = np.where(alphaL < 1, alphaL, 1)
        # vR = dU(i+1)-dU(i)
        vR = dU[:, :, 2:] - dU[:, :, 1:-1]
        # alphaR = min(1,max(vR,0)/(dv)),1,min(1,min(vR,0)/(dv)) for dv>0,dv=0,dv<0
        alphaR = (
            np.where(dv > 0, np.where(vR > 0, vR, 0), np.where(vR < 0, vR, 0))
            / dv
        )
        alphaR = np.where(np.abs(dv) <= eps, 1, alphaR)
        alphaR = np.where(alphaR < 1, alphaR, 1)
        alphaR = np.where(alphaR < alphaL, alphaR, alphaL)
        compute_min(alphaR, alphaL, 0)
        alpha = alphaL

    if dim == "y":
        dU = first_order_derivative(U, 1, h)
        d2U = first_order_derivative(dU, 1, h)

        dv = 0.5 * h * d2U
        # vL = dU(j-1)-dU(j)
        vL = dU[:, :-2, :] - dU[:, 1:-1, :]
        # alphaL = min(1,max(vL,0)/(-dv)),1,min(1,min(vL,0)/(-dv)) for dv<0,dv=0,dv>0
        alphaL = (
            -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0))
            / dv
        )
        alphaL = np.where(np.abs(dv) <= eps, 1, alphaL)
        alphaL = np.where(alphaL < 1, alphaL, 1)
        # vR = dU(j+1)-dU(j)
        vR = dU[:, 2:, :] - dU[:, 1:-1, :]
        # alphaR = min(1,max(vR,0)/(dv)),1,min(1,min(vR,0)/(dv)) for dv>0,dv=0,dv<0
        alphaR = (
            np.where(dv > 0, np.where(vR > 0, vR, 0), np.where(vR < 0, vR, 0))
            / dv
        )
        alphaR = np.where(np.abs(dv) <= eps, 1, alphaR)
        alphaR = np.where(alphaR < 1, alphaR, 1)
        alphaR = np.where(alphaR < alphaL, alphaR, alphaL)
        compute_min(alphaR, alphaL, 1)
        alpha = alphaL

    return alpha


def minmod(SlopeL,SlopeR):
    """
    args:
        SlopeL  (1, m, n)
        SlopeR  (1, m, n)
    returns:
        minmod slope limiter    (1, m, n)
    """
    #First compute ratio between slopes SlopeR/SlopeL
    #Then limit the ratio to be lower than 1
    #Finally, limit the ratio be positive and multiply by SlopeL to get the limited slope at the cell center
    #We use where instead of maximum/minimum as it doesn't propagte the NaNs caused when SlopeL=0
    ratio = SlopeR/SlopeL
    ratio = np.where(ratio<1,ratio,1)
    return np.where(ratio>0,ratio,0)*SlopeL


def moncen(dU_L,dU_R):
    """
    args:
        SlopeL  (1, m, n)
        SlopeR  (1, m, n)
    returns:
        moncen slope limiter    (1, m, n)
    """
    #Compute central slope
    dU_C = 0.5*(dU_L + dU_R)
    slope = np.minimum(np.abs(2*dU_L),np.abs(2*dU_R))
    slope = np.sign(dU_C)*np.minimum(slope,np.abs(dU_C))
    return np.where(dU_L*dU_R>=0,slope,0)


def compute_slopes(dU, slope_limiter = "moncen", dim = 'x'):
    """
    args:
        dU  (1, m, n)
        slope_limiter  'moncen' or 'minmod'
        dim 'x' or 'y'
    returns:
        slopes  (1, m, n - 1) if dim = 'x'
                (1, m - 1, n) if dim = 'y'
    """
    na = np.newaxis
    if dim == 'x':
        dU_left = dU[:,:,:-1]
        dU_right = dU[:,:,1:]
    elif dim == 'y':
        dU_left = dU[:,:-1,:]
        dU_right = dU[:,1:,:]
    if slope_limiter == "minmod":
        return minmod(dU_left,dU_right)
    elif slope_limiter == "moncen":
        return moncen(dU_left,dU_right)


def compute_second_order_fluxes(u0, dim):
    """
    args:
        u0      previous state array (1, ny + 4, nx + 4)
        dim     'x' or 'y'
    returns:
        north/south or east/west 2nd order reconstructed cell face values
            tuple of    (1, ny, nx + 2) if dim = 'x'
                        (1, ny + 2, nx) if dim = 'y'
    """
    # u0 has gw=2
    na = np.newaxis
    ########################
    # X-Direction
    ########################
    if dim == 'x':
        dM = u0[:,2:-2,1:] - u0[:,2:-2,:-1]
        dMx = compute_slopes(dM, slope_limiter = "moncen", dim=dim)
        Sx = 0.5*dMx #Slope_x*dx/2
        #UR = U - SlopeC*dx/2, UL = U + SlopeC*dx/2
        right_interpolation = u0[:,2:-2,1:-1] + Sx
        left_interpolation = u0[:,2:-2,1:-1] - Sx
    elif dim == 'y':
        dM = u0[:,1:,2:-2] - u0[:,:-1,2:-2]
        dMy = compute_slopes(dM, slope_limiter = "moncen", dim=dim)
        Sy = 0.5*dMy #Slope_x*dx/2
        #UR = U - SlopeC*dx/2, UL = U + SlopeC*dx/2
        right_interpolation = u0[:,1:-1,2:-2] + Sy
        left_interpolation = u0[:,1:-1,2:-2] - Sy

    return left_interpolation, right_interpolation