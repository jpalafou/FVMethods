import numpy as np

def trouble_detection(trouble, u0, unew):
    tolerance_ptge = 1e-8

    W_max = compute_W_max(u0, "xy")[:, 2:-2, 2:-2]
    W_min = compute_W_min(u0, "xy")[:, 2:-2, 2:-2]

    W_min -= np.abs(W_min) * tolerance_ptge
    W_max += np.abs(W_max) * tolerance_ptge

    possible_trouble = np.where(unew >= W_min, 0, 1)
    possible_trouble = np.where(unew <= W_max, possible_trouble, 1)
    # Now check for smooth extrema and relax the criteria for such cases
    if np.any(possible_trouble):
        alphax = compute_smooth_extrema(unew, "x")[:, 2:-2, :]
        alphay = compute_smooth_extrema(unew, "y")[:, :, 2:-2]
        alpha = np.where(alphax < alphay, alphax, alphay)
        trouble = np.where(
            possible_trouble == 1, np.where(alpha < 1, 1, trouble), trouble
        )
    else:
        trouble = possible_trouble


def trouble_detection1d(u0, unew, h):
    tolerance_ptge = 1e-5

    W_max = compute_W_max(u0, "x")[:, :, 2:-2]
    W_min = compute_W_min(u0, "x")[:, :, 2:-2]

    W_min -= np.abs(W_min) * tolerance_ptge
    W_max += np.abs(W_max) * tolerance_ptge

    possible_trouble = np.where(unew[:, :, 2:-2] >= W_min, 0, 1)
    possible_trouble = np.where(unew[:, :, 2:-2] <= W_max, possible_trouble, 1)

    # Now check for smooth extrema and relax the criteria for such cases
    trouble = np.zeros(len(possible_trouble))
    if np.any(possible_trouble):
        alphax = compute_smooth_extrema(unew, "x", h)
        alpha = alphax
        trouble = np.where(
            possible_trouble == 1, np.where(alpha < 1, 1, trouble), trouble
        )
    return trouble

def compute_W_ex(W, dim, case):
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

def compute_min(A, Amin, dim):
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
    na = np.newaxis
    if dim == 0:
        # dUdx(i) = [U(i+1)-U(i-1)]/[x_cv(i+1)-x_cv(i-1)]
        dU = (U[:, :, 2:] - U[:, :, :-2]) / (2 * h)

    elif dim == 1:
        # dUdy(j) = [U(j+1)-U(j-1)]/[y_cv(j+1)-y_cv(j-1)]
        dU = (U[:, 2:, :] - U[:, :-2, :]) / (2 * h)
    return dU


def compute_smooth_extrema(U, dim, h):
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
        vR = dU[..., 2:] - dU[..., 1:-1]
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
        ...
        # dU = first_order_derivative(U, s.dm.y_cv, 1)
        # d2U = first_order_derivative(dU, s.dm.y_cv[1:-1], 1)

        # dv = 0.5 * (s.dm.y_fp[na, 3:-2, na] - s.dm.y_fp[na, 2:-3, na]) * d2U
        # # vL = dU(j-1)-dU(j)
        # vL = dU[..., :-2, :] - dU[..., 1:-1, :]
        # # alphaL = min(1,max(vL,0)/(-dv)),1,min(1,min(vL,0)/(-dv)) for dv<0,dv=0,dv>0
        # alphaL = (
        #     -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0))
        #     / dv
        # )
        # alphaL = np.where(np.abs(dv) <= eps, 1, alphaL)
        # alphaL = np.where(alphaL < 1, alphaL, 1)
        # # vR = dU(j+1)-dU(j)
        # vR = dU[..., 2:, :] - dU[..., 1:-1, :]
        # # alphaR = min(1,max(vR,0)/(dv)),1,min(1,min(vR,0)/(dv)) for dv>0,dv=0,dv<0
        # alphaR = (
        #     np.where(dv > 0, np.where(vR > 0, vR, 0), np.where(vR < 0, vR, 0))
        #     / dv
        # )
        # alphaR = np.where(np.abs(dv) <= eps, 1, alphaR)
        # alphaR = np.where(alphaR < 1, alphaR, 1)
        # alphaR = np.where(alphaR < alphaL, alphaR, alphaL)
        # compute_min(s, alphaR, alphaL, 1)
        # alpha = alphaL

    return alpha


def compute_W_max(W, dim):
    return compute_W_ex(W, dim, "max")


def compute_W_min(W, dim):
    return compute_W_ex(W, dim, "min")


def minmod(SlopeL,SlopeR):
    #First compute ratio between slopes SlopeR/SlopeL
    #Then limit the ratio to be lower than 1
    #Finally, limit the ratio be positive and multiply by SlopeL to get the limited slope at the cell center
    #We use where instead of maximum/minimum as it doesn't propagte the NaNs caused when SlopeL=0
    ratio = SlopeR/SlopeL
    ratio = np.where(ratio<1,ratio,1)
    return np.where(ratio>0,ratio,0)*SlopeL


def moncen(dU_L,dU_R):
    #Compute central slope
    dU_C = 0.5*(dU_L + dU_R)
    slope = np.minimum(np.abs(2*dU_L),np.abs(2*dU_R))
    slope = np.sign(dU_C)*np.minimum(slope,np.abs(dU_C))
    return np.where(dU_L*dU_R>=0,slope,0)


def compute_slopes_x(dU, slope_limiter = "moncen"):
    na = np.newaxis
    if slope_limiter == "minmod":
        return minmod(dU[:,:,:-1],dU[:,:,1:])

    elif slope_limiter == "moncen":
        return moncen(dU[:,:,:-1],dU[:,:,1: ])


def compute_second_order_fluxes(u0):
    # u0 has gw=2
    na = np.newaxis
    ########################
    # X-Direction
    ########################
    dM = u0[:,:,1:] - u0[:,:,:-1]
    dMx = compute_slopes_x(dM, slope_limiter = "moncen")
    Sx = 0.5*dMx #Slope_x*dx/2

    #UR = U - SlopeC*dx/2, UL = U + SlopeC*dx/2
    right_interpolation = u0[:,:,1:-1] - Sx
    left_interpolation = u0[:,:,1:-1] + Sx

    return left_interpolation, right_interpolation