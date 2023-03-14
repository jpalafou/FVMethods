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

# def compute_slopes_x(dU,limiter):
#     na = np.newaxis
#     if limiter == "minmod":
#         return minmod(dU[:,:,:-1],dU[:,:,1:])

#     elif limiter == "moncen":
#         return moncen(dU[:,:,:-1],dU[:,:,1: ],1,1,1)

# def compute_slopes_y(dU,limiter):
#     na = np.newaxis
#     if limiter == "minmod":
#         return minmod(dU[:,:-1,:],dU[:,1:,:])

#     elif limiter == "moncen":
#         return moncen(dU[:,:-1,:],dU[:,1: ,:],1,1,1)

# def compute_second_order_fluxes(cell_values,limiter):
#     # assume constant mesh
#     ########################
#     # X-Direction
#     ########################
#     dM = cell_values[:,:,1:] - cell_values[:,:,:-1] # column subtraction
#     dMx = compute_slopes_x(dM,limiter)
#     Sx = 0.5*dMx

#     ########################
#     # Y-Direction
#     ########################
#     dM = cell_values[:,1:,:] - cell_values[:,:-1,:] # row subtraction
#     dMy = compute_slopes_y(dM,limiter)
#     Sy = 0.5*dMy

#     #UR = U - SlopeC*dx/2, UL = U + SlopeC*dx/2
#     MR_face_x = cell_values[:,2:-2,2:-1] - Sx[:,2:-2,1: ]
#     ML_face_x = cell_values[:,2:-2,1:-2] + Sx[:,2:-2,:-1]
#     MR_face_y = cell_values[:,2:-1,2:-2] - Sy[:,1: ,2:-2]
#     ML_face_y = cell_values[:,1:-2,2:-2] + Sy[:,:-1,2:-2]
#     return MR_face_x, ML_face_x, MR_face_y, ML_face_y
