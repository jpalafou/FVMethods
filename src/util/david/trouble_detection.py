import numpy as np
import finite_volume as fv

# david uses (8, Ny, Nx) for mhd
# i will shape my arrays into (1, Ny, Nx)


def trouble_detection(s, m):
    # Reset to check troubled control volumes
    na = np.newaxis
    s.dm.trouble[...] = 0
    s.dm.affected_face_x[...] = 0
    s.dm.affected_face_y[...] = 0
    if s.so_godunov:
        s.dm.trouble[...] = 1
    else:
        # if s.well_balance:
        if False:
            # U_new and W_cv are perturbations up to this point
            W_new = s.compute_primitives(s.dm.U_new + s.dm.U_eq_cv)
            if s.detect_over_perturbations:
                W_new -= s.dm.W_eq_cv
                s.dm.M_fv[:, 2:-2, 2:-2] = s.dm.W_cv
            else:
                s.dm.M_fv[:, 2:-2, 2:-2] = s.dm.W_cv + s.dm.W_eq_cv
        else:
            s.dm.M_fv[:, 2:-2, 2:-2] = s.dm.W_cv
            W_new = s.compute_primitives(s.dm.U_new)
            # W_new is the candidate solution
            # M_fv is the previous solution
            #   ignoring rk timesteps?
        ##############################################
        # NAD Check for numerically adimissible values
        ##############################################
        # First check if DMP criteria is met, if it is we can avoid computing alpha
        # W_old -> s.dm.M_fv
        if s.NAD == "1st":  # orthogonally adjacent (first neighbors)
            fv.FV_Boundaries_x(
                s, primitives=True, perturbation=s.detect_over_perturbations
            )
            W_max = compute_W_max(s.dm.M_fv, "x")
            W_min = compute_W_min(s.dm.M_fv, "x")
            fv.FV_Boundaries_y(
                s, primitives=True, perturbation=s.detect_over_perturbations
            )
            W_max = np.maximum(compute_W_max(s.dm.M_fv, "y"), W_max)[
                :, 2:-2, 2:-2
            ]
            W_min = np.minimum(compute_W_min(s.dm.M_fv, "y"), W_min)[
                :, 2:-2, 2:-2
            ]
        elif s.NAD == "2nd":  # corners (second neighbors)
            fv.FV_Boundaries(
                s, primitives=True, perturbation=s.detect_over_perturbations
            )
            W_max = compute_W_max(s.dm.M_fv, "xy")[:, 2:-2, 2:-2]
            W_min = compute_W_min(s.dm.M_fv, "xy")[:, 2:-2, 2:-2]
        else:
            print("Incorrect option for NAD")

        if s.n > 0:
            W_min -= np.abs(W_min) * s.tolerance_ptge
            W_max += np.abs(W_max) * s.tolerance_ptge

        possible_trouble = np.where(W_new >= W_min, 0, 1)
        possible_trouble = np.where(W_new <= W_max, possible_trouble, 1)
        # Now check for smooth extrema and relax the criteria for such cases
        if np.any(possible_trouble) and s.n > 1:
            s.dm.M_fv[:, 2:-2, 2:-2] = W_new
            fv.FV_Boundaries_x(
                s, primitives=True, perturbation=s.detect_over_perturbations
            )
            alphax = compute_smooth_extrema(s, s.dm.M_fv, "x")[:, 2:-2, :]
            fv.FV_Boundaries_y(
                s, primitives=True, perturbation=s.detect_over_perturbations
            )
            alphay = compute_smooth_extrema(s, s.dm.M_fv, "y")[:, :, 2:-2]
            alpha = np.where(alphax < alphay, alphax, alphay)
            # for var in s.limiting_vars:
            for var in [0]:
                s.dm.trouble[...] = np.where(
                    possible_trouble[var] == 1,
                    np.where(alpha[var] < 1, 1, s.dm.trouble),
                    s.dm.trouble,
                )
        else:
            for var in s.limiting_vars:
                s.dm.trouble[...] = np.where(
                    possible_trouble[var] == 1, 1, s.dm.trouble
                )

        # if s.complex_BC:
        if False:
            s.dm.trouble[:, :-1] = np.where(
                (s.dm.c_BC_fv[:, 1:] - s.dm.c_BC_fv[:, :-1]) < 0,
                1,
                s.dm.trouble[:, :-1],
            )
            s.dm.trouble[1:, :] = np.where(
                (s.dm.c_BC_fv[:-1, :] - s.dm.c_BC_fv[1:, :]) < 0,
                1,
                s.dm.trouble[1:, :],
            )
        ###########################
        # PAD Check for physically admissible values
        ###########################
        # if s.PAD:
        if False:
            if s.well_balance and s.detect_over_perturbations:
                W_new += s.dm.W_eq_cv
            # For the density
            s.dm.trouble[...] = np.where(
                W_new[s._d_, ...] >= s.min_rho, s.dm.trouble, 1
            )
            s.dm.trouble[...] = np.where(
                W_new[s._d_, ...] <= s.max_rho, s.dm.trouble, 1
            )
            # For the pressure
            s.dm.trouble[...] = np.where(
                W_new[s._p_, ...] >= s.min_P, s.dm.trouble, 1
            )

        # if s.mhd and s.detect_on_B:
        if False:
            ##############################################
            # Trouble detection for the Magnetic field at
            # at the respective faces
            ##############################################
            # Bx_fv -> [(N*(n+1))+4,(N*(n+1))+1]
            Bx_max = compute_W_max(s.dm.Bx_fv[np.newaxis], "y")[0, 2:-2, 1:-1]
            Bx_min = compute_W_min(s.dm.Bx_fv[np.newaxis], "y")[0, 2:-2, 1:-1]
            # By_fv -> [(N*(n+1))+1,(N*(n+1))+4]
            By_max = compute_W_max(s.dm.By_fv[np.newaxis], "x")[0, 1:-1, 2:-2]
            By_min = compute_W_min(s.dm.By_fv[np.newaxis], "x")[0, 1:-1, 2:-2]
            if s.n > 0:
                Bx_min -= np.abs(Bx_min) * s.tolerance_ptge
                Bx_max += np.abs(Bx_max) * s.tolerance_ptge
                By_min -= np.abs(By_min) * s.tolerance_ptge
                By_max += np.abs(By_max) * s.tolerance_ptge
            possible_trouble_Bx = np.where(s.dm.Bx_new < Bx_min, 1, 0)
            possible_trouble_Bx = np.where(
                s.dm.Bx_new > Bx_max, 1, possible_trouble_Bx
            )
            possible_trouble_By = np.where(s.dm.By_new < By_min, 1, 0)
            possible_trouble_By = np.where(
                s.dm.By_new > By_max, 1, possible_trouble_By
            )
            if (
                np.any(possible_trouble_Bx)
                or np.any(possible_trouble_By)
                and s.n > 0
            ):
                s.dm.Bx_fv[2:-2, 1:-1] = s.dm.Bx_new
                s.dm.By_fv[1:-1, 2:-2] = s.dm.By_new
                fv.B_Boundaries(s)
                alpha = compute_smooth_extrema(
                    s, s.dm.Bx_fv[np.newaxis, :, 1:-1], "y"
                )[0]
                s.dm.affected_face_x[...] = np.where(
                    possible_trouble_Bx == 1, np.where(alpha < 1, 1, 0), 0
                )
                alpha = compute_smooth_extrema(
                    s, s.dm.By_fv[np.newaxis, 1:-1, :], "x"
                )[0]
                s.dm.affected_face_y[...] = np.where(
                    possible_trouble_By == 1, np.where(alpha < 1, 1, 0), 0
                )

    s.n_troubles += s.dm.trouble.sum()
    # if s.substep_troubles:
    if False:
        s.dm.step_trouble += s.dm.trouble
    na = np.newaxis

    s.dm.affected_face_x[:, :-1] = s.dm.trouble
    s.dm.affected_face_x[:, 1:] = np.where(
        s.dm.trouble == 1, 1, s.dm.affected_face_x[:, 1:]
    )

    s.dm.affected_face_y[:-1, :] = s.dm.trouble
    s.dm.affected_face_y[1:, :] = np.where(
        s.dm.trouble == 1, 1, s.dm.affected_face_y[1:, :]
    )

    if s.BC[0] == "periodic":
        s.dm.affected_face_x[:, 0] = s.dm.affected_face_x[:, -1] = np.maximum(
            s.dm.affected_face_x[:, 0], s.dm.affected_face_x[:, -1]
        )
    if s.BC[1] == "periodic":
        s.dm.affected_face_y[0, :] = s.dm.affected_face_y[-1, :] = np.maximum(
            s.dm.affected_face_y[0, :], s.dm.affected_face_y[-1, :]
        )

    if s.mhd:
        # Finally we flag the affected corner points
        s.dm.affected_corner[...] = 0
        s.dm.affected_corner[:-1, :] = s.dm.affected_face_x
        s.dm.affected_corner[1:, :] = np.where(
            s.dm.affected_face_x == 1, 1, s.dm.affected_corner[1:, :]
        )
        s.dm.affected_corner[:, :-1] = np.where(
            s.dm.affected_face_y == 1, 1, s.dm.affected_corner[:, :-1]
        )
        s.dm.affected_corner[:, 1:] = np.where(
            s.dm.affected_face_y == 1, 1, s.dm.affected_corner[:, 1:]
        )


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


def compute_min(s, A, Amin, dim):
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


def first_order_derivative(U, x_cv, dim):
    na = np.newaxis
    if dim == 0:
        # dUdx(i) = [U(i+1)-U(i-1)]/[x_cv(i+1)-x_cv(i-1)]
        dU = (U[:, :, 2:] - U[:, :, :-2]) / (
            x_cv[na, na, 2:] - x_cv[na, na, :-2]
        )

    elif dim == 1:
        # dUdy(j) = [U(j+1)-U(j-1)]/[y_cv(j+1)-y_cv(j-1)]
        dU = (U[:, 2:, :] - U[:, :-2, :]) / (
            x_cv[na, 2:, na] - x_cv[na, :-2, na]
        )
    return dU


def compute_smooth_extrema(s, U, dim):
    na = np.newaxis
    eps = 0
    if dim == "x":
        # First derivative dUdx(i) = [U(i+1)-U(i-1)]/[x_cv(i+1)-x_cv(i-1)]
        dU = first_order_derivative(U, s.dm.x_cv, 0)
        # Second derivative d2Udx2(i) = [dU(i+1)-dU(i-1)]/[x_cv(i+1)-x_cv(i-1)]
        d2U = first_order_derivative(dU, s.dm.x_cv[1:-1], 0)

        dv = 0.5 * (s.dm.x_fp[na, na, 3:-2] - s.dm.x_fp[na, na, 2:-3]) * d2U
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
        compute_min(s, alphaR, alphaL, 0)
        alpha = alphaL

    if dim == "y":
        dU = first_order_derivative(U, s.dm.y_cv, 1)
        d2U = first_order_derivative(dU, s.dm.y_cv[1:-1], 1)

        dv = 0.5 * (s.dm.y_fp[na, 3:-2, na] - s.dm.y_fp[na, 2:-3, na]) * d2U
        # vL = dU(j-1)-dU(j)
        vL = dU[..., :-2, :] - dU[..., 1:-1, :]
        # alphaL = min(1,max(vL,0)/(-dv)),1,min(1,min(vL,0)/(-dv)) for dv<0,dv=0,dv>0
        alphaL = (
            -np.where(dv < 0, np.where(vL > 0, vL, 0), np.where(vL < 0, vL, 0))
            / dv
        )
        alphaL = np.where(np.abs(dv) <= eps, 1, alphaL)
        alphaL = np.where(alphaL < 1, alphaL, 1)
        # vR = dU(j+1)-dU(j)
        vR = dU[..., 2:, :] - dU[..., 1:-1, :]
        # alphaR = min(1,max(vR,0)/(dv)),1,min(1,min(vR,0)/(dv)) for dv>0,dv=0,dv<0
        alphaR = (
            np.where(dv > 0, np.where(vR > 0, vR, 0), np.where(vR < 0, vR, 0))
            / dv
        )
        alphaR = np.where(np.abs(dv) <= eps, 1, alphaR)
        alphaR = np.where(alphaR < 1, alphaR, 1)
        alphaR = np.where(alphaR < alphaL, alphaR, alphaL)
        compute_min(s, alphaR, alphaL, 1)
        alpha = alphaL

    return alpha
