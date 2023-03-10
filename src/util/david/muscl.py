import numpy as np
import finite_volume as fv

# 0 is the density parameter and it is the only one we need
def compute_prediction(s: "simulator",U,dxU,dyU,dxUeq=np.zeros((10)),dyUeq=np.zeros((10))):
    gamma=s.gamma
    _p_ = s._p_
    _s_ = s._s_
    _vx_ = s._vx_
    _vy_ = s._vy_
    if s.mhd:
        _vz_ = s._vz_
        Bx = U[s._bx_]
        By = U[s._by_]
        Bz = U[s._bz_]
        dxBy = dxU[s._by_]
        dxBz = dxU[s._bz_]
        dyBx = dyU[s._bx_]
        dyBz = dyU[s._bz_]
    else:
        Bx = 0
        By = 0
        Bz = 0
        dxBy = 0
        dxBz = 0
        dyBx = 0
        dyBz = 0
    if s.fv_slopes == "primitives":
        if s.well_balance:
            s.dm.dMt[0]   = - (U[_vx_]*(dxU[0]  +dxUeq[0]  ) +         U[0]*dxU[_vx_]) - (U[_vy_]*(dyU[0]+dyUeq[0])     +         U[0]*dyU[_vy_])
            # s.dm.dMt[_p_] = - (U[_vx_]*(dxU[_p_]+dxUeq[_p_]) + gamma*U[_p_]*dxU[_vx_]) - (U[_vy_]*(dyU[_p_]+dyUeq[_p_]) + gamma*U[_p_]*dyU[_vy_])
            # s.dm.dMt[_s_] = -U[_vx_]*(dxU[_s_]+dxUeq[_s_]) - U[_vy_]*(dyU[_s_]+dyUeq[_s_])
        else:
             s.dm.dMt[0]   = - (U[_vx_]*dxU[0]   +         U[0]*dxU[_vx_]) - (U[_vy_]*dyU[0]   +         U[0]*dyU[_vy_])
            #  s.dm.dMt[_p_] = - (U[_vx_]*dxU[_p_] + gamma*U[_p_]*dxU[_vx_]) - (U[_vy_]*dyU[_p_] + gamma*U[_p_]*dyU[_vy_])
            #  s.dm.dMt[_s_] = -U[_vx_]*dxU[_s_] - U[_vy_]*dyU[_s_]
        # s.dm.dMt[_vx_] = -(U[_vx_]*dxU[_vx_]+(dxU[_p_]+By*dxBy+Bz*dxBz)/U[0]) - (U[_vy_]*dyU[_vx_]-By*dyBx/U[0])
        # s.dm.dMt[_vy_] = -(U[_vx_]*dxU[_vy_]-Bx*dxBy/U[0]) - (U[_vy_]*dyU[_vy_]+(dyU[_p_]+Bx*dyBx+Bz*dyBz)/U[0])
        if s.mhd:
            # s.dm.dMt[3] =  -(U[_vx_]*dxU[_vz_]-Bx*dxBz/U[0]) - (U[_vy_]*dyU[_vz_]-By*dyBz/U[0])
            # s.dm.dMt[5] = 0
            # s.dm.dMt[6] = 0
            # s.dm.dMt[7] = -(U[_vx_]*dxBz+Bz*dxU[_vx_]-Bx*dxU[_vz_]) - (U[_vy_]*dyBz+Bz*dyU[_vy_]-By*dyU[_vz_])

    else:
        s.dm.dMt[0] = -dxU[1]-dyU[2]
        # s.dm.dMt[1] = (-dxU[0]*(0.5*(gamma-3)*U[1]**2+(1-gamma)*U[2]**2)/U[0]**2
        #                   -dxU[1]*(3-gamma)*U[1]/U[0]
        #                   -dxU[2]*2*(gamma-1)*U[2]/U[0]
        #                   -dxU[3]*(gamma-1)
        #                   +dyU[0]*U[1]*U[2]/U[0]**2
        #                   -dyU[1]*U[2]/U[0]
        #                   -dyU[2]*U[1]/U[0])
        # s.dm.dMt[2] = (-dyU[0]*(0.5*(gamma-3)*U[2]**2+(1-gamma)*U[1]**2)/U[0]**2
        #                   -dyU[1]*2*(gamma-1)*U[1]/U[0]
        #                   -dyU[2]*(3-gamma)*U[2]/U[0]
        #                   -dyU[3]*(gamma-1)
        #                   +dxU[0]*U[1]*U[2]/U[0]**2
        #                   -dxU[1]*U[2]/U[0]
        #                   -dxU[2]*U[1]/U[0])
        # s.dm.dMt[3] = (-dxU[0]*U[1]*(-gamma*U[3]*U[0] + (gamma-1)*(U[1]**2+U[2]**2))/U[0]**3
        #                   -dxU[1]*(gamma*U[3]/U[0] - 0.5*(gamma-1)*(3*U[1]**2+U[2]**2)/U[0]**2)
        #                   -dxU[2]*(1-gamma)*U[1]*U[2]/U[0]**2
        #                   -dxU[3]*gamma*U[1]/U[0]
        #                   -dyU[0]*U[2]*(-gamma*U[3]*U[0] + (gamma-1)*(U[1]**2+U[2]**2))/U[0]**3
        #                   -dyU[1]*(1-gamma)*U[1]*U[2]/U[0]**2
        #                   -dyU[2]*(gamma*U[3]/U[0] - 0.5*(gamma-1)*(U[1]**2+3*U[2]**2)/U[0]**2)
        #                   -dyU[3]*gamma*U[2]/U[0])

def minmod(SlopeL,SlopeR):
    #First compute ratio between slopes SlopeR/SlopeL
    #Then limit the ratio to be lower than 1
    #Finally, limit the ratio be positive and multiply by SlopeL to get the limited slope at the cell center
    #We use where instead of maximum/minimum as it doesn't propagte the NaNs caused when SlopeL=0
    ratio = SlopeR/SlopeL
    ratio = np.where(ratio<1,ratio,1)
    return np.where(ratio>0,ratio,0)*SlopeL


def moncen(dU_L,dU_R,dx_L,dx_R,dx_M):
    #Compute central slope
    dU_C = (dx_L*dU_L + dx_R*dU_R)/(dx_L+dx_R)
    slope = np.minimum(np.abs(2*dU_L*dx_L/dx_M),np.abs(2*dU_R*dx_R/dx_M))
    slope = np.sign(dU_C)*np.minimum(slope,np.abs(dU_C))
    return np.where(dU_L*dU_R>=0,slope,0)

def compute_slopes_x(s,dU):
    na = np.newaxis
    if s.slope_limiter == "minmod":
        return minmod(dU[:,:,:-1],dU[:,:,1:])

    elif s.slope_limiter == "moncen":
        dx   = (s.dm.x_cv[1:]-s.dm.x_cv[:-1])[na,na,:]
        dx_M = (s.dm.x_fp[2:-1]-s.dm.x_fp[1:-2])[na,na,:]
        return moncen(dU[:,:,:-1],dU[:,:,1: ],dx[:,:,:-1],dx[:,:,1: ],dx_M)

def compute_slopes_y(s,dU):
    na = np.newaxis
    if s.slope_limiter == "minmod":
        return minmod(dU[:,:-1,:],dU[:,1:,:])

    elif s.slope_limiter == "moncen":
        dy   = (s.dm.y_cv[1:]-s.dm.y_cv[:-1])[na,:,na]
        dy_M = (s.dm.y_fp[2:-1]-s.dm.y_fp[1:-2])[na,:,na]
        return moncen(dU[:,:-1,:],dU[:,1: ,:],dy[:,:-1,:],dy[:,1: ,:],dy_M)


def compute_second_order_fluxes(s: "simulator", m: int):
    na = s.dm.xp.newaxis
    smallp = s.min_P#s.min_rho*s.min_c2/s.gamma
    smallr = s.min_rho
    _p_ = s._p_
    prims = s.fv_slopes == "primitives"
    s.dm.M_fv[...]  = 0
    if prims:
        M_cv = s.dm.W_cv
    else:
        M_cv = s.dm.U_cv

    s.dm.M_fv[:,2:-2,2:-2] = M_cv

    if s.mhd:
        _bx_ = s._bx_
        _by_ = s._by_
        s.dm.Bx_fv[2:-2,1:-1] = s.dm.Bx_face_x[:,:]
        s.dm.By_fv[1:-1,2:-2] = s.dm.By_face_y[:,:]
        s.dm.M_fv[_bx_,2:-2,2:-2] = 0.5*(s.dm.Bx_face_x[:,1:]+s.dm.Bx_face_x[:,:-1])
        s.dm.M_fv[_by_,2:-2,2:-2] = 0.5*(s.dm.By_face_y[1:,:]+s.dm.By_face_y[:-1,:])

    #fv.FV_Boundaries(s,perturbation=True,primitives=prims,mhd=s.mhd)

    dx_cv = (s.dm.x_cv[1:]-s.dm.x_cv[:-1])
    dy_cv = (s.dm.y_cv[1:]-s.dm.y_cv[:-1])
    dx = (s.dm.x_fp[1: ]-s.dm.x_fp[:-1])
    dy = (s.dm.y_fp[1: ]-s.dm.y_fp[:-1])

    ########################
    # X-Direction
    ########################
    fv.FV_Boundaries_x(s,perturbation=True,primitives=prims)
    dM = (s.dm.M_fv[:,:,1:] - s.dm.M_fv[:,:,:-1])/dx_cv[na,na,:]
    s.dm.dMx[...] = compute_slopes_x(s,dM)
    Sx = 0.5*s.dm.dMx*dx[na,na,1:-1] #Slope_x*dx/2

    ########################
    # Y-Direction
    ########################
    fv.FV_Boundaries_y(s,perturbation=True,primitives=prims)
    dM = (s.dm.M_fv[:,1:,:] - s.dm.M_fv[:,:-1,:])/dy_cv[na,:,na]
    s.dm.dMy[...] = compute_slopes_y(s,dM)
    Sy = 0.5*s.dm.dMy*dy[na,1:-1,na] #Slope_y*dy/2

    if s.mhd:
        #Slopes at t^n for the Magnetic Field
        #this slopes are to be used when interpolating to corner points
        dxBy = (s.dm.By_fv[1:-1,1:] - s.dm.By_fv[1:-1,:-1])/dx_cv[na,:]
        dxBy = compute_slopes_x(s,dxBy[na,:,:])[0]
        dyBx = (s.dm.Bx_fv[1:,1:-1] - s.dm.Bx_fv[:-1,1:-1])/dy_cv[:,na]
        dyBx = compute_slopes_y(s,dyBx[na,:,:])[0]

    if s.use_predictor:
        dt = s.dt*s.dm.w_tp[m]
        if s.well_balance:
            #We move to the solution
            s.dm.M_fv += s.dm.M_eq_fv
            #dxMeq and dyMeq could be computed only once, but we are kind of limited by memory
            dxMeq = compute_slopes_x(s,(s.dm.M_eq_fv[:,:,1:] - s.dm.M_eq_fv[:,:,:-1])/dx_cv[na,na,:])
            dyMeq = compute_slopes_y(s,(s.dm.M_eq_fv[:,1:,:] - s.dm.M_eq_fv[:,:-1,:])/dy_cv[na,:,na])
            compute_prediction(s,s.dm.M_fv[:,1:-1,1:-1],s.dm.dMx[:,1:-1,:],s.dm.dMy[:,:,1:-1],dxMeq[:,1:-1,:],dyMeq[:,:,1:-1])
            if s.potential:
                s.dm.dMt[1] +=  ((s.dm.M_fv[0]-s.dm.M_eq_fv[0])/s.dm.M_fv[0]*s.dm.grad_phi_fv[0])[1:-1,1:-1]
                s.dm.dMt[2] +=  ((s.dm.M_fv[0]-s.dm.M_eq_fv[0])/s.dm.M_fv[0]*s.dm.grad_phi_fv[1])[1:-1,1:-1]
        else:
            compute_prediction(s,s.dm.M_fv[:,1:-1,1:-1],s.dm.dMx[:,1:-1,:],s.dm.dMy[:,:,1:-1])

        if s.mhd:
            #Corner point weighted averages
            Bx = ((s.dm.Bx_fv[1:,:]*dy[1:,na] + s.dm.Bx_fv[:-1,:]*dy[:-1,na])/(dy[1:,na]+dy[:-1,na]))
            By = ((s.dm.By_fv[:,1:]*dx[na,1:] + s.dm.By_fv[:,:-1]*dx[na,:-1])/(dx[na,1:]+dx[na,:-1]))
            vx = s.dm.M_fv[1]
            vx = (vx[:,1:]*dx[na,1:] + vx[:,:-1]*dx[na,:-1])/(dx[na,1:]+dx[na,:-1])
            vx = (vx[1:,:]*dy[1:,na] + vx[:-1,:]*dy[:-1,na])/(dy[1:,na]+dy[:-1,na])
            #vx = (vx[1:,1:] + vx[1:,:-1] + vx[:-1,1:] + vx[:-1,:-1])/4
            vy = s.dm.M_fv[2]
            vy = (vy[1:,:]*dy[1:,na] + vy[:-1,:]*dy[:-1,na])/(dy[1:,na]+dy[:-1,na])
            vy = (vy[:,1:]*dx[na,1:] + vy[:,:-1]*dx[na,:-1])/(dx[na,1:]+dx[na,:-1])
            #vy = (vy[1:,1:] + vy[1:,:-1] + vy[:-1,1:] + vy[:-1,:-1])/4
            Ez = vx*By - vy*Bx
            #We move (Bx,By)(t^n) ->(Bx,By)(t^n+1/2)
            s.dm.Bx_fv[1:-1,:] += 0.5*dt*(Ez[1: ,:]-Ez[:-1,:])/dy[1:-1,na]
            s.dm.By_fv[:,1:-1] -= 0.5*dt*(Ez[:,1: ]-Ez[:,:-1])/dx[na,1:-1]

        if s.well_balance:
            #We move back to the perturbation
            s.dm.M_fv -= s.dm.M_eq_fv
        #We move U(t^n) ->U(t^n+1/2)
        s.dm.M_fv[:,1:-1,1:-1] += 0.5*s.dm.dMt*dt


    #UR = U - SlopeC*dx/2, UL = U + SlopeC*dx/2
    s.dm.MR_face_x[...] = s.dm.M_fv[:,2:-2,2:-1] - Sx[:,2:-2,1: ]
    s.dm.ML_face_x[...] = s.dm.M_fv[:,2:-2,1:-2] + Sx[:,2:-2,:-1]
    s.dm.MR_face_y[...] = s.dm.M_fv[:,2:-1,2:-2] - Sy[:,1: ,2:-2]
    s.dm.ML_face_y[...] = s.dm.M_fv[:,1:-2,2:-2] + Sy[:,:-1,2:-2]

    if s.complex_BC:
        s.dm.MR_face_x[:,:,1:-1]  = np.where(s.dm.c_BC_fv[na,:,1: ]-s.dm.c_BC_fv[na,:,:-1]<0, s.dm.ML_face_x[:,:,1:-1], s.dm.MR_face_x[:,:,1:-1])
        s.dm.MR_face_x[1,:,1:-1] *= np.where(s.dm.c_BC_fv[   :,1: ]-s.dm.c_BC_fv[   :,:-1]<0,-1, 1)
        s.dm.ML_face_y[:,1:-1,:]  = np.where(s.dm.c_BC_fv[na,:-1,:]-s.dm.c_BC_fv[na,1:, :]<0, s.dm.MR_face_y[:,1:-1,:], s.dm.ML_face_y[:,1:-1,:])
        s.dm.ML_face_y[2,1:-1,:] *= np.where(s.dm.c_BC_fv[   :-1,:]-s.dm.c_BC_fv[   1:, :]<0,-1, 1)

    if s.well_balance:
        #Move to solution at interfaces
        s.dm.MR_face_x[...] += s.dm.M_eq_face_x
        s.dm.ML_face_x[...] += s.dm.M_eq_face_x
        s.dm.MR_face_y[...] += s.dm.M_eq_face_y
        s.dm.ML_face_y[...] += s.dm.M_eq_face_y

    if s.fv_safetynet and prims:
        M_re = [s.dm.MR_face_x,s.dm.ML_face_x,s.dm.MR_face_y,s.dm.ML_face_y]
        M_cv = [s.dm.M_fv[:,2:-2,2:-1],s.dm.M_fv[:,2:-2,1:-2],s.dm.M_fv[:,2:-1,2:-2],s.dm.M_fv[:,1:-2,2:-2]]
        for i in range(4):
            M_re[i][s._p_] = np.where(M_re[i][s._p_]>smallp,M_re[i][s._p_],smallp)
            M_re[i][0] = np.where(M_re[i][0]>smallr,M_re[i][0],M_cv[i][0])

    if s.mhd:
        fv.B_Boundaries(s)
        #We compute the cell centered values as the average of the face values, and then displace the values of the
        #transversal components using the slopes at t^n
        By_cv = 0.5*(s.dm.By_fv[1:-2,:]+s.dm.By_fv[2:-1,:])
        s.dm.MR_face_x[_by_,...] = By_cv[:,2:-1] - Sx[_by_,2:-2,1: ]
        s.dm.ML_face_x[_by_,...] = By_cv[:,1:-2] + Sx[_by_,2:-2,:-1]
        Bx_cv = 0.5*(s.dm.Bx_fv[:,1:-2]+s.dm.Bx_fv[:,2:-1])
        s.dm.MR_face_y[_bx_,...] = Bx_cv[2:-1,:] - Sy[_bx_,1: ,2:-2]
        s.dm.ML_face_y[_bx_,...] = Bx_cv[1:-2,:] + Sy[_bx_,:-1,2:-2]

        s.dm.MR_face_x[_bx_] = s.dm.ML_face_x[_bx_] = s.dm.Bx_fv[2:-2,1:-1]
        s.dm.MR_face_y[_by_] = s.dm.ML_face_y[_by_] = s.dm.By_fv[1:-1,2:-2]

    s.riemann_solver_FV(s, s.dm.ML_face_x, s.dm.MR_face_x, 1, 2,prims)
    s.riemann_solver_FV(s, s.dm.ML_face_y, s.dm.MR_face_y, 2, 1,prims)

    if s.well_balance:
        #We compute the perturbation over the flux for conservative variables
        s.dm.MR_face_x -= s.dm.F_eq_face_x
        s.dm.MR_face_y -= s.dm.F_eq_face_y
    if s.mhd:
        ###################################################
        #Now the interpolation to corner points to solve Ez
        ###################################################

        MBL = M[:, :-1, :-1] + Sx[:,1:-2,:-1] + Sy[:,:-1,1:-2]
        MTL = M[:,1:  , :-1] + Sx[:,2:-1,:-1] - Sy[:,1: ,1:-2]
        MBR = M[:, :-1,1:  ] - Sx[:,1:-2,1: ] + Sy[:,:-1,2:-1]
        MTR = M[:,1:  ,1:  ] - Sx[:,2:-1,1: ] - Sy[:,1: ,2:-1]
        if s.fv_safetynet and prims:
            M_re = [MBL,MTL,MBR,MTR]
            M_cv = [s.dm.M_fv[:,1:-2,1:-2],s.dm.M_fv[:,2:-1,1:-2],s.dm.M_fv[:,1:-2,2:-1],s.dm.M_fv[:,2:-1,2:-1]]
            for i in range(4):
                M_re[i][s._p_] = np.where(M_re[i][s._p_]>smallp,M_re[i][s._p_],smallp)
                M_re[i][0] = np.where(M_re[i][0]>smallr,M_re[i][0],M_cv[i][0])

        Sx = 0.5*dxBy*dx[na,1:-1]
        ByR = s.dm.By_fv[1:-1,2:-1] - Sx[:,1: ]
        ByL = s.dm.By_fv[1:-1,1:-2] + Sx[:,:-1]
        Sy = 0.5*dyBx*dy[1:-1,na]
        BxT = s.dm.Bx_fv[2:-1,1:-1] - Sy[1: ,:]
        BxB = s.dm.Bx_fv[1:-2,1:-1] + Sy[:-1,:]

        s.riemann_solver_FV_E(s,MBL,MTL,MBR,MTR,BxB,BxT,ByL,ByR,s.dm.Es,primitives=prims)
