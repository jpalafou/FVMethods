from mimetypes import init
from pickle import NEXT_BUFFER
from typing import Callable
from typing import Union

from pkg_resources import parse_requirements
from utils.sd_visualize import display_field

import numpy as np
import cupy as cp

from timeit import default_timer as timer
from data_management import CupyLocation
from data_management import GPUDataManager
from polynomials import gauss_legendre_quadrature
from polynomials import solution_points
from polynomials import lagrange_matrix
from polynomials import lagrangeprime_matrix
from polynomials import intfromsol_matrix
from polynomials import ader_matrix
from polynomials import quadrature_mean
import hydro
import muscl
import cfl
import finite_volume as fv
import riemann_solver as rs
from sd_ader import ader_predictor, ader_update, solve_faces
from sd_ader import solve_corners
from initial_conditions_2d import step_function
from initial_conditions_2d import orszag_tang_pot
from potential import phi
from well_ballance import init_equilibrium_state
from well_ballance import sponge_layers
from well_ballance import heating_fct, init_heating

import trouble_detection as td

import matplotlib.pyplot as plt

class SD_Simulator:
    def __init__(
        self,
        init_fct: Callable[[np.ndarray, int], np.ndarray] = step_function,
        equilibrium_fct: Callable[[np.ndarray, int], np.ndarray] = step_function,
        potential_fct: Callable[[np.ndarray], np.ndarray] = phi,
        vectorpot_fct: Callable[[np.ndarray], np.ndarray] = orszag_tang_pot,
        n: int = 8, 
        Nx: int = 32,
        Ny: int = 32,
        boxlen_x: float = 1,
        boxlen_y: float = 1,
        riemann_solver_SD: Callable = rs.solve_riemann_llf,
        riemann_solver_FV: Callable = rs.solve_riemann_hllc,
        riemann_solver_SD_E_1D: Callable = rs.solve_riemann_1D_llf_E,
        riemann_solver_SD_E_2D: Callable = rs.solve_riemann_2D_llf_E,
        riemann_solver_FV_E: Callable = rs.solve_riemann_hlld_E,
        riemann_in_picard: bool = True,
        cfl_coeff: float = 0.8,
        gamma: float = 1.4,
        use_cupy: bool = True,
        well_balance: bool = False,
        detect_over_perturbations: bool = True,
        potential: bool = False,
        modify_internal_energy: bool = False,
        entropy_fix: bool = False,
        BC: list = ["periodic","periodic"],
        max_rho: float = 1E10,
        min_rho: float = 1E-10,
        min_c: float = 1E-10,
        min_c2: float = 1E-10,
        min_P: float = 1E-10,
        detect_troubles: bool = False,
        PAD: bool = False,
        NAD: str = "1st",
        so_godunov: bool = False,
        SD_update: bool = False,
        advection: bool = False,
        mhd: bool = False,
        limiting_vars: list = [0,3],
        detect_on_B: bool = False,
        tolerance_ptge: float = 1E-8,
        eps: float = 1E-14,
        outputs: str = "outputs",
        timer: bool = True,
        substep_troubles: bool = False,
        sd_slopes: str = "conservatives",
        fv_slopes: str = "primitives",
        use_predictor: bool = True,
        slope_limiter: str = "minmod",
        plot=False,
        fv_safetynet: bool = False,
        sponge: Callable = sponge_layers,
        apply_sponge: bool = False,
        heating: bool = False,
        heating_fct:  Callable = heating_fct,
        complex_BC: bool = False,
    ):
        self.n = n
        self.Nx = Nx
        self.Ny = Ny
        self.gamma = gamma
        self.init_fct = init_fct
        self.equilibrium_fct = equilibrium_fct
        self.potential_fct = potential_fct
        self.vectorpot_fct = vectorpot_fct
        self.dm = GPUDataManager(use_cupy)
        
        self.dimension = 2
        self.boxlen_x = boxlen_x
        self.boxlen_y = boxlen_y
        self.cfl_coeff=cfl_coeff
        self.gamma=gamma
        self.time = 0.0
        self.well_balance = well_balance
        self.detect_over_perturbations = detect_over_perturbations
        self.potential = potential
        self.riemann_solver_SD = riemann_solver_SD
        self.riemann_solver_FV = riemann_solver_FV
        self.riemann_solver_SD_E_1D  = riemann_solver_SD_E_1D
        self.riemann_solver_SD_E_2D  = riemann_solver_SD_E_2D
        self.riemann_solver_FV_E = riemann_solver_FV_E
        self.riemann_in_picard = riemann_in_picard
        self.modify_internal_energy = modify_internal_energy
        self.entropy_fix = entropy_fix
        self.BC = BC
        self.max_rho=max_rho
        self.min_rho=min_rho
        self.min_c  =min_c
        self.min_c2 =min_c2
        self.min_P  =min_P
        self.detect_troubles = detect_troubles
        self.detect_on_B = detect_on_B
        self.PAD = PAD
        self.NAD = NAD
        self.so_godunov = so_godunov
        self.SD_update = SD_update
        self.advection = advection
        self.mhd = mhd
        self.limiting_vars = limiting_vars
        self.tolerance_ptge = tolerance_ptge
        self.eps = eps
        self.timer = timer
        self.execution_time = 0
        self.outputs = outputs
        self.substep_troubles = substep_troubles
        self.fv_slopes = fv_slopes
        self.sd_slopes = sd_slopes
        self.use_predictor = use_predictor
        self.slope_limiter = slope_limiter
        self.fv_safetynet = fv_safetynet
        self.sponge = sponge
        self.apply_sponge = apply_sponge
        self.heating = heating
        self.heating_fct = heating_fct
        self.complex_BC = complex_BC
        
        self.dx = boxlen_x/self.Nx
        self.dy = boxlen_y/self.Ny
        self.n_step = 0
        self.plot = plot
        nvar = 2 + self.dimension
        self._d_  = 0
        self._vx_ = 1
        self._vy_ = 2
        if self.mhd:
            nvar+= 4 #vz, Bx,By,Bz
            self._vz_ = 3
            self._p_  = 4
            self._bx_ = 5
            self._by_ = 6
            self._bz_ = 7
        else:
            self._p_  = 3
        self._s_ = nvar
        nvar+=1
        self.nvar = nvar
        self.scheme = "sd"

        # Use 1.0 instead of dx or dt for precision; rescaling happens in ADER iterations. (*)
        self.x, self.w = gauss_legendre_quadrature(0.0, 1.0, self.n)
        self.x_tp, self.dm.w_tp = gauss_legendre_quadrature(0.0, 1.0, self.n + 1)

        # (*)
        self.x_fp = np.hstack((0.0, self.x, 1.0))
        self.x_sp = solution_points(0.0, 1.0, self.n)

        # Values at flux pts from values at sol pts.
        self.dm.val_fp_from_val_sp = lagrange_matrix(self.x_fp, self.x_sp)
        self.dm.val_sp_from_val_fp = lagrange_matrix(self.x_sp, self.x_fp)
        # Spatial derivative of the flux at sol pts from density at flux pts.
        # Note: uses the fact that we have a linear flux/constant velocity.
        self.dm.deriv_sp_from_val_fp = lagrangeprime_matrix(self.x_sp, self.x_fp)
        # Mean values in control volumes from values at sol pts.
        self.val_cv_from_val_sp = intfromsol_matrix(self.x_sp, self.x_fp)
        self.val_sp_from_val_cv = np.linalg.inv(self.val_cv_from_val_sp)
        
        # ADER matrix.
        # (*)
        self.ader = ader_matrix(self.x_tp, self.dm.w_tp, 1.0)
        # Yes, "invader" is a good pun, but it does not match the naming scheme...
        self.dm.ader_inv = np.linalg.inv(self.ader)
        
        na =  np.newaxis
        self.mesh_cv = np.ndarray((2, Ny+2, Nx+2, n+2, n+2))
        self.mesh_cv[0] = (np.arange(Nx+2)[na, :, na, na] + self.x_fp[na, na, na, :])*(boxlen_x+2*self.dx)/(Nx+2)-self.dx
        self.mesh_cv[1] = (np.arange(Ny+2)[:, na, na, na] + self.x_fp[na, na, :, na])*(boxlen_y+2*self.dy)/(Ny+2)-self.dy

        x_sp = (np.arange(Nx)[:, na] + self.x_sp[na, :])*(boxlen_x)/(Nx)
        y_sp = (np.arange(Ny)[:, na] + self.x_sp[na, :])*(boxlen_y)/(Ny)
        self.dm.x_sp = x_sp.reshape(Nx,n+1)
        self.dm.y_sp = y_sp.reshape(Ny,n+1)
        self.mesh_fv = np.ndarray((2, Ny*(n+1), Nx*(n+1)))
        x,y = np.meshgrid(x_sp,y_sp)
        self.mesh_fv[0] = x
        self.mesh_fv[1] = y
        
        self.dm.val_cv_from_val_sp = self.val_cv_from_val_sp
        self.dm.val_sp_from_val_cv = self.val_sp_from_val_cv
        
        # Reserve the space but let it uninitialized for now.
        # Axes:
        #   0: ADER substeps
        #   1: variables
        #   2,3: cells (y,x)
        #   4,5: pts inside cells (y,x)
        self.dm.U_ader_sp = np.ndarray((nvar, n+1, Ny, Nx, n+1, n+1))
        #Conservative/Primitive varibles at flux points
        self.dm.M_ader_fp_x = np.ndarray((nvar, n+1, Ny, Nx, n+1, n+2))
        self.dm.M_ader_fp_y = np.ndarray((nvar, n+1, Ny, Nx, n+2, n+1))
        #Conservative fluxes at flux points
        self.dm.F_ader_fp_x = np.ndarray((nvar, n+1, Ny, Nx, n+1, n+2))
        self.dm.F_ader_fp_y = np.ndarray((nvar, n+1, Ny, Nx, n+2, n+1))
        
        #Arrays to be used by the Riemann Solver
        self.dm.ML_fp_x = np.ndarray(( nvar, n+1, Ny, Nx+1, n+1))
        self.dm.MR_fp_x = np.ndarray(( nvar, n+1, Ny, Nx+1, n+1))
        self.dm.ML_fp_y = np.ndarray(( nvar, n+1, Ny+1, Nx, n+1))
        self.dm.MR_fp_y = np.ndarray(( nvar, n+1, Ny+1, Nx, n+1))
        
        #Arrays to store and impose boundary conditions
        self.dm.BC_fp_x = np.ndarray((2, nvar, n+1, Ny, n+1))
        self.dm.BC_fp_y = np.ndarray((2, nvar, n+1, Nx, n+1))
        
        #Array for troubled cell detection
        self.n_troubles = 0
        self.dm.trouble = np.zeros((Ny*(n+1), Nx*(n+1)))
        self.dm.affected_face_x = np.zeros((Ny*(n+1)  , Nx*(n+1)+1))
        self.dm.affected_face_y = np.zeros((Ny*(n+1)+1, Nx*(n+1)  ))
        if self.mhd:
            self.dm.affected_corner = np.zeros((Ny*(n+1)+1, Nx*(n+1)+1))
        if self.substep_troubles:
            self.dm.step_trouble = np.zeros((Ny*(n+1), Nx*(n+1)))
        #Arrays for finitive volume fallback scheme     
        self.dm.U_new = np.ndarray((nvar, Ny*(n+1), Nx*(n+1)))
        self.dm.F_face_x = np.ndarray([nvar,Ny*(n+1)  ,(Nx*(n+1))+1])
        self.dm.F_face_y = np.ndarray([nvar,Ny*(n+1)+1,(Nx*(n+1))])
        self.dm.M_fv  = np.ndarray((nvar, Ny*(n+1)+4, Nx*(n+1)+4))
        self.dm.dMx = np.ndarray([nvar,(Ny*(n+1))+4,(Nx*(n+1))+2])
        self.dm.dMy = np.ndarray([nvar,(Ny*(n+1))+2,(Nx*(n+1))+4])
        self.dm.dMt = np.ndarray([nvar,(Ny*(n+1))+2,(Nx*(n+1))+2])
        self.dm.MR_face_x = np.ndarray([nvar,Ny*(n+1)  ,(Nx*(n+1))+1])
        self.dm.ML_face_x = np.ndarray([nvar,Ny*(n+1)  ,(Nx*(n+1))+1])
        self.dm.MR_face_y = np.ndarray([nvar,Ny*(n+1)+1,(Nx*(n+1))])
        self.dm.ML_face_y = np.ndarray([nvar,Ny*(n+1)+1,(Nx*(n+1))])   
        #Boundary conditions for the FV scheme
        self.dm.BC_x = np.ndarray((nvar, 2, Ny*(n+1)+4, 2))
        self.dm.BC_y = np.ndarray((nvar, 2, 2, Nx*(n+1)+4))

        #1-D array storing the position of interfaces
        self.dm.x_fp = np.ndarray((Nx*(n+1)+5))
        self.dm.x_fp[2:-2] = boxlen_x/Nx*np.hstack((np.arange(Nx).repeat(n+1)+np.tile(self.x_fp[:-1],Nx),Nx))
        self.dm.x_fp[0:2]  = -self.dm.x_fp[3:5][::-1]
        self.dm.x_fp[-2:]  = self.dm.x_fp[-3]+self.dm.x_fp[3:5]
        self.dm.y_fp = np.ndarray((Ny*(n+1)+5))
        self.dm.y_fp[2:-2] = boxlen_y/Ny*np.hstack((np.arange(Ny).repeat(n+1)+np.tile(self.x_fp[:-1],Ny),Ny))
        self.dm.y_fp[0:2]  = -self.dm.y_fp[3:5][::-1]
        self.dm.y_fp[-2:]  = self.dm.y_fp[-3]+self.dm.y_fp[3:5]
        #1-D array storing the position at cell centers     
        self.dm.x_cv = 0.5*(self.dm.x_fp[1:]+self.dm.x_fp[:-1])
        self.dm.y_cv = 0.5*(self.dm.y_fp[1:]+self.dm.y_fp[:-1])
        
        if self.well_balance:
            self.dm.U_eq_sp = np.ndarray((nvar, Ny, Nx, n+1, n+1))
            self.dm.W_eq_sp = np.ndarray((nvar, Ny, Nx, n+1, n+1))
            self.dm.U_eq_cv = np.ndarray((nvar, Ny, Nx, n+1, n+1))
            self.dm.W_eq_cv = np.ndarray((nvar, Ny, Nx, n+1, n+1))
            self.dm.M_eq_fp_x = np.ndarray((nvar, Ny, Nx, n+1, n+2))
            self.dm.M_eq_fp_y = np.ndarray((nvar, Ny, Nx, n+2, n+1))
            self.dm.F_eq_fp_x = np.ndarray((nvar, Ny, Nx, n+1, n+2))
            self.dm.F_eq_fp_y = np.ndarray((nvar, Ny, Nx, n+2, n+1))
            self.dm.M_eq_fv = np.ndarray((nvar, Ny*(n+1), Nx*(n+1)))
            init_equilibrium_state(self)
            
        if self.potential:
            self.dm.grad_phi_sp = np.ndarray((2, Ny, Nx, n+1, n+1))
            self.dm.grad_phi_fv = np.ndarray((2, Ny*(n+1)+4, Nx*(n+1)+4))
            self.init_potential()
        
        if self.heating:
            init_heating(self)

        if self.mhd:
            self.dm.Ez_ader_cp   = np.ndarray((n+1, Ny, Nx, n+2, n+2))
            self.dm.WL_cp   = np.ndarray((nvar,n+1, Ny+1, Nx+1, n+2))
            self.dm.WR_cp   = np.ndarray((nvar,n+1, Ny+1, Nx+1, n+2))
            self.dm.EL_cp   = np.ndarray((n+1, Ny+1, Nx+1, n+2))
            self.dm.ER_cp   = np.ndarray((n+1, Ny+1, Nx+1, n+2))
            self.dm.BC_cp_x = np.ndarray((2, nvar, n+1, Ny, n+2))
            self.dm.BC_cp_y = np.ndarray((2, nvar, n+1, Nx, n+2))

            self.dm.W_RT_cp = np.ndarray(( nvar, n+1, Ny+1, Nx+1))
            self.dm.W_RB_cp = np.ndarray(( nvar, n+1, Ny+1, Nx+1))
            self.dm.W_LT_cp = np.ndarray(( nvar, n+1, Ny+1, Nx+1))
            self.dm.W_LB_cp = np.ndarray(( nvar, n+1, Ny+1, Nx+1))
            self.dm.BC_cp_xy= np.ndarray((2, nvar, n+1, 2))
            self.dm.E_cp = np.ndarray((n+1, Ny+1, Nx+1))

            self.dm.Ez  = np.ndarray([(Ny*(n+1))+1,(Nx*(n+1))+1])
            self.dm.Bx_face_x = np.ndarray([(Ny*(n+1)),(Nx*(n+1))+1])
            self.dm.By_face_y = np.ndarray([(Ny*(n+1))+1,(Nx*(n+1))])

            self.dm.Bx_new = np.ndarray([(Ny*(n+1)),(Nx*(n+1))+1])
            self.dm.By_new = np.ndarray([(Ny*(n+1))+1,(Nx*(n+1))])
            self.dm.Es  = np.ndarray([(Ny*(n+1))+1,(Nx*(n+1))+1])

            self.dm.Bx_fv = np.ndarray([(Ny*(n+1))+4,(Nx*(n+1))+3])
            self.dm.By_fv = np.ndarray([(Ny*(n+1))+3,(Nx*(n+1))+4])
            self.dm.BC_Bx_x = np.ndarray([2,Ny*(n+1)+4])
            self.dm.BC_Bx_y = np.ndarray([2,2,Nx*(n+1)+3])
            self.dm.BC_By_x = np.ndarray([2,Ny*(n+1)+3,2])
            self.dm.BC_By_y = np.ndarray([2,Nx*(n+1)+4])

        self.dm.S_sp = np.zeros((nvar, Ny, Nx, n+1, n+1))
        self.dm.S_fv = np.zeros((nvar, Ny*(n+1), Nx*(n+1)))
        self.post_init()
        cfl.compute_dt(self)
        #print(self.dt)

    def compute_primitives(self,U):
        W = U.copy()
        W[self._vx_] = U[self._vx_]/U[0]
        W[self._vy_] = U[self._vy_]/U[0]
        if self.mhd:
            W[self._vz_] = U[self._vz_]/U[0]
            B2 = 0.5*(U[self._bx_]**2 + U[self._by_]**2 + U[self._bz_]**2)
            K  = 0.5*U[0]*(W[self._vx_]**2 + W[self._vy_]**2 + W[self._vz_]**2)
        else:
            B2 = 0 
            K  = 0.5*U[0]*(W[self._vx_]**2 + W[self._vy_]**2)
        W[self._s_] = U[self._s_]/U[0]
        if self.entropy_fix:
            W[self._p_] = W[self._s_]*W[0]**self.gamma
        else:
            W[self._p_] = (self.gamma-1)*(U[self._p_]-(K+B2))
        
        return W
                
    def compute_conservatives(self,W):
        U = W.copy()
        U[self._vx_] = W[0]*W[self._vx_]
        U[self._vy_] = W[0]*W[self._vy_]
        if self.mhd:
            U[self._vz_] = W[0]*W[self._vz_]
            B2 = W[self._bx_]**2 + W[self._by_]**2 + W[self._bz_]**2
            K  = U[0]*(W[self._vx_]**2 + W[self._vy_]**2 + W[self._vz_]**2)
        else:
            B2 = 0 
            K  = U[0]*(W[self._vx_]**2 + W[self._vy_]**2)
        U[self._p_] = W[self._p_]/(self.gamma-1)+0.5*(K+B2)
        U[self._s_] = W[self._s_]*W[0]
        return U
    
    def compute_fluxes(self,F,M,v_1,v_2,prims) -> None:
        _p_ = self._p_
        _s_ = self._s_
        if prims:
            W = M
        else:
            W = self.compute_primitives(M)
        m_1 = W[0]*W[v_1]
        m_2 = W[0]*W[v_2]
        if self.mhd:
            v_3 = self._vz_
            b_3 = self._bz_
            m_3 = W[0]*W[v_3]
            K = m_1*W[v_1] + m_2*W[v_2] + m_3*W[v_3]
            B_1 = M[v_1+(self._bx_-1)]
            B_2 = M[v_2+(self._bx_-1)]
            B_3 = M[self._bz_]
            v3  = W[v_3]
        else:
            K = m_1*W[v_1] + m_2*W[v_2]
            B_1=0
            B_2=0
            B_3=0
            v3 =0

        B2  = B_1**2 + B_2**2 + B_3**2
        p   = W[_p_]
        pT  = p + 0.5*B2 #Total pressure
        E   = p/(self.gamma-1) + 0.5*(K+B2)
        F[0  ,...] = m_1
        F[v_1,...] = m_1*W[v_1] + pT - B_1**2
        F[v_2,...] = m_2*W[v_1] - B_1*B_2
        F[_p_,...] = W[v_1]*(E + pT) - B_1*(B_1*W[v_1]+B_2*W[v_2]+B_3*v3)
        F[_s_,...] = W[v_1]*W[0]*W[_s_]
        if self.mhd:
            F[v_3,...]  = m_3*W[v_1] - B_1*B_3
            F[b_3,...]  = W[v_1]*B_3 - v3*B_1
        
        
    def post_init(self) -> None:
        na = np.newaxis
        nvar =self.nvar
        W_gh = np.ndarray((nvar,self.Ny+2,self.Nx+2,self.n+1,self.n+1))
        #This arrays contain two layers of ghost elements
        for i in range(nvar):
            W_gh[i] = quadrature_mean(self.mesh_cv,self.init_fct,2,i)
        if self.complex_BC:
            #self.dm.c_BC_sp = (quadrature_mean(self.mesh_cv,self.init_fct,2,-1))[1:-1,1:-1,:,:]
            self.dm.c_BC_fv = self.init_fct(self.mesh_fv,-1)
            self.dm.c_BC_sp = np.transpose(self.dm.c_BC_fv.reshape(self.Ny,self.n+1,self.Nx,self.n+1),(0,2,1,3))
            self.dm.c_BC = self.dm.c_BC_sp[:,:,0,0]
            #self.dm.c_BC_fv = np.transpose(self.dm.c_BC_sp,(0,2,1,3)).reshape((self.Ny)*(self.n+1),(self.Nx)*(self.n+1))
        #Let's compute here the entropy
        W_gh[self._s_] = W_gh[self._p_]/W_gh[0]**self.gamma

        if self.mhd:
            _bx_ = self._bx_
            _by_ = self._by_
            #Initialization of magnetic field
            Az_gh = self.vectorpot_fct(self.mesh_cv,0)
            self.A_cp = Az_gh[1:-1,1:-1,:,:]
            # A(y^f,x^f) -> By(x^f,x^s), Bx(y^s,x^f)
            Bx_fp_x = np.einsum("sf,ijfx->ijsx", self.dm.deriv_sp_from_val_fp, Az_gh)/self.dy
            By_fp_y =-np.einsum("sf,ijxf->ijxs", self.dm.deriv_sp_from_val_fp, Az_gh)/self.dx
            self.By_init_fp_y = By_fp_y[1:-1,1:-1,:,:]
            self.Bx_init_fp_x = Bx_fp_x[1:-1,1:-1,:,:]
            self.dm.Bx_fp_x = self.Bx_init_fp_x.copy()
            self.dm.By_fp_y = self.By_init_fp_y.copy()
            Bx_sp = self.compute_sp_from_fp(Bx_fp_x[na],0)[0]
            By_sp = self.compute_sp_from_fp(By_fp_y[na],1)[0]
            Bx_cv = self.compute_cv_from_sp(self.compute_sp_from_fp(Bx_fp_x[na],0))[0]
            By_cv = self.compute_cv_from_sp(self.compute_sp_from_fp(By_fp_y[na],1))[0]
            W_gh[_bx_] = Bx_cv
            W_gh[_by_] = By_cv
           
            Bx = np.einsum("mk,ijkn->ijmn",self.dm.val_cv_from_val_sp,Bx_fp_x)
            Bx_fv = np.ndarray(((self.Ny+2)*(self.n+1),(self.Nx+2)*(self.n+1)+1))
            Bx_fv[:,:-1] = np.transpose(Bx[:,:,:,:-1],(0,2,1,3)).reshape((self.Ny+2)*(self.n+1),(self.Nx+2)*(self.n+1))
            Bx_fv[:, -1] = Bx[:,-1,:,-1].reshape((self.Ny+2)*(self.n+1))

            By = np.einsum("mk,ijnk->ijnm",self.dm.val_cv_from_val_sp,By_fp_y)
            By_fv = np.ndarray(((self.Ny+2)*(self.n+1)+1,(self.Nx+2)*(self.n+1)))
            By_fv[:-1,:] = np.transpose(By[:,:,:-1,:],(0,2,1,3)).reshape((self.Ny+2)*(self.n+1),(self.Nx+2)*(self.n+1))
            By_fv[ -1,:] = By[-1,:,-1,:].reshape((self.Nx+2)*(self.n+1))
            self.init_FV_B_BC(Bx_fv,By_fv)

        self.W_init_cv = W_gh[:,1:-1,1:-1,:,:]
        self.W_cv = self.W_init_cv.copy()
        
        if self.sd_slopes=="primitives":
            W_sp = self.compute_sp_from_cv(W_gh)
            if self.mhd:
                W_sp[_bx_] = Bx_sp
                W_sp[_by_] = By_sp
            U_sp = self.compute_conservatives(W_sp)
            U_gh = self.compute_cv_from_sp(U_sp)
            M_fp_x = self.compute_fp_from_sp(W_sp,0)
            M_fp_y = self.compute_fp_from_sp(W_sp,1)
        elif self.sd_slopes=="conservatives":
            U_gh = self.compute_conservatives(W_gh)
            U_sp = self.compute_sp_from_cv(U_gh)
            if self.mhd:
                U_sp[_bx_] = Bx_sp
                U_sp[_by_] = By_sp
            W_sp = self.compute_primitives(U_sp)
            M_fp_x = self.compute_fp_from_sp(U_sp,0)
            M_fp_y = self.compute_fp_from_sp(U_sp,1)
        else:
            print("Unknown variables to interpolate")

        #This is necessary when the BCs are the ICs
        self.dm.BC_fp_x[0] =  M_fp_x[:,np.newaxis,1:-1, 0,:,-1]
        self.dm.BC_fp_x[1] =  M_fp_x[:,np.newaxis,1:-1,-1,:, 0]        
        self.dm.BC_fp_y[0] =  M_fp_y[:,np.newaxis, 0,1:-1,-1,:]
        self.dm.BC_fp_y[1] =  M_fp_y[:,np.newaxis,-1,1:-1, 0,:]
        if self.mhd:
            M_cp = self.compute_fp_from_sp(M_fp_x,1)
            self.dm.BC_cp_x[0] = M_cp[:,np.newaxis,1:-1, 0,:,-1]
            self.dm.BC_cp_x[1] = M_cp[:,np.newaxis,1:-1,-1,:, 0]        
            self.dm.BC_cp_y[0] = M_cp[:,np.newaxis, 0,1:-1,-1,:]
            self.dm.BC_cp_y[1] = M_cp[:,np.newaxis,-1,1:-1, 0,:]
            self.dm.BC_cp_xy[0,...,0] = M_cp[:,np.newaxis, 0, 0,-1,-1]
            self.dm.BC_cp_xy[0,...,1] = M_cp[:,np.newaxis,-1, 0, 0,-1]
            self.dm.BC_cp_xy[1,...,0] = M_cp[:,np.newaxis, 0,-1,-1, 0]
            self.dm.BC_cp_xy[1,...,1] = M_cp[:,np.newaxis,-1,-1, 0, 0]
        #We crop the active elements
        self.dm.U_sp = U_sp[:,1:-1,1:-1,:,:]
        self.dm.W_sp = W_sp[:,1:-1,1:-1,:,:]

        self.dm.U_cv = U_gh[:,1:-1,1:-1,:,:]
        self.dm.W_cv = W_gh[:,1:-1,1:-1,:,:]

        if self.fv_slopes=="primitives":
            M_gh = self.dm.xp.transpose(W_gh,(0,1,3,2,4)).reshape(nvar,(self.Ny+2)*(self.n+1),(self.Nx+2)*(self.n+1))
        elif self.fv_slopes=="conservatives":
            M_gh = self.dm.xp.transpose(U_gh,(0,1,3,2,4)).reshape(nvar,(self.Ny+2)*(self.n+1),(self.Nx+2)*(self.n+1))
        else:
            print("Unknown variables to interpolate")

        self.init_FV_BC(M_gh)
        
        if self.advection:
            W = W_gh[:,1:-1,1:-1,:,:]
            W = self.dm.xp.transpose(W,(0,1,3,2,4)).reshape(nvar,(self.Ny)*(self.n+1),(self.Nx)*(self.n+1))
            self.dm.vx = W[self._vx_]
            self.dm.vy = W[self._vy_]
            self.dm.P  = W[self._p_]
        
    def init_potential(self) -> None:
        na = np.newaxis
        n = self.n
        Nx = self.Nx
        Ny = self.Ny
        phi_cv   = quadrature_mean(self.mesh_cv,self.potential_fct,2,0)
        phi_sp   = np.einsum("km,ln,ijmn->ijkl", self.val_sp_from_val_cv, self.val_sp_from_val_cv, phi_cv)
        
        phi_fp_x = np.einsum("fs,zcxs->zcxf", self.dm.val_fp_from_val_sp, phi_sp)
        self.dm.grad_phi_sp[0] = (np.einsum("sf,zcxf->zcxs",self.dm.deriv_sp_from_val_fp,phi_fp_x)/self.dx)[1:-1,1:-1,:,:]

        phi_fp_y = np.einsum("fs,zcsx->zcfx", self.dm.val_fp_from_val_sp, phi_sp)
        self.dm.grad_phi_sp[1] = (np.einsum("sf,zcfx->zcsx",self.dm.deriv_sp_from_val_fp,phi_fp_y)/self.dy)[1:-1,1:-1,:,:]
        
        #Now for the finite volume update
        phi_fp_x = self.dm.xp.einsum("mk,ijkn->ijmn",self.dm.val_cv_from_val_sp, phi_fp_x)
        phi_fp_y = self.dm.xp.einsum("mk,ijnk->ijnm",self.dm.val_cv_from_val_sp, phi_fp_y)
        
        phi_fv_x = np.ndarray((Ny*(n+1)+4,Nx*(n+1)+5))
        phi_fv_y = np.ndarray((Ny*(n+1)+5,Nx*(n+1)+4))

        phi_x = self.dm.xp.transpose(phi_fp_x[:,:,:,:-1],(0,2,1,3)).reshape((Ny+2)*(n+1),((Nx+2)*(n+1)))
        phi_y = self.dm.xp.transpose(phi_fp_y[:,:,:-1,:],(0,2,1,3)).reshape((Ny+2)*(n+1),((Nx+2)*(n+1)))

        if self.n >1:
            phi_fv_x[...,:,:-1] = phi_x[n-1:-n+1,n-1:-n+1]
            phi_fv_y[...,:-1,:] = phi_y[n-1:-n+1,n-1:-n+1]
            phi_fv_x[...,:, -1] = phi_fp_x[:,-1,:,-1].reshape((Ny+2)*(n+1))[n-1:-n+1]
            phi_fv_y[...,-1, :] = phi_fp_y[-1,:,-1,:].reshape((Nx+2)*(n+1))[n-1:-n+1]
        else:
            phi_fv_x[...,:,:-1] = phi_x
            phi_fv_y[...,:-1,:] = phi_y
            phi_fv_x[...,:, -1] = phi_fp_x[:,-1,:,-1].reshape((Ny+2)*(n+1))
            phi_fv_y[...,-1, :] = phi_fp_y[-1,:,-1,:].reshape((Nx+2)*(n+1))
        
        
        self.dm.grad_phi_fv[0] = (phi_fv_x[:,1:]-phi_fv_x[:,:-1])/(self.dm.x_fp[na,1: ]-self.dm.x_fp[na,:-1])
        self.dm.grad_phi_fv[1] = (phi_fv_y[1:,:]-phi_fv_y[:-1,:])/(self.dm.y_fp[1: ,na]-self.dm.y_fp[:-1,na])
                
    def compute_sp_from_cv(self,U_cv):
        # Axes labels:
        #   u: Conservative variables
        #   i,j: cells
        #   k,l: sol pts
        #   m,n: control volumes
        U_sp = np.einsum("km,ln,uijmn->uijkl", self.val_sp_from_val_cv, self.val_sp_from_val_cv, U_cv)
        return U_sp
    
    def compute_cv_from_sp(self,U_sp) -> None:
        # Axes labels:
        #   u: Conservative variables
        #   i,j: cells
        #   k,l: sol pts
        #   m,n: control volumes
        U_cv = np.einsum("mk,nl,uijkl->uijmn", self.val_cv_from_val_sp, self.val_cv_from_val_sp, U_sp)
        return U_cv
    
    def compute_fp_from_sp(self,U_sp,dim) -> None:
        if dim==0:
            U_fp = np.einsum("fs,uzcxs->uzcxf", self.dm.val_fp_from_val_sp, U_sp)
        elif dim==1:
            U_fp = np.einsum("fs,uzcsx->uzcfx", self.dm.val_fp_from_val_sp, U_sp)
        return U_fp
    
    def compute_sp_from_fp(self,U_fp,dim) -> None:
        if dim==0:
            U_sp = np.einsum("sf,uzcxf->uzcxs", self.dm.val_sp_from_val_fp, U_fp)
        elif dim==1:
            U_sp = np.einsum("sf,uzcfx->uzcsx", self.dm.val_sp_from_val_fp, U_fp)
        return U_sp

    
    def init_FV_BC(self, M_gh) -> None:
        na=np.newaxis
        n = self.n
        if n>1:
            self.dm.BC_x[:,0,...] = M_gh[:, n-1:-n+1, n-1: n+1]
            self.dm.BC_y[:,0,...] = M_gh[:, n-1: n+1, n-1:-n+1]
            self.dm.BC_x[:,1,...] = M_gh[:, n-1:-n+1,-n-1:-n+1]
            self.dm.BC_y[:,1,...] = M_gh[:,-n-1:-n+1, n-1:-n+1]
        else:
            self.dm.BC_x[:,0,...] = M_gh[:,    :   , n-1:n+1]
            self.dm.BC_y[:,0,...] = M_gh[:, n-1:n+1,    :   ]
            self.dm.BC_x[:,1,...] = M_gh[:,    :   ,-n-1:   ]
            self.dm.BC_y[:,1,...] = M_gh[:,-n-1:   ,    :   ]

    def init_FV_B_BC(self, B_x,B_y):
        na=np.newaxis
        n = self.n

        if n>1:
            self.dm.BC_Bx_x[0,...] = B_x[n-1:-n+1,n      ]
            self.dm.BC_By_x[0,...] = B_y[n  :-n  ,n-1:n+1]

            self.dm.BC_Bx_y[0,...] = B_x[n-1:n+1,n  :-n  ]
            self.dm.BC_By_y[0,...] = B_y[n      ,n-1:-n+1]

            self.dm.BC_Bx_x[1,...] = B_x[n-1:-n+1,-n-1     ]
            self.dm.BC_By_x[1,...] = B_y[n  :-n  ,-n-1:-n+1]

            self.dm.BC_Bx_y[1,...] = B_x[-n-1:-n+1,  n:-n  ]
            self.dm.BC_By_y[1,...] = B_y[-n-1     ,n-1:-n+1]
        else:
            self.dm.BC_Bx_x[0,...] = B_x[ :  ,n      ]
            self.dm.BC_By_x[0,...] = B_y[n:-n,n-1:n+1]

            self.dm.BC_Bx_y[0,...] = B_x[n-1:n+1,n:-n]
            self.dm.BC_By_y[0,...] = B_y[n      , :  ]
            
            self.dm.BC_Bx_x[1,...] = B_x[ :  ,-n-1   ]
            self.dm.BC_By_x[1,...] = B_y[n:-n,-n-1:  ]

            self.dm.BC_Bx_y[1,...] = B_x[-n-1:  ,n:-n]
            self.dm.BC_By_y[1,...] = B_y[-n-1   , :  ]                        
   
            
    @property
    def t(self):
        return self.time

    def perform_update(self) -> None:
        self.n_step += 1
        na = self.dm.xp.newaxis
        updated = False
        if self.well_balance:
            #U -> U'
            #if s.apply_sponge:
            #    s.sponge(s)
            self.dm.U_sp -= self.dm.U_eq_sp
        #This way godunov overrules SD_update.
        if not(self.so_godunov):
            ader_predictor(self)
            if self.SD_update:
                ader_update(self)
                updated = True
    
        if not(updated):
            fv.switch_to_finite_volume(self)
            if self.substep_troubles:
                self.dm.step_trouble[...] = 0

            # modify this to run at every rk stage
            for ader_iter in range(self.n+1):
                fv.store_high_order_fluxes(self,ader_iter)
                fv.finite_volume_update(self,ader_iter)
                if self.detect_troubles or self.so_godunov:
                    td.trouble_detection(self,ader_iter)
                    if self.plot:
                        self.display_field()
                    if self.dm.xp.any(self.dm.trouble):
                        muscl.compute_second_order_fluxes(self,ader_iter)   
                        #Replace high-order fluxes for the faces belonging a troubled control volume
                        self.dm.F_face_x[...] = np.where(self.dm.affected_face_x[na,...]==1,
                                                                 self.dm.MR_face_x,self.dm.F_face_x)
                        self.dm.F_face_y[...] = np.where(self.dm.affected_face_y[na,...]==1,
                                                                 self.dm.MR_face_y,self.dm.F_face_y)
                        if self.mhd:
                            self.dm.Ez[...] = np.where(self.dm.affected_corner[...]==1,self.dm.Es,self.dm.Ez)
                        #Perform partial update with new fluxes
                        fv.finite_volume_update(self,ader_iter)
                    if self.plot:
                        self.display_field()
                self.dm.U_cv[...] = self.dm.U_new
                if self.well_balance:
                    self.dm.W_cv[...] = self.compute_primitives(self.dm.U_cv+self.dm.U_eq_cv)-self.dm.W_eq_cv
                else:
                    self.dm.W_cv[...] = self.compute_primitives(self.dm.U_cv)
                if self.mhd:
                    self.dm.By_face_y = self.dm.By_new
                    self.dm.Bx_face_x = self.dm.Bx_new

            fv.switch_to_high_order(self)

            if self.substep_troubles:
                self.dm.trouble[...] = self.dm.step_trouble
                self.dm.step_trouble[...] = 0
            self.Ntroubles.append(self.dm.trouble.sum())
       
        self.time   += self.dt

    def perform_iterations(self, n_step: int) -> None:
        self.dm.switch_to(CupyLocation.device)
        if self.timer:
            self.execution_time = -timer() 
        self.Ntroubles = []
        for i in range(n_step):
            if not self.n_step % 1:
                print(f"Time step #{self.n_step} (t = {self.time}, min dens = {self.dm.W_cv[0].min()}, min p = {self.dm.W_cv[self._p_].min()})")
            cfl.compute_dt(self)
            self.perform_update()
        self.dm.switch_to(CupyLocation.host)
        if self.timer:
            self.execution_time += timer() 
        self.W_cv[...] = self.compute_W_cv
        print(f"{self.n_step} time steps (t = {self.time}, Execution time: {round(self.execution_time,6)} s)")
        
    def perform_time_evolution(self, t_end: float, nsteps=0) -> None:
        _p_=self._p_
        self.dm.switch_to(CupyLocation.device)
        if self.timer:
            self.execution_time = -timer() 
        self.Ntroubles = []
        while(self.time < t_end):
            if nsteps!=0:
                if not self.n_step % nsteps:
                    n_output=self.n_step//nsteps
                    print(f"Output {n_output} at time step #{self.n_step} (t = {self.time})")
                    self.dm.switch_to(CupyLocation.host)
                    if self.n_step == 0:
                        U = self.dm.xp.transpose(self.U_init_cv,(0,1,3,2,4)).reshape(
                            self.nvar,self.Ny*(self.n+1),self.Nx*(self.n+1))
                    else:
                        U = self.dm.U_new
                    np.save(self.outputs+'/rho_'+str(n_output).zfill(5),U[0])
                    np.save(self.outputs+'/troubles_'+str(n_output).zfill(5),self.dm.trouble)
                    self.dm.switch_to(CupyLocation.device)
            else:
                if not self.n_step % 1000:
                    print(f"Time step #{self.n_step} (t = {self.time}, min dens = {self.dm.W_sp[0].min()}, min p = {self.dm.W_sp[_p_].min()})")
            if not(self.advection):
                cfl.compute_dt(self)   
            if(self.time + self.dt >= t_end):
                self.dt = t_end-self.time
            if(self.dt < 1E-14):
                print(f"dt={self.dt}")
                break
            self.perform_update()
        if self.timer:
            self.execution_time += timer() 
            print(f"{self.n_step} time steps (t = {self.time}, Execution time: {round(self.execution_time,6)} s)")
        self.dm.switch_to(CupyLocation.host)
        self.W_cv[...] = self.compute_W_cv

    def get_cv_from_Bx_face_x(self,Bx):
        #Need to find a better way to do this, it's super expensive
        self.dm.Bx_fp_x[:,:,:,:-1]  = self.dm.xp.transpose(Bx[:,:-1].reshape(self.Ny,self.n+1,self.Nx,self.n+1),(0,2,1,3))
        self.dm.Bx_fp_x[:,:-1,:,-1] = self.dm.Bx_fp_x[:,1: ,:,0]
        self.dm.Bx_fp_x[:,-1,:,-1]  = Bx[:, -1].reshape(self.Ny,self.n+1)
        self.dm.Bx_fp_x[...] = self.dm.xp.einsum("mk,ijkn->ijmn",self.dm.val_sp_from_val_cv,self.dm.Bx_fp_x)
        return np.transpose(self.Bx_cv,(0,2,1,3)).reshape(self.Ny*(self.n+1),self.Nx*(self.n+1))

    def get_cv_from_By_face_y(self,By):
        #Need to find a better way to do this, it's super expensive
        self.dm.By_fp_y[:,:,:-1,:]  = self.dm.xp.transpose(By[:-1,:].reshape(self.Ny,self.n+1,self.Nx,self.n+1),(0,2,1,3))
        self.dm.By_fp_y[:-1,:,-1,:] = self.dm.By_fp_y[1:,:,0,:]
        self.dm.By_fp_y[-1,:,-1,:]  = By[-1 ,:].reshape(self.Nx,self.n+1)
        self.dm.By_fp_y[...] = self.dm.xp.einsum("mk,ijnk->ijnm",self.dm.val_sp_from_val_cv,self.dm.By_fp_y)
        return np.transpose(self.By_cv,(0,2,1,3)).reshape(self.Ny*(self.n+1),self.Nx*(self.n+1))

    @property
    def compute_U_fp_x(self):
        if self.sd_slopes == "primitives":
            W_fp_x = self.compute_fp_from_sp(self,self.dm.W_sp,0)
            U_fp_x = self.compute_conservatives(W_fp_x)
        else:
            U_fp_x = self.compute_fp_from_sp(self,self.dm.U_sp,0)
        return U_fp_x     

    @property
    def compute_U_fp_y(self):
        if self.sd_slopes == "primitives":
            W_fp_y = self.compute_fp_from_sp(self.dm.W_sp,1)
            U_fp_y = self.compute_conservatives(W_fp_y)
        else:
            U_fp_y = self.compute_fp_from_sp(self.dm.U_sp,0)
        return U_fp_y  

    @property
    def compute_U_cv(self):
        if self.sd_slopes == "primitives":
            W_cv = self.compute_cv_from_sp(self.dm.W_sp)
            U_cv = self.compute_conservatives(W_cv)
        else:
            U_cv = self.compute_cv_from_sp(self.dm.U_sp)
        return U_cv

    @property
    def compute_W_fp_x(self):
        if self.sd_slopes == "primitives":
            W_fp_x = self.compute_fp_from_sp(self.dm.W_sp,0)
        else:
            U_fp_x = self.compute_fp_from_sp(self.dm.U_sp,0)
            W_fp_x = self.compute_primitives(U_fp_x)
        return W_fp_x    

    @property
    def compute_W_fp_y(self):
        if self.sd_slopes== "primitives":
            W_fp_y = self.compute_fp_from_sp(self.dm.W_sp,1)
        else:
            U_fp_y = self.compute_fp_from_sp(self.dm.U_sp,1)
            W_fp_y = self.compute_primitives(U_fp_y)
        return W_fp_y  

    @property
    def compute_W_cv(self):
        if self.sd_slopes == "primitives":
            self.dm.W_sp = self.compute_primitives(self.dm.U_sp)
            W_cv = self.compute_cv_from_sp(self.dm.W_sp)
        else:
            U_cv = self.compute_cv_from_sp(self.dm.U_sp)
            W_cv = self.compute_primitives(U_cv)
        return W_cv

    @property
    def Bx_init_sp(self):
        return np.einsum("sf,cdxf->cdxs", self.dm.val_sp_from_val_fp, self.Bx_init_fp_x)
    @property
    def Bx_init_sp(self):
        return np.einsum("sf,cdfx->cdsx", self.dm.val_sp_from_val_fp, self.By_init_fp_y)
    @property
    def Bx_sp(self):
        return np.einsum("sf,cdxf->cdxs", self.dm.val_sp_from_val_fp, self.dm.Bx_fp_x)
    @property
    def By_sp(self):
        return np.einsum("sf,cdfx->cdsx", self.dm.val_sp_from_val_fp, self.dm.By_fp_y)
    @property
    def Bz_sp(self):
        return self.dm.U_sp[self._bz_]
    @property
    def Bx_cv(self):
        return np.einsum("mk,nl,ijkl->ijmn", self.val_cv_from_val_sp, self.val_cv_from_val_sp, self.Bx_sp)
    @property
    def By_cv(self):
        return np.einsum("mk,nl,ijkl->ijmn", self.val_cv_from_val_sp, self.val_cv_from_val_sp, self.By_sp)
    @property
    def Bz_sp(self):
        return self.dm.U_cv[self._bz_]
    @property
    def B2_sp(self):
        return self.Bx_sp**2 + self.By_sp**2  + self.Bz_sp**2
    
    @property
    def B2_cv(self):
        if self.scheme == "sd":
            return np.einsum("mk,nl,ijkl->ijmn", self.val_cv_from_val_sp, self.val_cv_from_val_sp, self.B2_sp)
        else:
            return self.dm.W_cv[self._bx_]**2 + self.dm.W_cv[self._by_]**2 + self.dm.W_cv[self._bz_]**2

    def display_field(self):
        self.dm.switch_to(CupyLocation.host)
        display_field(self,0,cmap=plt.cm.GnBu,show_trouble="both",drawcells=True,candidate=True)
        self.dm.switch_to(CupyLocation.device)