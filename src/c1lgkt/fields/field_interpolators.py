# %% -*- coding: utf-8 -*-
"""
@author: maple


"""

import numpy as np
import scipy.interpolate
import scipy.integrate

from .equilibrium import Equilibrium
from .field_handlers import FieldHandler, RotatingFrameInfo, MeshInterpBundle, BallooningInterpBundle, BallooningModeInterpolator
from .geometry_handlers import XgcGeomHandler


# %% Functions for computing ballooning interpolations

def sum_balloon_mode(
        q, theta,
        l_max: int, ntor: int,
        mode: BallooningModeInterpolator,
        gradient: bool = True
        ):
    """
    Takes the ballooning mode interpolator and evaluates the Poisson summation
    with the eikonal to return phi_n(q, theta) in the original domain

    l_max is the maximum index to include in the poisson summation

    If in gradient mode, returns the grad(q), grad(zeta), grad(theta) components of grad(phi_n).
    Otherwise, returns the value of phi_n.
    """
    if gradient:
        dphi = np.zeros((3,)+q.shape, dtype=complex)

        for l in range(-l_max, l_max+1):
            eta = theta + 2*np.pi*l
            f, f_q, f_eta = mode(q, eta, gradient=True)
            eik = np.exp(-1j*ntor*q*eta)

            dphi[0,:] += (f_q - 1j*ntor*eta*f) * eik
            dphi[1,:] += 1j*ntor*f * eik
            dphi[2,:] += (f_eta - 1j*ntor*q*f) * eik

        return dphi
    else:
        phi = np.zeros(q.shape, dtype=complex)

        for l in range(-l_max, l_max+1):
            eta = theta + 2*np.pi*l
            f = mode(q, eta, gradient=False)
            phi += f * np.exp(-1j*ntor*q*eta)
    
        return phi

def compute_balloon_interpolation(
        tfrac, r, z, varphi,
        psi_ev,
        eq: Equilibrium,
        geom: XgcGeomHandler,
        interp_balloon: list[BallooningInterpBundle],
        gradient: bool = True
        ):
    ## Unpack some parameters
    nump = len(r)
    (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

    # Compute q, and geometric theta
    q = geom.interp_q(psi)
    gtheta = np.arctan2(z - eq.zaxis, r - eq.raxis)
    

    if gradient:
        # Compute straight field line theta from the gradient
        gdtheta, dgdtheta = geom.interp_gdtheta_grid(psi, gtheta, nu=(0,1))
        theta = gtheta + gdtheta

        # dq/dpsi
        dq = geom.interp_q(psi, nu=1)
        
        # Storage for grad(phi)
        dphi_b = np.zeros((3, nump))

        # Sum up the ballooning modes
        for bundle in interp_balloon[0]:
            ntor, mode = bundle
            tor_eik = np.exp(1j*ntor*varphi)
            dphi_b += 2*np.real(sum_balloon_mode(q, theta, 1, ntor, mode) * (1.0-tfrac) * tor_eik)
        
        # If we need to do temporal interpolation, do that too
        if tfrac > 0.0:
            for bundle in interp_balloon[1]:
                ntor, mode = bundle
                tor_eik = np.exp(1j*ntor*varphi)
                dphi_b += 2*np.real(sum_balloon_mode(q, theta, 1, ntor, mode) * tfrac * tor_eik)

        ## Convert dphi_b into the proper coordinate system

        # Right now, dphi_b has the components phi_q grad(q), phi_zeta grad(zeta), and phi_eta grad(eta)
        # We need to convert this into phi_R \vu{R}, phi_varphi \vu{varphi}, and phi_Z \vu{Z}
        dphi = np.empty((3, nump))

        # grad(zeta) = \vu{varphi} / R
        dphi[1,:] = dphi_b[1,:] / r

        # grad(q) = dq/dpsi (psi_R \vu{R} + \psi_R \vu{Z})
        # grad(theta_g) = (R-R0) / rg^2 \vu{Z} - (Z-Z0) / rg^2 \vu{R}
        # grad(eta) = (1 + \delta_theta) grad(theta_g) + \delta_psi grad(psi)

        # rg2 = geometric minor radius squared
        rg2 = (r - eq.raxis)**2 + (z - eq.zaxis)**2

        # Compute grad(eta)
        thetagr = (eq.zaxis - z) / rg2
        thetagz = (r - eq.raxis) / rg2
        etar = (1 + dgdtheta[1,:]) * thetagr + dgdtheta[0,:] * psidr
        etaz = (1 + dgdtheta[1,:]) * thetagz + dgdtheta[0,:] * psidz

        dphi[0,:] = dphi_b[0,:] * dq * psidr + dphi_b[2,:] * etar
        dphi[2,:] = dphi_b[0,:] * dq * psidz + dphi_b[2,:] * etaz

        return dphi
    else:
        # Compute straight field line theta from the gradient
        gdtheta = geom.interp_gdtheta_grid(psi, gtheta, nu=0)
        theta = gtheta + gdtheta

        # Storage for phi
        phi = np.zeros(nump)

        # Sum up the ballooning modes
        for bundle in interp_balloon[0]:
            ntor, mode = bundle
            tor_eik = np.exp(1j*ntor*varphi)
            phi += 2*np.real(sum_balloon_mode(q, theta, 1, ntor, mode, gradient=False) * (1.0-tfrac) * tor_eik)

        # If we need to do temporal interpolation, do that too
        if tfrac > 0.0:
            for bundle in interp_balloon[1]:
                ntor, mode = bundle
                tor_eik = np.exp(1j*ntor*varphi)
                phi += 2*np.real(sum_balloon_mode(q, theta, 1, ntor, mode, gradient=False) * tfrac * tor_eik)
        
        return phi

# %% Functions for computing poloidal punctures, used for computing interpolation on the mesh

def f_fieldline_phi(tr, yr, eq: Equilibrium):
    """
    Push a field line follower in the (R,Z) plane, along with the variational equations,
    parameterized by phi
    """
    r = yr[0,:]
    z = yr[1,:]

    r00 = yr[2,:]
    r01 = yr[3,:]
    r10 = yr[4,:]
    r11 = yr[5,:]

    # Evaluate psi and its derivatives
    psi = eq.interp_psi.ev(r, z)
    psidr = eq.interp_psi.ev(r, z, dx=1)
    psidz = eq.interp_psi.ev(r, z, dy=1)
    psidrr = eq.interp_psi.ev(r,z,dx=2)
    psidrz = eq.interp_psi.ev(r,z,dx=1,dy=1)
    psidzz = eq.interp_psi.ev(r,z,dy=2)

    # Detect if a particle is outside the LCFS; if so, use different interpolation
    outside_lcfs = np.logical_or(psi > eq.psix, z < eq.zx)
    ff = np.choose(outside_lcfs, (eq.interp_ff(psi), eq.ff[-1]))
    dff = np.choose(outside_lcfs, (eq.interp_ff(psi, nu=1), 0))

    r_over_ff = r/ff
    dr_over_ff = r_over_ff * dff / ff

    # Field line
    dyr = np.empty(yr.shape)
    dyr[0,:] =  r_over_ff*psidz
    dyr[1,:] = -r_over_ff*psidr

    # Variational equation matrix
    a00 =  psidz/ff + r_over_ff * psidrz - dr_over_ff * psidz*psidr
    a01 =           + r_over_ff * psidzz - dr_over_ff * psidz**2   
    a10 = -psidr/ff - r_over_ff * psidrr + dr_over_ff * psidr**2   
    a11 =           - r_over_ff * psidrz + dr_over_ff * psidz*psidr

    # Old code
    """
    # Field line
    dyr = np.empty(yr.shape)
    dyr[0,:] =  r*psidz/ff
    dyr[1,:] = -r*psidr/ff

    # Variational equation matrix
    a00 =  psidz/ff + r*( psidrz / ff - psidz*psidr * dff / ff**2)
    a01 =             r*( psidzz / ff - psidz**2    * dff / ff**2)
    a10 = -psidr/ff + r*(-psidrr / ff + psidr**2    * dff / ff**2)
    a11 =             r*(-psidrz / ff + psidz*psidr * dff / ff**2)
    """

    # Change in variational matrix. Elements in order are 00, 01, 10, 11
    dyr[2,:] = a00 * r00 + a01 * r10
    dyr[3,:] = a00 * r01 + a01 * r11
    dyr[4,:] = a10 * r00 + a11 * r10
    dyr[5,:] = a10 * r01 + a11 * r11

    return dyr

def rk4_step_broadcast(f, tr_span, yr0, dyr0, args):
    """
    This function goes from t_span[0,:] to t_span[1,:] in a single RK4 step. Returns (y1, dy1).

    Note that there is a slightly different requirement on the signature of f compared to
    scipy.solve_ivp, since since tr is a vector of times associated to each yr
    """
    tr = tr_span[0]
    nstep = 4
    dtr = (tr_span[1] - tr_span[0])/nstep

    for kt in range(nstep):
        if kt == 0:
            k1 = dyr0
        else:
            k1 = f(tr, yr0, *args)
        k2 = f(tr+dtr/2, yr0+k1*(dtr/2), *args)
        k3 = f(tr+dtr/2, yr0+k2*(dtr/2), *args)
        k4 = f(tr+dtr, yr0+k3*dtr, *args)

        y1 = yr0 + (dtr/6)[np.newaxis,:]*(k1 + 2*k2 + 2*k3 + k4)

        yr0 = y1
        tr = tr + dtr

    return y1, f(tr+dtr, y1, *args)

def compute_poloidal_punctures(r, z, varphi, eq: Equilibrium):
    """
    Compute the poloidal puncture points for particles following field lines.
    """
    nump = len(r)

    # The indices of the reference poloidal plane
    dvarphi = 2*np.pi/48
    kphir = np.floor_divide(varphi,dvarphi).astype(int)

    # Set up initial conditions
    yr0 = np.empty((6, nump))
    yr0[0,:] = r
    yr0[1,:] = z
    yr0[2,:] = 1
    yr0[3,:] = 0
    yr0[4,:] = 0
    yr0[5,:] = 1

    dyr0 = f_fieldline_phi(0, yr0, eq)

    # Compute the puncture points
    yp1, dyp1 = rk4_step_broadcast(f_fieldline_phi, [varphi, kphir*dvarphi], yr0, dyr0, args=(eq,))
    yp0, dyp0 = rk4_step_broadcast(f_fieldline_phi, [kphir*dvarphi, (kphir-1)*dvarphi], yp1, dyp1, args=(eq,))
    yp2, dyp2 = rk4_step_broadcast(f_fieldline_phi, [varphi, (kphir+1)*dvarphi], yr0, dyr0, args=(eq,))
    yp3, dyp3 = rk4_step_broadcast(f_fieldline_phi, [(kphir+1)*dvarphi, (kphir+2)*dvarphi], yp2, dyp2, args=(eq,))

    # Line the puncture points up in a grid
    y_hits = [yp0, yp1, yp2, yp3]
    dy_hits = [dyp0, dyp1, dyp2, dyp3]

    return y_hits, dy_hits


# %%

## Coefficient arrays used to compute the cubic hermite interpolation of the electric fields.
# Note that the kth basis function is h_k(t) = phi_coefs[k,3] * t**3 + phi_coefs[k,2] * t**2 + ...
'''
# Cubic interpolation
phi_coefs = np.array([[ 0. , -0.5,  1. , -0.5],
                      [ 1. ,  0. , -2.5,  1.5],
                      [ 0. ,  0.5,  2. , -1.5],
                      [ 0. ,  0. , -0.5,  0.5]])

# Linear interpolation
phi_coefs = np.array([[ 0. ,  0. ,  0. ,  0. ],
                      [ 1. , -1. ,  0. ,  0. ],
                      [ 0. ,  1. ,  0. ,  0. ],
                      [ 0. ,  0. ,  0. ,  0. ]])
'''
# Quadratic smoothing interpolation
phi_coefs = np.array([[ 0.25, -0.5,  0.25,  0. ],
                      [ 0.5 ,  0. , -0.25,  0. ],
                      [ 0.25,  0.5, -0.25,  0. ],
                      [ 0.  ,  0. ,  0.25,  0. ]])

# Matrix of derivatives
dphi_coefs = np.zeros((4, 3))
for k in range(3):
    dphi_coefs[:,k] = (k+1)*phi_coefs[:,k+1]


# %% Functions for computing fields

def compute_mesh_interpolation(tfrac, r, z, varphi, psi_ev, eq: Equilibrium, interp_npot: list[MeshInterpBundle], gradient: bool = True):
    ## Unpack some parameters
    nump = len(r)
    (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

    tor_sectors, interp_npot0 = interp_npot[0]
    
    if tfrac > 0.0:
        tor_sectors, interp_npot1 = interp_npot[1]

    # Spacing between poloidal planes
    dvarphi = 2*np.pi/(tor_sectors*len(interp_npot0))
    kphir = np.floor_divide(varphi,dvarphi).astype(int)
    
    ## Prepare sorted particle positions
    kphir_argsort = np.argsort(kphir%16)
    kphir_sorted = kphir[kphir_argsort]
    
    # Prepare indices to reverse the argsort at the end
    kphir_invargsort = np.argsort(kphir_argsort)
    
    # Toroidal position of particle between nearest poloidal planes normalized to [0,1) for interpolation
    varphi_sorted = varphi[kphir_argsort]
    phifrac_sorted = varphi_sorted/dvarphi - kphir_sorted

    r_sorted = r[kphir_argsort]
    z_sorted = z[kphir_argsort]
    
    ## Compute poloida punctures
    y_hits_sorted, dy_hits_sorted = compute_poloidal_punctures(r_sorted, z_sorted, varphi_sorted, eq)

    # Prepare arrays to hold the interpolations
    phi_hits_sorted = np.empty((4, nump))
    if gradient == True:
        dphi_hits_sorted = np.empty((3, 4, nump))

    # Where the breakpoints are for the poloidal planes. Namely, for an array of
    # kphir_sorted = [0, 16 | 1, 17, 1 | 6 | 15] we want to find the indices of the
    # breakpoints, denoted with |.
    kphi_breaks = np.searchsorted(kphir_sorted%16, np.arange(16, dtype=int))

    ## Iterate over the poloidal planes, then iterate over the poloidal hit points
    # The basic strategy is that we want to perform all evaluations for a single poloidal plane
    # at the same time. Thus, for e.g. kphi=0, we want to evaluate particles with
    # kphir=1,0,-1,-2 (mod 16), as shown by this figure:
    #
    # kphi =  -2  -1   0   1   2
    #          | * | *[|]* | * |
    # kphir=    -2  -1   0   1
    #
    for kphi in range(16):
        for k in range(4):
            # Breakpoint indices
            kphi_k = (kphi+1-k)%16
            kp0 = kphi_breaks[kphi_k]
            
            if kphi_k == 15:
                kp1 = nump
            else:
                kp1 = kphi_breaks[kphi_k+1]

            if kp0 == kp1:
                # If there are no particles in this breakpoint, skip this step
                continue
            else:
                # Otherwise, perform the interpolation!
                yk = y_hits_sorted[k][:,kp0:kp1]
                dyk = dy_hits_sorted[k][:,kp0:kp1]

                # Compute phi in the plane
                if tfrac > 0.0:
                    phi_hits_sorted[k,kp0:kp1] = (1-tfrac) * interp_npot0[kphi](yk[0,:], yk[1,:]) + tfrac * interp_npot1[kphi](yk[0,:], yk[1,:])
                else:
                    phi_hits_sorted[k,kp0:kp1] = interp_npot0[kphi](yk[0,:], yk[1,:])

                # Compute dphi in the plane
                if gradient == True:
                    if tfrac > 0.0:
                        dnpot0 = interp_npot0[kphi].gradient(yk[0,:], yk[1,:])
                        dnpot1 = interp_npot1[kphi].gradient(yk[0,:], yk[1,:])
                        dnpot = ((1-tfrac) * dnpot0[0] + tfrac * dnpot1[0], (1-tfrac) * dnpot0[1] + tfrac * dnpot1[1])
                    else:
                        dnpot = interp_npot0[kphi].gradient(yk[0,:], yk[1,:])
                    dphi_hits_sorted[0,k,kp0:kp1] = yk[2,:] * dnpot[0] + yk[3,:] * dnpot[1]
                    dphi_hits_sorted[2,k,kp0:kp1] = yk[4,:] * dnpot[0] + yk[5,:] * dnpot[1]
                    dphi_hits_sorted[1,k,kp0:kp1] = -(dyk[0,:] * dnpot[0] + dyk[1,:] * dnpot[1])
    
    ## Compute the basis functions
    if gradient == True:
        hbasis_sorted = np.empty((4, nump))
        dhbasis_sorted = np.empty((4, nump))
        for k in range(4):
            dhbasis_sorted[k,:] = (dphi_coefs[k,2] * phifrac_sorted + dphi_coefs[k,1]) * phifrac_sorted + dphi_coefs[k,0]
            hbasis_sorted[k,:] = ((phi_coefs[k,3] * phifrac_sorted + phi_coefs[k,2])
                            * phifrac_sorted + phi_coefs[k,1]) * phifrac_sorted + phi_coefs[k,0]

        ## Sum the basis functions with the computed phi
        dphi_sorted = np.empty((3,nump))
        dphi_sorted[0,:] = np.sum(hbasis_sorted * dphi_hits_sorted[0,:,:], axis=0)
        dphi_sorted[1,:] = (np.sum(hbasis_sorted * dphi_hits_sorted[1,:,:], axis=0) + np.sum(dhbasis_sorted * phi_hits_sorted, axis=0) / dvarphi)/ r_sorted
        dphi_sorted[2,:] = np.sum(hbasis_sorted * dphi_hits_sorted[2,:,:], axis=0)

        return dphi_sorted[:,kphir_invargsort]
    else:
        hbasis_sorted = np.empty((4, nump))
        for k in range(4):
            hbasis_sorted[k,:] = ((phi_coefs[k,3] * phifrac_sorted + phi_coefs[k,2])
                            * phifrac_sorted + phi_coefs[k,1]) * phifrac_sorted + phi_coefs[k,0]
        phi_sorted = np.sum(hbasis_sorted * phi_hits_sorted, axis=0)

        return phi_sorted[kphir_invargsort]

def compute_fields(
        t, r, z, varphi,
        psi_ev,
        eq: Equilibrium,
        fields: FieldHandler,
        frame: RotatingFrameInfo | None,
        gradient: bool = True
        ):
    """
    Responsible for taking a FieldHandler and doing all the unpacking necessary to compute the fields
    at the given physical time and position.

    NOTE: Generally only one of mesh or balloon should be specified. If both are specified, mesh will be used.
    """
    # Unpack some parameters
    nump = len(r)
    (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

    ## First, check if a fixed rotating frame has been specified
    if frame is None:
        # If no frame has been specified, get the time and tfrac as usual
        tind, tfrac = fields.request_tind(t)
    else:
        # Else, pick out the frozen time frame and go into the moving frame
        tind = frame.tind_frozen
        tfrac = 0.0
        varphi = varphi - frame.omega_rotation * (t - frame.t0)

    interp0 = fields.request_interp(tind)

    # If tfrac is greater than zero, we'll also need the other interpolation functions
    if tfrac > 0.0:
        interp1 = fields.request_interp(tind+1)
        interp_mesh = [interp0.mesh, interp1.mesh]
        interp_balloon = [interp0.balloon, interp1.balloon]
    else:
        interp_mesh = [interp0.mesh]
        interp_balloon = [interp0.balloon]

    if gradient == True:
        ## If the non-zonal component of the potential is specified, we need to do some interpolation
        if interp0.mesh is not None:
            dphi = compute_mesh_interpolation(tfrac, r, z, varphi, psi_ev, eq, interp_mesh)
        elif interp0.balloon is not None:
            geom = fields.request_geom()
            dphi = compute_balloon_interpolation(tfrac, r, z, varphi, psi_ev, eq, geom, interp_balloon)
        else:
            # If the non-zonal fields are not specified, initialize the fields with zeros
            dphi = np.zeros((3,nump))

        ## Compute the zonal potential
        if interp0.zonal is not None:
            if tfrac > 0.0:
                dzpot = (1-tfrac) * interp0.zonal(psi, nu=1) + tfrac * interp1.zonal(psi, nu=1)
            else:
                dzpot = interp0.zonal(psi, nu=1)
            dphi[0,:] += dzpot * psidr
            dphi[2,:] += dzpot * psidz

        return dphi * fields.scale_conversion()
    else:
        ## If the non-zonal component of the potential is specified, we need to do some interpolation
        if interp0.mesh is not None:
            phi = compute_mesh_interpolation(tfrac, r, z, varphi, psi_ev, eq, interp_mesh, gradient=False)
        elif interp0.balloon is not None:
            geom = fields.request_geom()
            phi = compute_balloon_interpolation(tfrac, r, z, varphi, psi_ev, eq, geom, interp_balloon, gradient=False)
        else:
            # If the non-zonal fields are not specified, initialize the fields with zeros
            phi = np.zeros(nump)

        ## Compute the zonal potential
        if interp0.zonal is not None:
            if tfrac > 0.0:
                phi += (1-tfrac) * interp0.zonal(psi) + tfrac * interp1.zonal(psi)
            else:
                phi += interp0.zonal(psi)
        
        return phi * fields.scale_conversion()
