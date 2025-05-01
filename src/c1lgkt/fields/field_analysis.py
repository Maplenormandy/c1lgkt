# %% -*- coding: utf-8 -*-
"""
@author: maple

Contains codes for performing analysis on the fields
"""

import numpy as np
import scipy.interpolate

from .geometry_handlers import XgcGeomHandler
from .utility import periodify

# %%

def upsample_fft_dpot(dpot: np.ndarray, geom: XgcGeomHandler, klim: tuple | None = None) -> np.ndarray:
    """
    A function which upsamples then computes the FFT of the non-zonal potential.

    The basic idea is this: the finite toroidal resolution of the XGC data
    leads to aliasing issues if we try to take the toroidal FFT of dpot directly.
    Instead, we perform the following:

    for each flux surface:
        1) upsample to a fine grid in phi
        2) take the FFT in phi
        3) restrict back to the original mode numbers in phi

    This should give us a more accurate representation of the toroidal mode
    structure of the potential.
    """
    
    ## Parameters
    # Range of flux surfaces to consider
    if klim is None:
        ksurf0, ksurf1 = 1, geom.nsurf
    else:
        ksurf0, ksurf1 = klim
    # number of phi values; 0 is the original, 1 is the upsampled, 2 is the truncated
    nphi0 = dpot.shape[0]
    nphi1 = 96
    nphi2 = 64

    ## Grids and functions
    # Set up grids
    phi_orig = np.linspace(0, 2*np.pi/3, nphi0, endpoint=False)
    phi_samp = np.linspace(0, 2*np.pi/3, nphi1, endpoint=False)

    # Set up the upsampled FFT
    dpot_fft = np.zeros((nphi2//2+1, dpot.shape[1]), dtype=complex)


    # Quadratic smoothing interpolation
    phi_coefs = np.array([[ 0.25, -0.5,  0.25,  0. ],
                          [ 0.5 ,  0. , -0.25,  0. ],
                          [ 0.25,  0.5, -0.25,  0. ],
                          [ 0.  ,  0. ,  0.25,  0. ]])

    for ksurf in range(ksurf0, ksurf1):
        ## Set up the interpolation functions in theta
        theta_surf = geom.theta_node[geom.breaks_surf[ksurf]:geom.breaks_surf[ksurf+1]]
        dpot_surf = dpot[:,geom.breaks_surf[ksurf]:geom.breaks_surf[ksurf+1]]
        dpot_funcs = [scipy.interpolate.CubicSpline(periodify(theta_surf), periodify(dpot_surf[k,:], period=0), bc_type='periodic') for k in range(dpot_surf.shape[0])]

        # Get the q value
        q_surf = geom.interp_q(geom.psi_surf[ksurf])

        # Set up the upsampled grids
        dpot_upsamp = np.empty((nphi1, dpot_surf.shape[1]))

        # Array to hold field line interpolations
        dpot_fieldline = np.empty((nphi0, dpot_surf.shape[1]))

        ## Upsample the potential
        for kphi in range(nphi1):
            # Index of the reference phi value
            kphi_ref = kphi * nphi0 // nphi1

            # Compute the potential along the field line
            for kphi0 in range(nphi0):
                # Shift kphi0 to the correct range
                dkphi0 = (kphi0 + kphi_ref) % nphi0
                # Restrict dphi0 to [-2pi/6, 2pi/6]
                dphi0 = np.mod(phi_samp[kphi] - phi_orig[dkphi0] + 2*np.pi/6, 2*np.pi/3) - 2*np.pi/6
                # Use the theta value on the field line
                theta_k = theta_surf - dphi0 / q_surf
                # Compute the potentials
                dpot_fieldline[(kphi0+1)%nphi0,:] = dpot_funcs[dkphi0](theta_k)

            ## Compute the interpolation/filtering

            # Compute the basis functions for interpolation
            dvarphi = 2*np.pi/48 
            phifrac = (phi_samp[kphi] - phi_orig[kphi_ref]) / (2*np.pi/48)
            hbasis = ((phi_coefs[:,3] * phifrac + phi_coefs[:,2]) * phifrac + phi_coefs[:,1]) * phifrac + phi_coefs[:,0]

            # Compute filter to remove high-frequency noise in E_||.
            # Here we use a Lanczos filter with 15 points.
            hfilter = scipy.signal.windows.lanczos(16-1)[1:-1]
            hfilter = hfilter / np.sum(hfilter)

            # Compute the interpolated potential.
            dpot_interp = [np.convolve(hbasis, dpot_fieldline[:,k], mode='valid') for k in range(dpot_surf.shape[1])]
            dpot_filter = [np.convolve(hfilter, dpot_interp[k], mode='valid')[0] for k in range(dpot_surf.shape[1])]
            dpot_upsamp[kphi,:] = dpot_filter

        ## Compute then restrict the fft
        dpot_upsamp_fft = np.fft.rfft(dpot_upsamp, axis=0, norm='forward')
        dpot_fft[:,geom.breaks_surf[ksurf]:geom.breaks_surf[ksurf+1]] = dpot_upsamp_fft[:nphi2//2+1,:]

    return dpot_fft