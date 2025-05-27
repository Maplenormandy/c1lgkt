# -*- coding: utf-8 -*-
"""
@author: maple

This file contains codes to analyze particles and particle trajectories
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

from ..fields.equilibrium import Equilibrium
from .particle_motion import ParticleParams, RotatingFrameInfo
from ..fields.field_handlers import FieldHandler
from ..fields.geometry_handlers import XgcGeomHandler
from ..fields.field_interpolators import compute_fields

# %% Various functions for computing analysis quantities

def compute_integrals_dk(t, ysol, eq: Equilibrium, pp: ParticleParams, fields: FieldHandler, frame: RotatingFrameInfo | None = None):
    """
    Computes the integrals of motion: Hamiltonian, angular momentum, and magnetic
    moment. Note mu is conserved, but might be useful later on.
    
    The integrals are computed assuming the first axis of y holds the data, and if
    there is a second axis, then we iterate over it
    """
    # Reshape into a uniform shape
    y_ = np.reshape(ysol, (ysol.shape[0], -1))
    t_ = np.reshape(t, (np.size(t),))
    nump = y_.shape[0] // 5

    ham = np.empty((nump, len(t_)))
    lphi = np.empty((nump, len(t_)))

    # Iterate over time to compute the integrals
    for k in range(y_.shape[1]):
        yr = np.reshape(y_[:,k], (5,-1))
        
        r = yr[0,:]
        varphi = yr[1,:]
        z = yr[2,:]
        vll = yr[3,:]
        mu = yr[4,:]

        # Magnetic stuff
        psi = eq.interp_psi.ev(r, z)
        bv = eq.compute_bv(r, z)
        modb = np.linalg.norm(bv, axis=0)

        psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
        pot = compute_fields(t_[k], r, z, varphi, psi_ev, eq, fields, frame, gradient=False)

        ham[:,k] = 0.5 * pp.m * vll**2 + mu * modb + pp.z * pot
        lphi[:,k] = pp.z * psi + pp.m * vll * r * bv[1,:] / modb
    
    if nump == 1:
        return np.reshape(ham, np.shape(t)), np.reshape(lphi, np.shape(t))
    else:
        return ham, lphi

def compute_frequencies(t, r, z, varphi, eq: Equilibrium):
    """
    Compute frequencies as well as various winding numbers
    z, r, varphi should be arrays of shape [nt]
    """
    
    # Compute winding number around the magnetic axis
    poloidal_angle = np.unwrap(np.arctan2(z - eq.zaxis, r - eq.raxis))
    j_axis = int(np.round((poloidal_angle[-1] - poloidal_angle[0])/2/np.pi))
    
    # Compute the orientation of the contour in the poloidal plane using the
    # shoelace formula to get the signed area
    j_orientation = int(np.sign(np.sum(r*np.roll(z,-1)) - np.sum(z*np.roll(r,-1))))
    
    omega_pol = 2*np.pi / (t[-1] - t[0]) * j_orientation
    omega_tor = (varphi[-1] - varphi[0]) / (t[-1] - t[0])
    
    # q_kinetic = omeg_tor / omeg_pol
    return omega_tor, omega_pol, j_axis

# %% Functions for computing initial conditions and such

def compute_parallel_energy(t, r, z, varphi, mu, ham, lphi, eq: Equilibrium, pp: ParticleParams, fields: FieldHandler, frame: RotatingFrameInfo):
    """
    Computes the Kll = (pll - m Omega R bt)^2/2m term. Returns a tuple containing (Kll, m Omega R bt)

    t should be a scalar, while r, z, varphi, etc... should be arrays of shape [nump], or scalars
    """
    initial_shape = np.shape(r)

    r = np.reshape(r, (-1,))
    z = np.reshape(z, (-1,))
    varphi = np.reshape(varphi, (-1,))
    mu = np.reshape(mu, (-1,))

    # Compute the adiabatic invariant
    kam = ham - frame.omega_rotation * lphi

    # Magnetic stuff
    psi = eq.interp_psi.ev(r, z)
    bv = eq.compute_bv(r, z)
    modb = np.linalg.norm(bv, axis=0)

    psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
    pot = compute_fields(t, r, z, varphi, psi_ev, eq, fields, frame, gradient=False)

    pll0 = pp.m * frame.omega_rotation * r * bv[1,:] / modb

    kll = kam + pll0**2 / (2*pp.m) - mu*modb - pp.z*(pot - frame.omega_rotation * psi)

    return np.reshape(kll, initial_shape), np.reshape(pll0, initial_shape)


# %% Functions for computing Poincare sections

def compute_punctures(t, ysol, fpunc, condpunc=None, period=-1):
    """
    Compute punctures. Returns a tuples consisting of positive punctures and
    one consisting of negative punctures.

    We look for sign crossings of fpunc when condpunc is true
    """
    nump = ysol.shape[0]//5

    ppuncs = [None] * nump
    npuncs = [None] * nump

    if condpunc is None:
        condpunc = np.ones_like(fpunc)

    for k in range(nump):
        fpunc_k = fpunc[k,:]
        # TODO: we're missing a special case where fpunc[0] = 0. Should try to fix eventually
        if period > 0:
            jumps = np.diff(np.floor_divide(fpunc_k, period)) * condpunc[k,:-1] * condpunc[k,1:]
        else:
            jumps = np.diff((fpunc_k >= 0) * 1.0) * condpunc[k,:-1] * condpunc[k,1:]

        ind_ppuncs = np.argwhere(jumps > 0)
        ind_npuncs = np.argwhere(jumps < 0)

        t_ppuncs = np.empty(ind_ppuncs.shape[0])
        t_npuncs = np.empty(ind_npuncs.shape[0])

        y_ppuncs = np.empty((5, ind_ppuncs.shape[0]))
        y_npuncs = np.empty((5, ind_npuncs.shape[0]))

        # Out of laziness, compute a linear interpolation to the puncture point.
        # TODO: Do some sort of fancier interpolation to compute the punctures?
        for j in range(ind_ppuncs.shape[0]):
            tind = ind_ppuncs[j,0]
            if period > 0:
                ffrac = (np.mod(-fpunc_k[tind] + period/2, period) - period/2) / (fpunc_k[tind+1] - fpunc_k[tind])
            else:
                ffrac = -fpunc_k[tind] / (fpunc_k[tind+1] - fpunc_k[tind])

            t_ppuncs[j] = (1-ffrac)*t[tind] + ffrac*t[tind+1]
            y_ppuncs[:,j] = (1-ffrac)*ysol[k:k+5*nump:nump,tind] + ffrac*ysol[k:k+5*nump:nump,tind+1]

        for j in range(ind_npuncs.shape[0]):
            tind = ind_npuncs[j,0]
            if period > 0:
                ffrac = (np.mod(-fpunc_k[tind] + period/2, period) - period/2) / (fpunc_k[tind+1] - fpunc_k[tind])
            else:
                ffrac = -fpunc_k[tind] / (fpunc_k[tind+1] - fpunc_k[tind])

            t_npuncs[j] = (1-ffrac)*t[tind] + ffrac*t[tind+1]
            y_npuncs[:,j] = (1-ffrac)*ysol[k:k+5*nump:nump,tind] + ffrac*ysol[k:k+5*nump:nump,tind+1]

        ppuncs[k] = (t_ppuncs, y_ppuncs)
        npuncs[k] = (t_npuncs, y_npuncs)

    return ppuncs, npuncs

def compute_toroidal_punctures(t, ysol, frame: RotatingFrameInfo, period=2*np.pi/3, offset=0.0):
    nump = ysol.shape[0]//5

    fpunc = ysol[1*nump:2*nump,:] - frame.omega_rotation * (t - frame.t0) - offset

    return compute_punctures(t, ysol, fpunc, period=period)

def compute_midplane_punctures(t, ysol, geom: XgcGeomHandler):
    """
    Compute midplane punctures. Returns a tuples consisting of positive punctures and
    one consisting of negative punctures.
    """
    nump = ysol.shape[0]//5

    zpunc = ysol[2*nump:3*nump,:] - geom.zaxis
    cpunc = ysol[:nump,:] > geom.raxis

    return compute_punctures(t, ysol, zpunc, cpunc)
