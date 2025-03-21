# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:51:33 2024

@author: maple

This file contains codes for the functions that push various particles
"""

import numpy as np
from collections import namedtuple
import scipy.integrate

from ..fields.equilibrium import Equilibrium
from ..fields.field_handlers import RotatingFrameInfo, FieldHandler, ZonalFieldHandler
from ..fields.field_interpolators import compute_fields

import matplotlib.tri as tri

from typing import NamedTuple, Any

# %% Parameters for various particles

### Unit normalizations ###
# Length normalized to 1 m, and B to 1 T
# Normalize charge to 1 electron charge
# Normalize energy to 1 keV. The derived units are:
# m_ref = (1 e * 1 T * 1 m)^2 / 1 keV = 1.60217663e-22 kg
#  => google search calculator version: (1 electron charge * 1 T * 1 m)^2 / 1 keV
# t_ref = 1e-3 s
# E_ref = 1 kV

# New particleparams definition, with type hinting
class ParticleParams(NamedTuple):
    """
    Named tuple holding particle properties for convenience and type hinting.
    vt is the velocity of a 1keV particle
    """
    z: float
    m: float
    vt: float

    

# Some default particle parameters
deut = ParticleParams(z=1, m=2.08793698e-5, vt=1/np.sqrt(2.08793698e-5))
elec = ParticleParams(z=-1, m=5.6856301e-9, vt=1/np.sqrt(5.6856301e-9))

# 3.5421666e-3 <-- ion gyroradius at 0.5 keV, 1.29 T

# %% Pushers for field lines

def f_fieldline(t, y, eq: Equilibrium, pp: Any = None, fields: Any = None):
    """
    Push a (single) field line tracer in (R, varphi, Z) coordinates.
    """
    yr = np.reshape(y, (3,-1))
    
    r = yr[0,:]
    #varphi = yr[1,:]
    z = yr[2,:]
    
    # Evaluate psi and its derivatives
    psi = eq.interp_psi.ev(r, z)
    psidr = eq.interp_psi.ev(r, z, dx=1)
    psidz = eq.interp_psi.ev(r, z, dy=1)
    
    # Detect if a particle is outside the LCFS; if so, use different interpolation
    outside_lcfs = np.logical_or(psi > eq.psix, z < eq.zx)
    ff = np.choose(outside_lcfs, (eq.interp_ff(psi), eq.ff[-1]))
    
    return np.ravel(np.array([-psidz / r, -ff / r**2, psidr / r]))


# %% Events for particle termination

def event_copassing(t, y, eq: Any = None, pp: Any = None, fields: Any = None, frame: RotatingFrameInfo | None = None):
    """
    Returns the z coordinate of a particle, used to determine when the particle
    has completed a poloidal transit
    """
    
    # Add a little bit of 'easing' to ignore the first crossing; added since the
    # number of event counting doesn't work for some reason
    return y[2] + max(0, 1e-8-t)

def event_ctrpassing(t, y, eq: Any = None, pp: Any = None, fields: Any = None, frame: RotatingFrameInfo | None = None):
    """
    Returns the z coordinate of a particle, used to determine when the particle
    has completed a poloidal transit
    """
    
    # Add a little bit of 'easing' to ignore the first crossing; added since the
    # number of event counting doesn't work for some reason
    return y[2] - max(0, 1e-8-t)

def event_botloss(t, y, eq: Equilibrium, pp: Any = None, fields: Any = None, frame: RotatingFrameInfo | None = None):
    """
    Returns the z coordinate of a particle shifted by the bottom of the divertor
    """
    
    return y[2] - eq.zmin

event_copassing.terminal = True
event_copassing.direction = 1

event_ctrpassing.terminal = True
event_ctrpassing.direction = -1

event_botloss.terminal = True

    
# %% Functions for gyrokinetic particle pushing with zonally symmetric equilibria

def f_driftkinetic(t, y, eq: Equilibrium, pp: ParticleParams, fields: FieldHandler, frame: RotatingFrameInfo | None = None):
    """
    Push a (single) drift-kinetic tracer in (R, varphi, Z) coordinates.
    Note y = (R, varphi, Z, v_||, mu).

    Note the particles are always axis=-1. The philosophy is that rather than keeping particles
    local in memory (i.e. [r0, z0, r1, z1, ...]), it's better to pack coordinates locally in
    memory (i.e. [r0, r1, ..., z0, z1, ...]) since we typically operate on batches of coordinates,
    rather than on a per-particle basis
    """
    yr = np.reshape(y, (5,-1))
    
    r = yr[0,:]
    varphi = yr[1,:]
    z = yr[2,:]
    vll = yr[3,:]
    mu = yr[4,:]
    
    ## Magnetic terms
    psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
    bv, bu, modb, gradmodb, curlbu = eq.compute_geom_terms(r, psi_ev, ff_ev)
    ## Bstar and Bstar parallel
    bstar = bv + (pp.m / pp.z) * curlbu * vll[np.newaxis,:]
    bstarll = np.sum(bu*bstar, axis=0)

    ## Electric potential gradient
    dphi = compute_fields(t, r, z, varphi, psi_ev, eq, fields, frame)
    
    ## Finally compute the (spatial) gradient of the Hamiltonian
    gradh = mu * gradmodb + pp.z * dphi

    dyr = np.empty(yr.shape)
    
    rdot = (np.cross(bu, gradh, axis=0) / pp.z + vll[np.newaxis,:] * bstar) / bstarll[np.newaxis,:]
    dyr[0,:] = rdot[0,:]
    dyr[1,:] = rdot[1,:] / r
    dyr[2,:] = rdot[2,:]
    dyr[3,:] = -np.sum(bstar*gradh, axis=0) / bstarll / pp.m
    dyr[4,:] = 0

    return np.ravel(dyr)

def f_kinetic(t, y, eq: Equilibrium, pp: ParticleParams, fields: ZonalFieldHandler):
    """
    Push a (single) kinetic tracer in (R, varphi, Z) coordinates.
    Note y = (R, varphi, Z, vR, vphi, vZ)
    """
    
    # Evaluate psi and its derivatives
    psi = eq.interp_psi.ev(y[0],y[2])
    psidr = eq.interp_psi.ev(y[0],y[2],dx=1)
    psidz = eq.interp_psi.ev(y[0],y[2],dy=1)
    
    # Detect if a particle is outside the LCFS; if so, use different interpolation
    outside_lcfs = (psi > eq.psix or y[2] < eq.zx)
    if outside_lcfs:
        ff = eq.ff[-1]
    else:
        ff = eq.interp_ff(psi)
        
    bv = np.array([-psidz / y[0], -ff / y[0], psidr / y[0]])
    
    dy = np.zeros(6)
    
    # v = r' er + z' ez + r th' eth
    dy[0] = y[3]
    dy[1] = y[4] / y[0]
    dy[2] = y[5]
    
    vxB = np.cross(y[3:], bv)
    
    # Electrostatic potential
    dphi = fields.interp_phi(psi, nu=1)*fields.scale_conversion()
    
    # a = (r'' - r th' ** 2) er + z'' ez + (r th'' + 2 r' th') eth
    # (r')' = r'' = ar + vth**2/r
    dy[3] = pp.z*(vxB[0] - dphi*psidr)/pp.m + y[4]**2/y[0]
    # (r th')' = r' th' + r th'' = ath - r' th'
    dy[4] = pp.z*vxB[1]/pp.m - y[3] * y[4] / y[0]
    # (z')' = z'' = az
    dy[5] = pp.z*(vxB[2] - dphi*psidz)/pp.m
    
    return dy
