# %% -*- coding: utf-8 -*-
"""
@author: maple

This file defines the abstract FieldHandler class, and gives some implementations
"""

import numpy as np
import scipy.interpolate
import scipy.integrate

# Below are some hacks to load eqtools. TODO: Figure out how to remove dependencies
#import sys
#sys.path.append(R'C:\Users\maple\OneDrive\Research\IFS\Scripts\libs\eqtools')
#from eqtools import eqdskreader, pfilereader # type: ignore

from .equilibrium import Equilibrium
from .geometry_handlers import XgcGeomHandler, CubicTriInterpolatorMemoized

from typing import Type, TypeVar, Protocol, NamedTuple

import netCDF4

import matplotlib.tri as tri

# %% Basic types

class BallooningModeInterpolator(Protocol):
    """
    A protocol for a function that returns the ballooning mode structure at a given flux coordinate.
    Represents something like f_n(q,eta) type functions.

    TODO: Think about how to implement gyro-averaging
    """
    def __call__(self, q: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a tuple (f, df/dq, df/deta) at the given coordinates
        """
        ...

type ZonalInterpBundle = scipy.interpolate.CubicSpline
type MeshInterpBundle = tuple[int,list[tri.TriInterpolator]]
type BallooningInterpBundle = list[tuple[int,BallooningModeInterpolator]]

class InterpTuple(NamedTuple):
    """
    Tuple containing interpolation functions for functions of different types. Each field can be a certain type, or None
    if that interpolation is not active. The types are as follows:
    - zonal: f(psi) type functions within the LCFS, which are represented scipy spline interpolators.
    - mesh: f_i(r,z) functions defined on poloidal planes on the toroidal index i. This is stored as a tuple, whose first
    element is the number of repeated toroidal sectors, and the second is a list of tri.TriInterpolator representing the f_i(r,z)
    - ballooning: f_n(q,eta) type functions for ballooning modes with toroidal mode number n. This is stored as a list of tuples,
    each containing the toroidal mode number and a BallooningModeInterpolator
    """
    zonal: ZonalInterpBundle | None = None
    mesh: MeshInterpBundle | None = None
    balloon: BallooningInterpBundle | None = None

class FieldHandler(Protocol):
    """
    A protocol that specifies the functions that a field handler should expose. The interpolation functions are (usually)
    stored on a per time index basis, so the field handler should tell us how to find the time index associated with a
    physical time t, and include functions to then request the interpolation fields at that time index.
    """
    def request_interp(self, tind: int) -> InterpTuple:
        """
        Request the interpolation function(s) at a given time index.
        """
        ...

    def request_tind(self, t: float) -> tuple[int, float]:
        """
        This function should either return a tuple of (tind, tfrac). If tfrac = 0.0,
        then only one tind needs to be polled.
        """
        ...

    def scale_conversion(self) -> float:
        """
        Returns a conversion factor for the field data.
        """
        ...

class RotatingFrameInfo(NamedTuple):
    """
    Named tuple holding some information for the reference frame the particle pusher should use
    """
    t0: float
    omega_rotation: float
    tind_frozen: int

# %% Simple zonal field handler

tfrac_one = np.array([1.0])

class ZonalFieldHandler:
    def __init__(self, eq:Equilibrium):
        """
        Initializes a simple zero zonal electric field
        """
        self.eq = eq
        self._phi_x = np.linspace(np.min(eq.psirz), np.max(eq.psirz), 4)
        self._phi_data = np.zeros(len(self._phi_x))
        self.phimult = 1.0
        self.interp_phi: scipy.interpolate.CubicSpline = scipy.interpolate.CubicSpline(self._phi_x, self._phi_data, bc_type='natural', extrapolate=True)
    
    def request_interp(self, tind: int) -> InterpTuple:
        return InterpTuple(zonal=self.interp_phi)
    
    def request_tind(self, t: float) -> tuple[int, float]:
        return (0, 0.0)

    def scale_conversion(self):
        return self.phimult
    
# %% Class for fields constructed from force balance

# TODO: Figure out how to remove dependencies on eqtools
'''
# Some static typing stuff to help with type hinting
T = TypeVar('T', bound='ForceBalanceFieldHandler')

class ForceBalanceFieldHandler(ZonalFieldHandler):
    def __init__(self, eq:Equilibrium):
        super().__init__(eq)

    @classmethod
    def from_pfile(cls: Type[T], eq: Equilibrium, filename: str) -> T:
        """
        Compute a model electrostatic potential using force balance and a model
        for SOL phi
        """
        pf = pfilereader.PFileReader(pfile=filename, verbose=False)

        fields = cls(eq)

        fields._pf = pf
        
        # In lowest order force balance, -dphi/dpsi = omega + dpi/dpsi / (ni * e)
        # Rotational contribution to Er. kRad/s becomes Rad/ms, so no numerical factors needed
        epsi_omeg = pf.omeg.y
        # Diamagnetic contribution to Er. keV is already in the appropriate unit.
        epsi_diamag = (pf.ti.dydx + pf.ti.y * pf.ne.dydx / pf.ne.y) / 1 / eq.psix
        # e_psin is dphi/dpsin, i.e. not actually the electric field
        epsi = epsi_omeg + epsi_diamag
        
        # Compute phi by integrating numerically
        phi_data = -scipy.integrate.cumulative_trapezoid(epsi, pf.ti.x * eq.psix, initial=0)
        # Shift phi at the LCFS so it's equal to -Te/2 * ln(2pi me / mi * (1 + Ti/Te)).
        phi_data = phi_data - phi_data[-1] + 2.49220173811 * pf.te.y[-1]
        
        # Compute the poloidal magnetic field
        rmax_lcfs = np.max(eq.lcfsrz[:,0])
        
        drpsi = eq.interp_psi.ev(rmax_lcfs, 0, dx=1)
        dzpsi = eq.interp_psi.ev(rmax_lcfs, 0, dy=1)
        
        bp = np.sqrt(drpsi**2+dzpsi**2)/rmax_lcfs
        
        # Use the eich scaling to get the heat flux width
        lamq = 6.3e-4 * (bp**(-1.2))
        # Convert it to a heat flux width in flux coordinates
        lamq_psi = lamq * drpsi
        
        # Extend out the grid to the maximum of psi
        psin_extra = np.arange(int((np.max(eq.psirz) - eq.psix)/eq.psix/0.001))*0.001 + 1.001
        # Compute an extra width to the potential that has continuous phi and dphi/dpsi using
        # C1 exp(-lamq_psi * (psi-psi_lcfs)) + C2 * (psi - psi_lcfs) * exp(-lamq_psi * (psi-psi_lcfs))
        c1 = phi_data[-1]
        c2 = (-epsi[-1]*lamq_psi + c1)
        psi_shifted = (psin_extra - 1)* eq.psix / lamq_psi
        phi_extra = (c1 + c2 * psi_shifted) * np.exp(-psi_shifted)
        
        ## Interpolation function for phi. Should at some point think about parallel connection
        fields._phi_data = np.concatenate((phi_data, phi_extra))
        fields._phi_x = np.concatenate((pf.ti.x, psin_extra)) * eq.psix
        fields.phimult = 1.0
        fields.interp_phi = scipy.interpolate.CubicSpline(fields._phi_x, fields._phi_data, bc_type='natural', extrapolate=True)

        return fields
'''

# %% Class for fields loaded from XGC data

class XgcZonalFieldHandler(ZonalFieldHandler):
    """
    XGC zonal field frozen at a specific time index
    """

    def __init__(self, eq: Equilibrium, xgcdata: netCDF4.Dataset, tind: int):
        super().__init__(eq)
        self.xgcdata = xgcdata

        ## Load the zonal potential. Note we should convert to kV
        self._zpot = self.xgcdata['pot00'][:,:]*1e-3
        # The psi grid on which the zonal potential is defined
        self._zpot_psi = self.xgcdata['psi00'][:]

        self._phi_data = self._zpot[tind,:]
        self._phi_x = self._zpot_psi
        self.phimult = 1.0
        # TODO: Do something about stuff outside the LCFS
        self.interp_phi = scipy.interpolate.CubicSpline(self._phi_x, self._phi_data, bc_type='natural', extrapolate=True)

    def set_tind(self, tind: int):
        self._phi_data = self._zpot[tind,:]
        self.interp_phi = scipy.interpolate.CubicSpline(self._phi_x, self._phi_data, bc_type='natural', extrapolate=True)

# %% Define XGC Field handler class

class XgcFieldHandler(FieldHandler):
    """
    Class which handles loading of field data from the XGC data file, and exposes interpolation
    functions for using the field data. The basic strategy is to have a block of ~3 time indices
    loaded at each time into a cache. Whenever a new time index is requested, the cache will
    update with new field data.

    Note this class also has a time-independent rotating frame mode which changes the behavior
    of request_interp
    """
    def __init__(self, xgcdata, geom: XgcGeomHandler, tor_sectors = 3):
        """
        Upon loading the new field data, some information (i.e. spline coefficients) will be
        precomputed and stored into the cache.

        tor_sectors: the number of toroidal sectors to repeat the mesh data over
        """
        self.xgcdata = xgcdata
        self.geom = geom
        self.eq = geom.eq
        
        ## Set up stuff related to the cache for the potential
        # Maximum number of time indices to keep cached
        self.max_cache = 4
        # Array holding cached interpolation functions
        self.interp_cache = []
        # For doing updates
        self.tind_cache = -1

        ## Load the zonal potential
        self.zpot = self.xgcdata['pot00'][:,:]
        # The psi grid on which the zonal potential is defined
        self.zpot_psi = self.xgcdata['psi00'][:]

        ## Time
        self.t = self.xgcdata['t'][:]

        ## Info on whether or not the time is frozen
        self.tind_frozen = None

        # Set up zonal interpolation function
        self.interp_zpot = [None] * len(self.t)
        for tind in range(len(self.t)):
            # TODO: Do something about stuff outside the LCFS
            self.interp_zpot[tind] = scipy.interpolate.CubicSpline(self.zpot_psi, self.zpot[tind,:])

        ## Whether or not we apply a gyroaveraging filter
        self.jmat = None

        ## Number of toroidal sectors
        self.tor_sectors = tor_sectors

    def prepare_interp(self, tind: int) -> InterpTuple:
        """
        Load the non-zonal fields at a given time index and compute the spline data necessary
        for interpolation.
        """
        geom = self.geom

        # Load non-zonal component of the potential at the requested time index
        npot = self.xgcdata['dpot'][tind,:,:]
        
        interp_npot_tind = [None] * npot.shape[0]

        for kphi in range(npot.shape[0]):
            if self.jmat is None:
                npot_kphi = npot[kphi,:]
            else:
                npot_kphi = self.jmat @ npot[kphi,:]
            dr = geom.diff_r @ npot_kphi
            dz = geom.diff_z @ npot_kphi
            interp_npot_tind[kphi] = CubicTriInterpolatorMemoized(geom.rz_tri, npot_kphi, kind='min_E', dz=(dr,dz))

        return InterpTuple(zonal=self.interp_zpot[tind], mesh=(self.tor_sectors, interp_npot_tind))

    def request_interp(self, tind: int) -> InterpTuple:
        """
        Requests the interpolation at the given time index. Returns a tuple containing
        the zonal and non-zonal interpolators
        """
        # If the cache is empty, initialize it and return the interpolation functions
        if len(self.interp_cache) == 0:
            self.tind_cache = tind
            self.interp_cache.append(self.prepare_interp(tind))
            return self.interp_cache[0]
        
        # If the requested tind lies inside the cache, return it
        elif tind >= self.tind_cache and tind < self.tind_cache + len(self.interp_cache):
            return self.interp_cache[tind - self.tind_cache]
        
        # If the requested tind is one index beyond the cache, extend the cache and return it
        elif tind == self.tind_cache + len(self.interp_cache):
            self.interp_cache.append(self.prepare_interp(tind))
            
            # If the cache is too big, cull it
            if len(self.interp_cache) > self.max_cache:
                self.interp_cache.pop(0)
                self.tind_cache += 1
            
            return self.interp_cache[-1]
        
        # If the requested tind is one before the cache, extend the cache and return it
        elif tind == self.tind_cache - 1:
            self.interp_cache.insert(0, self.prepare_interp(tind))
            self.tind_cache -= 1
                
            # If the cache is too big, cull it
            if len(self.interp_cache) > self.max_cache:
                self.interp_cache.pop()
            
            return self.interp_cache[0]
                
        # Otherwise, rebuild the cache
        else:
            print('Warning: rebuilding cache')
            self.interp_cache = []
            return self.request_interp(tind)

    def request_tind(self, t: float) -> tuple[int, float]:
        # NOTE: XGC data is stored in sec, but unit conventions are in millisec
        tind = np.searchsorted(self.t, t*1e-3)
        tfrac = (t*1e-3 - self.t[tind]) / (self.t[tind+1] - self.t[tind])
        return (tind, tfrac)

    def scale_conversion(self):
        # NOTE: XGC data is in V, but unit conventions are in kV
        return 1e-3
    
    def set_jmat(self, jmat):
        """
        Sets the jmat filter matrix. Also resets the cache.
        """
        self.jmat = jmat
        self.interp_cache = []
        self.tind_cache = -1

class XgcLinearFieldHandler(XgcFieldHandler):
    """
    Same as the XgcFieldHandler, but uses linear interpolation on triangles rather than
    cubic interpolation
    """
    def __init__(self, xgcdata, geom: XgcGeomHandler):
        super().__init__(xgcdata, geom)

    def prepare_interp(self, tind: int) -> InterpTuple:
        geom = self.geom

        # Load non-zonal component of the potential at the requested time index
        npot = self.xgcdata['dpot'][tind,:,:]
        
        interp_npot_tind = [None] * npot.shape[0]

        for kphi in range(npot.shape[0]):
            dr = geom.diff_r @ npot[kphi,:]
            dz = geom.diff_z @ npot[kphi,:]
            interp_npot_tind[kphi] = tri.LinearTriInterpolator(geom.rz_tri, npot[kphi,:])

        return InterpTuple(zonal=self.interp_zpot[tind], mesh=(self.tor_sectors, interp_npot_tind))

# %% Field handler from Gauss-Hermite coefficients

# Normalization coefficients for Gauss-Hermite functions
n1 = np.sqrt(2)
n2 = np.sqrt(8)

class GaussHermiteFunction(BallooningModeInterpolator):
    """
    A class that wraps an evaluation for Gauss-Hermite functions
    """
    def __init__(self, params: np.ndarray, coefs: np.ndarray):
        """
        Set up a GaussHermiteFunction with given params and coefs.

        params is a numpy array of [mu_q, mu_eta, sigma_q, sigma_eta]
        coefs is a numpy array of the Gauss-Hermite coefficients, with order (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), etc.

        Internally, coefs will be converted to also compute gradients
        """

        # Store the parameters
        self.params = params
        # Store the coefficients as complex values
        if np.iscomplexobj(coefs):
            self.coefs = coefs
        else:
            self.coefs = coefs[::2] + 1j*coefs[1::2]
        
        # Old code: uses np.polynomial.hermite.hermval2d, which may be slow
        '''
        # Compute derivatives of the Hermite coefficients
        dcoefs0 = np.polynomial.hermite.hermder(coefs, axis=0)
        dcoefs1 = np.polynomial.hermite.hermder(coefs, axis=1)

        if dcoefs0.shape[0] != coefs.shape[0]:
            dcoefs0 = np.concatenate((dcoefs0, np.zeros((1, coefs.shape[1]))), axis=0)
        if dcoefs1.shape[1] != coefs.shape[1]:
            dcoefs1 = np.concatenate((dcoefs1, np.zeros((coefs.shape[0], 1))), axis=1)

        # Stack the coefficients together on the last axis
        self.coefs = np.stack((coefs, dcoefs0, dcoefs1), axis=-1)'
        '''

    def __call__(self, q, eta, gradient=True):
        mu_q, mu_eta, sigma_q, sigma_eta = self.params
        c = self.coefs

        ## Normalized coordinates
        z_q = (q - mu_q) / sigma_q
        z_eta = (eta - mu_eta) / sigma_eta

        ## Compute Gaussian prefactor
        g = np.exp(-0.5*(z_q**2 + z_eta**2))

        ## Start computing Hermite polynomials.
        p = c[0]*np.ones(z_q.shape, dtype=complex)

        if len(c) > 1:
            p += (c[1] * z_q + c[2] * z_eta)/n1
        if len(c) > 3:
            p += (c[3] * (4*z_q**2 - 2) + c[5] * (4*z_eta**2 - 2)) / n2 + (c[4] * z_q * z_eta) / n1**2
        
        if gradient:
            p_q = np.zeros_like(p)
            p_eta = np.zeros_like(p)

            if len(c) > 1:
                p_q += c[1] / n1
                p_eta += c[2] / n1
            if len(c) > 3:
                p_q += c[3] * 8 * z_q / n2 + c[4] * z_eta / n1**2
                p_eta += c[5] * 8 * z_eta / n2 + c[4] * z_q / n1**2

        ## Compute the function and its derivatives
        if gradient:
            return (p*g, (p_q - z_q*p) * g/sigma_q, (p_eta - z_eta*p) * g/sigma_eta)
        else:
            return p * g


class GaussHermiteFieldHandler(FieldHandler):
    """
    Field handler for fields constructed from Gauss-Hermite coefficients
    """
    def __init__(self, params):
        self.params = params

        # TODO: Set up a PlanarFluxCoordInterpolator

    def scale_conversion(self):
        # NOTE: XGC data is in V, but unit conventions are in kV
        return 1e-3