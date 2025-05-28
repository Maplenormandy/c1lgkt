"""
Codes to deal with analysis of XGC distribution functions
"""

from ..fields.equilibrium import Equilibrium
from ..fields.geometry_handlers import XgcGeomHandler
from ..fields.field_handlers import XgcZonalFieldHandler, XgcFieldHandler
from ..particles import particle_tools
from ..particles.analysis_setup import AnalysisData, IntegralTuple
from ..particles.particle_motion import ParticleParams, RotatingFrameInfo

import numpy as np
import scipy.interpolate

from adios2 import FileReader



# %%


def compute_interpolated_distribution(tind, kphi, ksurf_lim, mesh_file, f0_file, pp: ParticleParams, integrals: IntegralTuple, geom: XgcGeomHandler, xgcFields: XgcFieldHandler):
    """
    Compute the interpolated distribution function on a (mu, K) slice at a given time index and toroidal (half-integer) plane.
    """
    ## Load the reference temperatures and other info that are used for normalization
    with FileReader(mesh_file) as s:
        f0_T_ev = s.read('f0_T_ev')
        f0_smu_max = s.read('f0_smu_max')
        f0_vp_max = s.read('f0_vp_max')

    # NOTE: it appears that f0_T_ev[1,:] is the ion temperature
    f0_Ti_ev = f0_T_ev[1,:]

    eq = geom.eq

    ## Unpack the integrals
    rotating_frame, ham0, lphi0, mu0 = integrals
    t0 = rotating_frame.t0

    # Set range of flux surfaces to plot
    ksurf0, ksurf1 = ksurf_lim

    ## Prepare arrays for holding the distribution function data on the mesh
    # These hold the interpolated values of f_xgc on the mesh, one array for the
    # positive and negative branches of the constant (mu, K) slice
    fp_interp = np.zeros(geom.nnode)
    fn_interp = np.zeros(geom.nnode)

    # These hold the values of f_i = f_xgc / N where N = (2 pi / m)^(3/2) * sqrt(2 mu / B) * Bstar_||
    # f_i is the physical gyrocenter distribution function
    fp_physical = np.zeros(geom.nnode)
    fn_physical = np.zeros(geom.nnode)

    # This masks out the nodes that are not in the range of flux surfaces, or
    # do not intersect the (mu, K) slice
    f_mask = np.zeros(geom.nnode, dtype=bool)


    ## Load all of the distribution function data at once
    # First and last node to load
    n0 = geom.breaks_surf[ksurf0]
    n1 = geom.breaks_surf[ksurf1+1]

    with FileReader(f0_file) as f0_reader:
        # Get number of toroidal planes, and dimension in mu and v_||
        nphi = f0_reader.read('nphi')
        nmu = f0_reader.read('mudata')
        nvp = f0_reader.read('vpdata')

        # Load the distribution function data
        f_xgc = np.squeeze(f0_reader.read('i_f', start=[kphi, 0, n0, 0], count=[1, nmu, n1-n0, nvp]))

    # Prepare the coordinate grids
    xgc_vpara = np.linspace(-f0_vp_max, f0_vp_max, nvp)
    xgc_vperp = np.linspace(0, f0_smu_max, nmu)

    # Get varphi of the toroidal plane; note that f0 is on half-integer planes starting at -1/2 (?)
    varphi = 2*np.pi/48 * (kphi-0.5)

    varphi_arr = np.ones(n1-n0) * varphi
    r_arr = geom.rz_node[n0:n1,0]
    z_arr = geom.rz_node[n0:n1,1]

    kll_arr, pll_mean_arr = particle_tools.compute_parallel_energy(t0, r_arr, z_arr, varphi_arr, np.ones(n1-n0)*mu0, ham0, lphi0, eq, pp, xgcFields, rotating_frame)

    ## Iterate over the nodes
    for knode in range(n0, n1):
        ## Get the initial parallel energy and mean parallel velocity
        kll = kll_arr[knode-n0]
        pll_mean = pll_mean_arr[knode-n0]
        
        if kll < 0:
            # If parallel energy is negative, skip this node
            f_mask[knode] = False
            continue

        ## Unmask nodes that we iterate over
        f_mask[knode] = True

        ## Spatial coordinates
        # Get the (R,Z) coordinates of the node
        r, z = geom.rz_node[knode,:]

        ## Magnetic geometry stuff
        psi_ev, ff_ev = eq.compute_psi_and_ff(np.array([r]), np.array([z]))
        bv, bu, modb, gradmodb, curlbu = eq.compute_geom_terms(np.array([r]), psi_ev, ff_ev)

        ## Compute normalized velocities
        # Thermal velocity in meters/millisec, which are my unit conventions
        vt = np.sqrt(f0_Ti_ev[knode]*1e-3 / pp.m)
        # Normalized perpendicular velocity
        vperp_n = np.sqrt(2 * mu0 * modb[0] / pp.m) / vt

        # Positive and negative parallel velocities
        vllp = (pll_mean + np.sqrt(2 * pp.m * kll) / pp.m)
        vlln = (pll_mean - np.sqrt(2 * pp.m * kll) / pp.m)
        vparap_n = vllp / vt
        vparan_n = vlln / vt

        ## Load and interpolate the distribution function
        xgc_interp = scipy.interpolate.RegularGridInterpolator((xgc_vperp, xgc_vpara), f_xgc[:,knode-n0,:], method='cubic')
        fp_interp[knode] = xgc_interp((vperp_n, vparap_n))
        fn_interp[knode] = xgc_interp((vperp_n, vparan_n))

        ## Compute the conversion factor from XGC distribution function to physical distribution function
        bstarp = bv[:,0] + (pp.m / pp.z) * curlbu[0] * vllp
        bstarn = bv[:,0] + (pp.m / pp.z) * curlbu[0] * vlln
        bstarllp = np.sum(bu[:,0]*bstarp, axis=0)
        bstarlln = np.sum(bu[:,0]*bstarn, axis=0)

        fp_physical[knode] = np.sqrt(modb[0]/2/mu0) / bstarllp * fp_interp[knode] * (pp.m/(2*np.pi))**(1.5)
        fn_physical[knode] = np.sqrt(modb[0]/2/mu0) / bstarlln * fn_interp[knode] * (pp.m/(2*np.pi))**(1.5)

    fp_physical[np.logical_not(f_mask)] = np.nan
    fn_physical[np.logical_not(f_mask)] = np.nan

    return fp_physical, fn_physical, f_mask
