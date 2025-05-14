# %% -*- coding: utf-8 -*-
"""
@author: maple

Code to plot slices of the distribution function from XGC data using multiprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate

from adios2 import FileReader
from netCDF4 import Dataset

from c1lgkt.fields.equilibrium import Equilibrium
from c1lgkt.fields.geometry_handlers import XgcGeomHandler
from c1lgkt.fields.field_handlers import XgcZonalFieldHandler, XgcFieldHandler
from c1lgkt.particles import particle_motion, particle_tools

import multiprocessing as mp

from tqdm import tqdm

import glob
import os

import time

mpl.use('Agg')

# %% Load basic data files

eq = Equilibrium.from_eqdfile(R'./outputs/D3D141451.eqd')

xgcdata = Dataset('/global/cfs/cdirs/m3736/Rotation_paper_data/XGC1.nc')

geom_files = {
    'ele_filename': R'/global/cfs/cdirs/m3736/XGC1_H/D3D_elec_rgn1_run4/Seo.eqd.ele',
    'fdmat_filename': R'./outputs/fdmat.pkl',
    'min_e_filename': R'./outputs/min_E_mat.pkl'
    
}
geom = XgcGeomHandler(eq, xgcdata, theta0_mode='midplane', **geom_files)

xgcFields = XgcFieldHandler(xgcdata, geom)

uph = np.load('./outputs/phase_vel.npz')['u_lstsq']



# %% Load the reference temperatures and other info that are used for normalization

file_root = R'/global/cfs/cdirs/m3736/XGC1_H/D3D_elec_rgn1_run4'
mesh_file = file_root + '/xgc.f0.mesh.bp'

with FileReader(mesh_file) as s:
    f0_T_ev = s.read('f0_T_ev')
    f0_smu_max = s.read('f0_smu_max')
    f0_vp_max = s.read('f0_vp_max')

# NOTE: it appears that f0_T_ev[1,:] is the ion temperature
f0_Ti_ev = f0_T_ev[1,:]

## Compute zonal flows
zpot = xgcdata['pot00'][:,:]
zpot_psi = xgcdata['psi00'][:]

zfield = np.diff(zpot, axis=1) / np.diff(zpot_psi)[np.newaxis,:]
t = xgcdata['t'][:]

# %% Analysis presets


#analysis = {
#    'xi0': np.sqrt(0.67),
#    'name': 'trapped'
#}

analysis = {
    'xi0': np.sqrt(0.33),
    'name': 'passing'
}

# %% Define function to plot the data

def compute_initial_integrals(tind, ksurf, pp: particle_motion.ParticleParams):
    ## Set up the rotating frame
    # NOTE: Be careful about unit conventions, XGC is in sec while we work in millisec
    omega_frame = -uph[tind,ksurf]*geom.q_surf[ksurf]*1e-3
    rotating_frame = particle_motion.RotatingFrameInfo(xgcdata['t'][tind], omega_frame, tind)
    t0 = rotating_frame.t0

    ## Load the zonal fields
    zonalFields = XgcZonalFieldHandler(eq, xgcdata, tind)

    ## Set initial position, which in this case is (approximately) the outboard midplane
    r0 = eq.interp_router(geom.psi_surf[ksurf]/eq.psix)
    z0 = geom.zaxis
    x0 = np.array([r0, 0.0, z0])

    # Compute magnetic field at initial position
    bv = eq.compute_bv(x0[0], x0[2])
    modb = np.linalg.norm(bv)
    bu = bv / modb

    ## Compute the rotation frequency and the mean parallel velocity
    psi0 = eq.interp_psi.ev(x0[0], x0[2])
    omega0 = -zonalFields.interp_phi(psi0, nu=1)*zonalFields.scale_conversion()
    vll_mean = eq.interp_ff(psi0) * omega0 / modb

    ## Determine the initial values of the integrals

    # Particle kinetic energy in keV and cos(pitch angle)
    ev0 = 0.78
    xi0 = analysis['xi0']
    # Set the initial parallel velocity
    vll0 = vll_mean + pp.vt * xi0 * np.sqrt(ev0)
    # Initial magnetic moment
    mu0 = pp.m * (1-xi0**2) * (pp.vt * np.sqrt(ev0))**2 / 2 / modb

    # Compute initial value of the integrals
    ham0, lphi0 = particle_tools.compute_integrals_dk(t0, np.concatenate((x0, [vll0, mu0])), eq, pp, zonalFields, rotating_frame)

    # Set up the gyroaverage matrix
    jmat = geom.assemble_jmat(mu0, pp.m, pp.z)
    xgcFields.set_jmat(jmat)

    return rotating_frame, mu0, ham0, lphi0

def compute_interpolated_distribution(tind, f0_reader: FileReader, pp: particle_motion.ParticleParams, integrals):
    ## Unpack the integrals
    rotating_frame, mu0, ham0, lphi0 = integrals
    t0 = rotating_frame.t0

    # Set range of flux surfaces to plot
    ksurf0, ksurf1 = 180, 220

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

    # Get number of toroidal planes, and dimension in mu and v_||
    nphi = f0_reader.read('nphi')
    nmu = f0_reader.read('mudata')
    nvp = f0_reader.read('vpdata')

    # Prepare the coordinate grids
    xgc_vpara = np.linspace(-f0_vp_max, f0_vp_max, nvp)
    xgc_vperp = np.linspace(0, f0_smu_max, nmu)

    ## Load all of the distribution function data at once
    # First and last node to load
    n0 = geom.breaks_surf[ksurf0]
    n1 = geom.breaks_surf[ksurf1+1]

    #print('Loading data', flush=True)
    start_time = time.perf_counter()

    # Load the distribution function data
    f_xgc = np.squeeze(f0_reader.read('i_f', start=[0, 0, n0, 0], count=[1, nmu, n1-n0, nvp]))

    end_time = time.perf_counter()
    print(f'Loaded {n1-n0} nodes in {end_time-start_time:.2f} seconds', flush=True)

    ## Iterate over the nodes
    for knode in range(n0, n1):
        ## Unmask nodes that we iterate over
        f_mask[knode] = True

        ## Spatial coordinates
        # Get the (R,Z) coordinates of the node
        r, z = geom.rz_node[knode,:]
        # Get varphi of the toroidal plane; note that f0 is on half-integer planes starting at -1/2 (?)
        varphi = 2*np.pi/48 * -0.5

        ## Magnetic geometry stuff
        psi_ev, ff_ev = eq.compute_psi_and_ff(np.array([r]), np.array([z]))
        bv, bu, modb, gradmodb, curlbu = eq.compute_geom_terms(np.array([r]), psi_ev, ff_ev)

        ## Compute normalized velocities
        # Thermal velocity in meters/millisec, which are my unit conventions
        vt = np.sqrt(f0_Ti_ev[knode]*1e-3 / pp.m)
        # Normalized perpendicular velocity
        vperp_n = np.sqrt(2 * mu0 * modb[0] / pp.m) / vt
        # Parallel energy
        kll, pll_mean = particle_tools.compute_parallel_energy(t0, r, z, varphi, mu0, ham0, lphi0, eq, pp, xgcFields, rotating_frame)
        if kll < 0:
            # If parallel energy is negative, skip this node
            f_mask[knode] = False
            continue

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

def plot_phase_space(tind, fp_physical, fn_physical):
    # Set up colorbar
    tw_repeated = mpl.cm.twilight(np.mod(np.linspace(0, 2, 256),1))
    twr_cmap = mpl.colors.LinearSegmentedColormap.from_list('twilight_repeated', tw_repeated, N=256)

    # Set up range for colorbar
    f_max = np.nanpercentile([fp_physical, fn_physical], 95)
    f_min = np.nanpercentile([fp_physical, fn_physical], 5)

    # Pick out q=2 surface
    ksurf = np.searchsorted(-geom.q_surf, 2.0)
    rz_surf = geom.rz_node[geom.breaks_surf[ksurf]:geom.breaks_surf[ksurf+1],:]

    # Set up the figure
    fig, axs = plt.subplots(1, 3, width_ratios=[1, 1, 1], figsize=(19.2, 10.8))

    axs[0].set_aspect('equal', adjustable='box')
    axs[0].tripcolor(geom.rz_tri, fp_physical, shading='gouraud', rasterized=True, cmap=twr_cmap, vmin=f_min, vmax=f_max)
    axs[0].plot(rz_surf[:,0], rz_surf[:,1], c='k', ls=':')

    axs[0].set_title(R'$v_\parallel > 0$')

    axs[0].set_xlim([1.1, 2.3])
    axs[0].set_ylim([-0.95, 0.85])

    #ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    axs[1].set_aspect('equal', adjustable='box')
    #plt.axis('equal')
    pc = axs[1].tripcolor(geom.rz_tri, fn_physical, shading='gouraud', rasterized=True, cmap=twr_cmap, vmin=f_min, vmax=f_max)
    axs[1].plot(rz_surf[:,0], rz_surf[:,1], c='k', ls=':')

    axs[1].set_title(R'$v_\parallel < 0$')

    axs[1].set_xlim([1.1, 2.3])
    axs[1].set_ylim([-0.95, 0.85])

    cax = axs[0].inset_axes([0.6, 0.07, 0.4, 0.01])

    plt.colorbar(pc, cax=cax, orientation='horizontal', label='ion gyrocenter distrib. $f_i$')

    pc = axs[2].pcolormesh(zpot_psi[130:-30], t[tind0-1:]*1e3, -zfield[tind0:,130:-30])
    axs[2].set_title('zonal flows')
    axs[2].axhline(t[tind]*1e3, c='tab:red', ls='--')
    axs[2].set_ylabel(R'$t$ [ms]')
    axs[2].set_xlabel(R'$\psi$')

    plt.colorbar(pc, ax=axs[2], label=R'$d\langle\phi\rangle/d\psi$')

    plt.tight_layout(pad=0.08)

    plt.savefig(f'./outputs/phase_space_movie/phase_space_{analysis["name"]}_{tind:04d}.png', dpi=300, bbox_inches='tight')

    plt.close(fig)



def plot_frame(tind):
    ## Get the right file to load
    step = tind*20
    f0_file = file_root + f'/xgc.orbit.f0.{step:05d}.bp'
    print(f0_file, flush=True)

    ## Choose which particle properties to use
    pp = particle_motion.deut
    
    integrals = compute_initial_integrals(tind, 196, pp)
    with FileReader(f0_file) as f0_reader:
        fp_physical, fn_physical, f_mask = compute_interpolated_distribution(tind, f0_reader, pp, integrals)
    plot_phase_space(tind, fp_physical, fn_physical)

    print(f'Finished {tind}', flush=True)

    return True


# %% Main function to run the code

if __name__ == '__main__':
    n_procs = mp.cpu_count()

    print(f'Using {n_procs} processes', flush=True)


    tinds = np.arange(100, 500, 5, dtype=int)

    # Set up the multiprocessing pool
    with mp.Pool(processes=n_procs) as pool:
        # Use the pool to process the files in parallel
        results = pool.map(plot_frame, tinds)

    


