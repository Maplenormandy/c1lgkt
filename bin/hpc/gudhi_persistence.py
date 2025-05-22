# %% -*- coding: utf-8 -*-
"""
@author: maple

Code to plot slices of the distribution function from XGC data using multiprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate

import gudhi

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

import pickle

mpl.use('Agg')

# %% Load basic data files

eq = Equilibrium.from_eqdfile(R'./outputs/D3D141451.eqd')

xgcdata = Dataset('/pscratch/sd/n/normandy/XGC1.nc')

geom_files = {
    'ele_filename': R'/pscratch/sd/n/normandy/D3D_elec_rgn1_run4/Seo.eqd.ele',
    'fdmat_filename': R'./outputs/fdmat.pkl',
    'min_e_filename': R'./outputs/min_E_mat.pkl'
    
}
geom = XgcGeomHandler(eq, xgcdata, theta0_mode='midplane', **geom_files)

xgcFields = XgcFieldHandler(xgcdata, geom)

uph = np.load('./outputs/phase_vel.npz')['u_lstsq']



# %% Load the reference temperatures and other info that are used for normalization

file_root = R'/pscratch/sd/n/normandy/D3D_elec_rgn1_run4'
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


analysis = {
    'xi0': np.sqrt(0.67),
    'name': 'passing'
}

#analysis = {
#    'xi0': np.sqrt(0.33),
#    'name': 'trapped'
#}

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
    #jmat = geom.assemble_jmat(mu0, pp.m, pp.z)
    #xgcFields.set_jmat(jmat)

    return rotating_frame, mu0, ham0, lphi0

def compute_interpolated_distribution(tind, kphi, f0_reader: FileReader, pp: particle_motion.ParticleParams, integrals):
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
    

    # Load the distribution function data
    f_xgc = np.squeeze(f0_reader.read('i_f', start=[kphi, 0, n0, 0], count=[1, nmu, n1-n0, nvp]))

    
    #print(f'Loaded {n1-n0} nodes in {end_time-start_time:.2f} seconds', flush=True)

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
        # Parallel energy

        

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


def compute_sublevel_persistence(f):
    """
    Computes the sublevel persistence diagram for the given filtration values.
    """

    st = gudhi.simplex_tree.SimplexTree()

    # Insert vertices with filtration values into the tree.
    for i in range(geom.nnode_surf):
        if np.isfinite(f[i]):
            # Insert the vertex with the filtration value
            st.insert([i], f[i])

    # Insert edges and triangles with max filtration value of their vertices
    for tri in geom.rz_tri.triangles:
        edges = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]
        for edge in edges:
            f_edge = np.max([f[edge[0]], f[edge[1]]])
            if np.isfinite(f_edge):
                st.insert(edge, f_edge)

        f_tri = np.max([f[tri[0]], f[tri[1]], f[tri[2]]])
        if np.isfinite(f_tri):
            st.insert(tri, f_tri)

    # Compute the persistence diagram
    persistence = st.persistence()

    return persistence

def compute_all_persistences(fp_physical, fn_physical):
    # Set up normalization range
    f_max = np.nanpercentile([fp_physical, fn_physical], 95)
    f_min = np.nanpercentile([fp_physical, fn_physical], 5)

    # Compute the persistence diagrams
    pp_lower = compute_sublevel_persistence(fp_physical)
    pn_lower = compute_sublevel_persistence(fn_physical)
    pp_upper = compute_sublevel_persistence(-fp_physical)
    pn_upper = compute_sublevel_persistence(-fn_physical)

    return pp_lower, pn_lower, pp_upper, pn_upper


def compact_persistence(p):
    """
    This function takes a persistence diagram and returns numpy arrays of births and deaths suitable
    for plotting
    """
    
    # Put all the data into numpy arrays
    births = np.empty(len(p))
    deaths = np.empty(len(p))
    dims = np.empty(len(p), dtype=int)
    for i, (dim, (b, d)) in enumerate(p):
        births[i] = b
        deaths[i] = d
        dims[i] = dim

    # Sort the data by dimension
    dimsort = np.argsort(dims)
    births = births[dimsort]
    deaths = deaths[dimsort]
    dims = dims[dimsort]

    # Find the first instance of dimension 1
    ind = np.searchsorted(dims, 1)
    # Split the data into dimension 0 and dimension 1
    p0 = births[:ind], deaths[:ind]
    p1 = births[ind:], deaths[ind:]

    # Return the data
    return p0, p1

def plot_persistence_diagram(tind, p_all):
    pp_lower = []
    pn_lower = []
    pp_upper = []
    pn_upper = []

    for p_list in p_all:
        pp_lower.extend(p_list[0])
        pn_lower.extend(p_list[1])
        pp_upper.extend(p_list[2])
        pn_upper.extend(p_list[3])

    p0_lower, p1_lower = compact_persistence(pp_lower + pn_lower)
    p0_upper, p1_upper = compact_persistence(pp_upper + pn_upper)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.set_aspect('equal', adjustable='box')

    ax.scatter( p0_lower[0],  p0_lower[1], c='C0', s=((p0_lower[1]-p0_lower[0])*3e-7)**2, alpha=0.2)
    ax.scatter( p1_lower[0],  p1_lower[1], c='C1', s=((p1_lower[1]-p1_lower[0])*3e-7)**2, alpha=0.2)
    ax.scatter(-p0_upper[0], -p0_upper[1], c='C2', s=((p0_upper[1]-p0_upper[0])*3e-7)**2, alpha=0.2)
    ax.scatter(-p1_upper[0], -p1_upper[1], c='C3', s=((p1_upper[1]-p1_upper[0])*3e-7)**2, alpha=0.2)

    ax.plot([2.3e8, 3.3e8], [2.3e8, 3.3e8], color='k', linestyle='--')

    ax.set_title(f'Persistence diagram at t = {t[tind]*1e3:.3f} ms')

    ax.set_xlim(2.3e8, 3.3e8)
    ax.set_ylim(2.3e8, 3.3e8)

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')

    plt.tight_layout()

    plt.savefig(f'./outputs/phase_space_{analysis['name']}_tda/persistence_diagram_{tind}.png', dpi=300)

    plt.close(fig)

def analyze_frame(tind):
    ## Get the right file to load
    step = tind*20
    f0_file = file_root + f'/xgc.orbit.f0.{step:05d}.bp'
    print(f0_file, flush=True)

    ## Choose which particle properties to use
    pp = particle_motion.deut
    
    integrals = compute_initial_integrals(tind, 196, pp)
    p_all = []
    with FileReader(f0_file) as f0_reader:
        for kphi in range(16):
            start_time = time.perf_counter()
            fp_physical, fn_physical, f_mask = compute_interpolated_distribution(tind, kphi, f0_reader, pp, integrals)
            p_list = compute_all_persistences(fp_physical, fn_physical)
            p_all.append(p_list)
            end_time = time.perf_counter()
            print(f'Finished tind {tind} kphi {kphi} in {end_time-start_time:.2f} seconds', flush=True)

    with open(f'./outputs/phase_space_{analysis['name']}_tda/persistence_diagram_{tind}.pkl', 'wb') as f:
        pickle.dump(p_all, f)

    plot_persistence_diagram(tind, p_all)

    print(f'Finished {tind}', flush=True)

    return True


# %% Main function to run the code

if __name__ == '__main__':
    n_procs = mp.cpu_count()

    print(f'Running {analysis["name"]}')
    print(f'Using {n_procs} processes', flush=True)


    tinds = np.arange(100, 500, 5, dtype=int)

    #tind = 350
    #with open(f'./outputs/phase_space_tda/persistence_diagram_{tind}.pkl', 'rb') as f:
    #    p_all = pickle.load(f)
    #plot_persistence_diagram(tind, p_all)

    #analyze_frame(350)

    # Set up the multiprocessing pool
    with mp.Pool(processes=n_procs) as pool:
        # Use the pool to process the files in parallel
        results = pool.map(analyze_frame, tinds)

    


