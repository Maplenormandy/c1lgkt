# %% -*- coding: utf-8 -*-
"""
@author: maple

Code to analyze slices of the distribution function from XGC data using multiprocessing
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
from c1lgkt.particles import analysis_setup, particle_motion, particle_tools
from c1lgkt.phase_space.xgc_analysis import compute_interpolated_distribution

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

file_root = R'/pscratch/sd/n/normandy/D3D_elec_rgn1_run4'
mesh_file = file_root + '/xgc.f0.mesh.bp'

# XGC time data
t = xgcdata['t'][:]

## Compute zonal flows
zpot = xgcdata['pot00'][:,:]
zpot_psi = xgcdata['psi00'][:]

zfield = np.gradient(zpot, axis=1) / np.gradient(zpot_psi)[np.newaxis,:]

# %% Analaysis data

pp = particle_motion.deut
ksurf0, ksurf1 = 174, 221
ksurf_ref = 196

analysis_data = dict(
    name='trapped',
    ksurf=ksurf_ref,
    ev0=0.78,
    xi0=np.sqrt(0.33),
    pp=pp,
)

output_folder = f'./outputs/phase_space_analysis_{analysis_data["name"]}'


# %% Functions related to computing persistence diagrams

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
    # Compute the persistence diagrams
    pp_lower = compute_sublevel_persistence(fp_physical)
    pn_lower = compute_sublevel_persistence(fn_physical)
    pp_upper = compute_sublevel_persistence(-fp_physical)
    pn_upper = compute_sublevel_persistence(-fn_physical)

    return pp_lower, pn_lower, pp_upper, pn_upper

# %% Functions related to plotting persistence diagrams

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

    plt.savefig(f'{output_folder}/persistence_diagram_{tind}.png', dpi=300)

    plt.close(fig)

# %% Functions related to plotting the distribution function


def plot_phase_space(tind, fp_physical, fn_physical):
    # Set up colorbar
    #tw_repeated = mpl.cm.twilight(np.mod(np.linspace(0, 2, 256),1))
    twr_cmap = mpl.cm.twilight_shifted_r

    # Set up range for colorbar
    f_max = np.nanpercentile([fp_physical, fn_physical], 85)
    f_min = np.nanpercentile([fp_physical, fn_physical], 10)

    # Pick out q=2 surface
    #ksurf = np.searchsorted(-geom.q_surf, 2.0)
    #rz_surf = geom.rz_node[geom.breaks_surf[ksurf]:geom.breaks_surf[ksurf+1],:]

    # Set up the figure
    fig, axs = plt.subplots(1, 3, width_ratios=[1, 1, 1], figsize=(19.2, 10.8))

    axs[0].set_aspect('equal', adjustable='box')
    axs[0].tripcolor(geom.rz_tri, fp_physical, shading='gouraud', rasterized=True, cmap=twr_cmap, vmin=f_min, vmax=f_max)
    #axs[0].plot(rz_surf[:,0], rz_surf[:,1], c='k', ls=':')

    axs[0].set_title(R'$v_\parallel > 0$')

    eq.plot_magnetic_geometry(axs[0], alpha=0.25)

    axs[0].set_xlim([1.1, 2.3])
    axs[0].set_ylim([-0.95, 0.85])

    #ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    axs[1].set_aspect('equal', adjustable='box')
    #plt.axis('equal')
    pc = axs[1].tripcolor(geom.rz_tri, fn_physical, shading='gouraud', rasterized=True, cmap=twr_cmap, vmin=f_min, vmax=f_max)
    #axs[1].plot(rz_surf[:,0], rz_surf[:,1], c='k', ls=':')

    axs[1].set_title(R'$v_\parallel < 0$')

    eq.plot_magnetic_geometry(axs[1], alpha=0.25)

    axs[1].set_xlim([1.1, 2.3])
    axs[1].set_ylim([-0.95, 0.85])

    cax = axs[0].inset_axes([0.6, 0.07, 0.4, 0.01])

    plt.colorbar(pc, cax=cax, orientation='horizontal', label='ion gyrocenter distrib. $f_i$')

    tind0 = 100
    pc = axs[2].pcolormesh(zpot_psi[130:-30], t[tind0:]*1e3, -zfield[tind0:,130:-30])
    axs[2].set_title('zonal flows')
    axs[2].axhline(t[tind]*1e3, c='tab:red', ls='--')
    axs[2].set_ylabel(R'$t$ [ms]')
    axs[2].set_xlabel(R'$\psi$')

    plt.colorbar(pc, ax=axs[2], label=R'$d\langle\phi\rangle/d\psi$')

    plt.tight_layout(pad=0.08)

    plt.savefig(f'{output_folder}/phase_space_{tind:04d}.png', dpi=300, bbox_inches='tight')

    plt.close(fig)


# %% Actual function for controlling the analysis

def analyze_frame(tind):
    ## Set up the analysis info
    omega_frame = -uph[tind,ksurf_ref]*geom.q_surf[ksurf_ref]*1e-3
    rotating_frame = particle_motion.RotatingFrameInfo(xgcdata['t'][tind], omega_frame, tind)

    ## Load the zonal fields
    zonalFields = XgcZonalFieldHandler(eq, xgcdata, tind)

    analysis = analysis_setup.MidplaneAnalysisData(
        **analysis_data,
        geom=geom,
        zonalFields=zonalFields,
        frame=rotating_frame
    )

    ## Get the right file to load
    step = tind*20
    f0_file = file_root + f'/xgc.orbit.f0.{step:05d}.bp'
    print(f0_file, flush=True)

    ## Choose which particle properties to use
    integrals = analysis.get_reference_integrals()
    jmat = geom.assemble_jmat(integrals.mu, pp.m, pp.z)
    xgcFields.set_jmat(jmat)

    ## Iterate over the toroidal planes
    p_all = []
    
    for kphi in range(16):
        # Start performance timer
        start_time = time.perf_counter()

        # Compute the interpolated distribution function on the mesh
        fp_physical, fn_physical, f_mask = compute_interpolated_distribution(
            tind=tind,
            kphi=kphi,
            ksurf_lim=(ksurf0, ksurf1),
            mesh_file=mesh_file,
            f0_file=f0_file,
            pp=analysis.pp,
            integrals=integrals,
            geom=geom,
            xgcFields=xgcFields
        )

        # Compute persistences for this toroidal plane
        p_list = compute_all_persistences(fp_physical, fn_physical)
        p_all.append(p_list)

        # Print the time taken for this kphi
        end_time = time.perf_counter()
        print(f'Finished tind {tind} kphi {kphi} in {end_time-start_time:.2f} seconds', flush=True)

        # If this is the first kphi, save and plot plot the phase space
        if kphi == 0:
            ## Save the phase space data to a file
            f_phi = np.empty((2, geom.breaks_surf[ksurf1+1]-geom.breaks_surf[ksurf0]))
            f_phi[0, :] = fp_physical[geom.breaks_surf[ksurf0]:geom.breaks_surf[ksurf1+1]]
            f_phi[1, :] = fn_physical[geom.breaks_surf[ksurf0]:geom.breaks_surf[ksurf1+1]]
            np.savez(f'{output_folder}/phase_space_{tind:04d}.npz', f_phi=f_phi, t=t[tind], ksurf0=ksurf0, ksurf1=ksurf1)

            ## Plot the phase space
            plot_phase_space(tind, fp_physical, fn_physical)

    

    ## Save the persistence diagrams to a file
    with open(f'{output_folder}/persistence_diagram_{tind}.pkl', 'wb') as f:
        pickle.dump(p_all, f)

    ## Plot the persistence diagram
    plot_persistence_diagram(tind, p_all)


    ## Print the completion message
    print(f'Finished {tind}', flush=True)

    return True


# %% Main function to run the code

if __name__ == '__main__':
    n_procs = mp.cpu_count()

    print(f'Running {analysis_data["name"]}')
    print(f'Using {n_procs} processes', flush=True)

    tinds = np.arange(100, 500, 5, dtype=int)

    #tind = 400
    #with open(f'./outputs/phase_space_tda/persistence_diagram_{tind}.pkl', 'rb') as f:
    #    p_all = pickle.load(f)
    #plot_persistence_diagram(tind, p_all)

    #analyze_frame(370)

    # Set up the multiprocessing pool
    with mp.Pool(processes=n_procs) as pool:
        # Use the pool to process the files in parallel
        results = pool.map(analyze_frame, tinds)

    


