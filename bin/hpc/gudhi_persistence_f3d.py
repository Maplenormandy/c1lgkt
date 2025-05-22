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


# %% Define function to plot the data

def load_f3d(f3d_reader):
    """
    Loads the desired 3d field data from the f3d file reader.
    """

    # Set range of flux surfaces to plot
    ksurf0, ksurf1 = 180, 220

    # Load the electron density data
    e_den = f3d_reader.read('e_den', start=[0, 0], count=[132048, 16])

    # Set the electron density data to NaN for the flux surfaces outside the range
    e_den[:geom.psi_surf[ksurf0],:] = np.nan
    e_den[geom.psi_surf[ksurf1+1]:,:] = np.nan

    return e_den

def compute_all_persistences(e_den):
    # Compute the persistence diagrams
    p_lower = compute_sublevel_persistence(e_den)
    p_upper = compute_sublevel_persistence(-e_den)

    return p_lower, p_upper

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
    p_lower = []
    p_upper = []

    for p_list in p_all:
        p_lower.append(p_list[0])
        p_upper.append(p_list[1])

    p0_lower, p1_lower = compact_persistence(p_lower)
    p0_upper, p1_upper = compact_persistence(p_upper)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    ax.set_aspect('equal', adjustable='box')

    ax.scatter( p0_lower[0],  p0_lower[1], c='C0', s=((p0_lower[1]-p0_lower[0])*3e-7)**2, alpha=0.5)
    ax.scatter( p1_lower[0],  p1_lower[1], c='C1', s=((p1_lower[1]-p1_lower[0])*3e-7)**2, alpha=0.5)
    ax.scatter(-p0_upper[0], -p0_upper[1], c='C2', s=((p0_upper[1]-p0_upper[0])*3e-7)**2, alpha=0.5)
    ax.scatter(-p1_upper[0], -p1_upper[1], c='C3', s=((p1_upper[1]-p1_upper[0])*3e-7)**2, alpha=0.5)

    #ax.plot([2.3e8, 3.3e8], [2.3e8, 3.3e8], color='k', linestyle='--')

    ax.set_title(f'Persistence diagram at t = {t[tind]*1e3:.3f} ms')

    #ax.set_xlim(2.3e8, 3.3e8)
    #ax.set_ylim(2.3e8, 3.3e8)

    plt.savefig(f'./outputs/f3d_tda/persistence_diagram_{tind}.png', dpi=100)

    plt.close(fig)

def analyze_frame(tind):
    ## Get the right file to load
    step = tind*20
    f3d_file = file_root + f'/xgc.f3d.{step:05d}.bp'
    print(f3d_file, flush=True)

    p_all = []
    with FileReader(f3d_file) as f3d_reader:
        e_den = load_f3d(f3d_reader)

        for nphi in range(16):
            p_list = compute_all_persistences(e_den)
            p_all.append(p_list)

    with open(f'./outputs/f3d_tda/persistence_diagram_{tind}.pkl', 'wb') as f:
        pickle.dump(p_all, f)

    plot_persistence_diagram(tind, p_all)

    print(f'Finished {tind}', flush=True)

    return True


# %% Main function to run the code

if __name__ == '__main__':
    n_procs = mp.cpu_count()

    print(f'Using {n_procs} processes', flush=True)


    tinds = np.arange(100, 500, 1, dtype=int)

    #tind = 350
    #with open(f'./outputs/phase_space_tda/persistence_diagram_{tind}.pkl', 'rb') as f:
    #    p_list = pickle.load(f)
    #plot_persistence_diagram(tind, p_list)

    analyze_frame(350)

    # Set up the multiprocessing pool
    #with mp.Pool(processes=n_procs) as pool:
    #    # Use the pool to process the files in parallel
    #    results = pool.map(analyze_frame, tinds)

    


