# %% -*- coding: utf-8 -*-
"""
@author: maple

Code to extract summary data from f3d files
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from adios2 import FileReader
from netCDF4 import Dataset

from c1lgkt.fields.equilibrium import Equilibrium
from c1lgkt.fields.geometry_handlers import XgcGeomHandler

import multiprocessing as mp

import glob
import os

import time

mpl.use('Agg')

# %% Load data files

eq = Equilibrium.from_eqdfile(R'./outputs/D3D141451.eqd')

xgcdata = Dataset('/global/cfs/cdirs/m3736/Rotation_paper_data/XGC1.nc')

geom_files = {
    'ele_filename': R'/global/cfs/cdirs/m3736/XGC1_H/D3D_elec_rgn1_run4/Seo.eqd.ele',
    'fdmat_filename': R'./outputs/fdmat.pkl',
    'min_e_filename': R'./outputs/min_E_mat.pkl'
}
geom = XgcGeomHandler(eq, xgcdata, theta0_mode='midplane', **geom_files)

# %% Functions to load and compute the needed data

file_root = R'/global/cfs/cdirs/m3736/XGC1_H/D3D_elec_rgn1_run4'

var_names = [
    'i_T_perp',
    'e_den',
]

def compute_summary_data(tind):
    # Get the right file to load
    step = tind*20
    f3d_file = os.path.join(file_root, f'xgc.f3d.{step:05d}.bp')

    summary_data = np.empty((geom.nsurf, len(var_names)*3))

    # Read the f3d file using FileReader
    with FileReader(f3d_file) as s:
        for nvar, var in enumerate(var_names):
            # Read data from all the poloidal planes
            data = s.read(var, start=[0,0], count=[geom.nnode, 16])

            # Compute the flux surface average
            fsavg, fsavg_node = geom.flux_surf_avg(data, nodal=True)

            # Compute the centered moments
            fsvar = geom.flux_surf_avg((data - fsavg_node[:,np.newaxis])**2)
            fsskw = geom.flux_surf_avg((data - fsavg_node[:,np.newaxis])**3)

            # Store the results in the summary_data array
            summary_data[:, 0 + 3*nvar] = fsavg
            summary_data[:, 1 + 3*nvar] = fsvar
            summary_data[:, 2 + 3*nvar] = fsskw
        
    return summary_data


if __name__ == '__main__':
    # Set up list of time indices to process
    tinds = np.arange(1, 501, 1, dtype=int)

    # Setup number of processors
    n_procs = mp.cpu_count()
    #n_procs = 16
    print('n_procs:', n_procs)

    # Set up the multiprocessing pool and process the tinds
    with mp.Pool(processes=n_procs) as pool:
        results = pool.map(compute_summary_data, tinds)

    # Combine the results into a single array
    summary_data = np.stack(results, axis=0)

    # Split the summary data into separate variables
    summary_vars = dict()
    for nvar, var in enumerate(var_names):
        summary_vars[var + '_avg'] = summary_data[:, :, 0 + 3*nvar]
        summary_vars[var + '_var'] = summary_data[:, :, 1 + 3*nvar]
        summary_vars[var + '_skw'] = summary_data[:, :, 2 + 3*nvar]

    # Save the summary data to a file
    np.savez_compressed('./outputs/summary_data.npz', **summary_vars)