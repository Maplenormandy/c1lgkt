# %% -*- coding: utf-8 -*-
"""
@author: maple

Code to plot some movies from XGC data using multiprocessing
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

ksurf = np.searchsorted(-geom.q_surf, 2.0)
rz_surf = geom.rz_node[geom.breaks_surf[ksurf]:geom.breaks_surf[ksurf+1],:]

# %% Get the glob of files that we want to read


def plot_2d(ax, data, label, cmap, vmin=None, vmax=None):
    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Plot the data
    pc = ax.tripcolor(geom.rz_tri, data, shading='gouraud', rasterized=True, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(rz_surf[:,0], rz_surf[:,1], c='k', ls=':')

    # Make the colorbar
    cax = ax.inset_axes([0.57, 0.06, 0.4, 0.01])
    plt.colorbar(pc, cax=cax, orientation='horizontal', label=label)

def test_file(f3d_file):
    # Read the f3d file using FileReader
    with FileReader(f3d_file) as s:
        step = s.read('step')

    print(step)

    return step

def process_file(f3d_file):
    """
    Process a single f3d file and extract the required data.
    """

    start_time = time.perf_counter()

    # Read the f3d file using FileReader
    with FileReader(f3d_file) as s:
        # Load step and check if we exist
        step = s.read('step')
        outfile = f'./outputs/f3d_movie/f3d_movie_{step:05d}.png'

        if os.path.exists(outfile):
            print(f'{step:05d} already exists, skipping.', flush=True)
            return False

        # Load the kinetic data
        e_den = s.read('e_den', start=[0, 0], count=[132048, 1])
        i_T_perp = s.read('i_T_perp', start=[0, 0], count=[132048, 1])
        e_T_perp = s.read('e_T_perp', start=[0, 0], count=[132048, 1])

        # Load field data
        dpot = s.read('dpot', start=[0, 0], count=[132048, 1])

        # Time
        t = s.read('time')
        

    

    # Set up colormaps
    tw_repeated_16 = mpl.cm.twilight(np.mod(np.linspace(0, 16, 256),1))
    twr_cmap_16 = mpl.colors.LinearSegmentedColormap.from_list('twilight_repeated', tw_repeated_16, N=256)

    tw_repeated_8 = mpl.cm.twilight(np.mod(np.linspace(0, 6, 256),1))
    twr_cmap_8 = mpl.colors.LinearSegmentedColormap.from_list('twilight_repeated', tw_repeated_8, N=256)

    # Set up the figure and axes
    fig, axs = plt.subplots(1, 4, figsize=(19.2,10.8), sharex=True, sharey=True)
    plt.tight_layout(pad=0)
    plt.suptitle(f't = {t:.4f} s')

    # Plot the data
    plot_2d(axs[0], dpot[:,0], R'$\delta\phi$ [V]', 'PiYG', vmin=-20, vmax=20)
    plot_2d(axs[1], e_den[:,0]*1e-19, R'$n_e$ [$10^{19}$ m$^{-3}$]', twr_cmap_16, vmin=0, vmax=5)
    plot_2d(axs[2], i_T_perp[:,0]*1e-3, R'$T_{\perp,i}$ [keV]', twr_cmap_8, vmin=0, vmax=1.2)
    plot_2d(axs[3], e_T_perp[:,0]*1e-3, R'$T_{\perp,e}$ [keV]', twr_cmap_8, vmin=-0, vmax=1.2)

    plt.savefig(outfile, dpi=100)

    plt.close(fig)

    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time)*1e3

    print(f'Processed {step:05d} in {elapsed_time:.4f} ms')

    return True


if __name__ == '__main__':
    ## Get list of files to process
    f3d_files = sorted(glob.glob('/global/cfs/cdirs/m3736/XGC1_H/D3D_elec_rgn1_run4/xgc.f3d.*.bp'))
    print('num files:', len(f3d_files))

    # Get number of processors
    n_procs = mp.cpu_count()
    n_procs = 16
    print('n_procs:', n_procs)

    # Set up the multiprocessing pool
    with mp.Pool(processes=n_procs) as pool:
        # Use the pool to process the files in parallel
        results = pool.map(process_file, f3d_files)

    #process_file(list(f3d_files)[0])