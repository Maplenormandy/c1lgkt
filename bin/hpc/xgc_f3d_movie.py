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

mpl.use('Agg')

# %% Load data files

eq = Equilibrium.from_eqdfile(R'./outputs/D3D141451.eqd')

xgcdata = Dataset('/global/cfs/cdirs/m3736/Rotation_paper_data/XGC1.nc')

geom_files = {
    'ele_filename': R'/global/cfs/cdirs/m3736/XGC1_H/D3D_elec_rgn1_run4/Seo.eqd.ele',
    'fdmat_filename': R'./outputs/fdmat.pkl',
}
geom = XgcGeomHandler(eq, xgcdata, theta0_mode='midplane', **geom_files)

# %% Get the glob of files that we want to read

ksurf = np.searchsorted(-geom.q_surf, 2.0)
rz_surf = geom.rz_node[geom.breaks_surf[ksurf]:geom.breaks_surf[ksurf+1],:]

def plot_2d(ax, data, label, cmap, vmin=None, vmax=None):
    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Plot the data
    pc = ax.tripcolor(geom.rz_tri, data, shading='gouraud', rasterized=True, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(rz_surf[:,0], rz_surf[:,1], c='k', ls=':')

    # Make the colorbar
    cax = ax.inset_axes([0.57, 0.06, 0.4, 0.01])
    plt.colorbar(pc, cax=cax, orientation='horizontal', label=label)

def process_file(f3d_file):
    """
    Process a single f3d file and extract the required data.
    """

    # Read the f3d file using FileReader
    with FileReader(f3d_file) as s:
        # Load the kinetic data
        e_den = s.read('e_den', start=[0, 0], count=[132048, 1])
        i_T_perp = s.read('i_T_perp', start=[0, 0], count=[132048, 1])
        e_T_perp = s.read('e_T_perp', start=[0, 0], count=[132048, 1])

        # Load field data
        dpot = s.read('dpot', start=[0, 0], count=[132048, 1])

        # Time
        time = s.read('time')
        step = s.read('step')

    # Set up colormaps
    tw_repeated_16 = mpl.cm.twilight(np.mod(np.linspace(0, 16, 256),1))
    twr_cmap_16 = mpl.colors.LinearSegmentedColormap.from_list('twilight_repeated', tw_repeated_16, N=256)

    tw_repeated_8 = mpl.cm.twilight(np.mod(np.linspace(0, 6, 256),1))
    twr_cmap_8 = mpl.colors.LinearSegmentedColormap.from_list('twilight_repeated', tw_repeated_8, N=256)

    # Set up the figure and axes
    fig, axs = plt.subplots(1, 4, figsize=(19.2,10.8), sharex=True, sharey=True)
    plt.tight_layout(pad=0)
    plt.title(f't = {time[0]:.4f} s')

    # Plot the data
    plot_2d(axs[0], dpot[:,0], R'electrostatic potential $\delta\phi$ [V]', 'PiYG', vmin=-20, vmax=20)
    plot_2d(axs[1], e_den[:,0]*1e-19, R'electron density $n_e$ [$10^{19}$ m$^{-3}$]', twr_cmap_16, vmin=0, vmax=5)
    plot_2d(axs[2], i_T_perp[:,0]*1e-3, R'ion temperature $T_{\perp,i}$ [keV]', twr_cmap_8, vmin=0, vmax=1.2)
    plot_2d(axs[3], e_T_perp[:,0]*1e-3, R'electron temperature $T_{\perp,e}$ [keV]', twr_cmap_8, vmin=-0, vmax=1.2)

    plt.savefig(f'./outputs/f3d_movie_{step:05d}.png', dpi=100)

    return True


if __name__ == '__main__':
    ## Get list of files to process
    f3d_files = sorted(glob.glob('/global/cfs/cdirs/m3736/XGC1_H/D3D_elec_rgn1_run4/xgc.f3d.*.bp'))

    # Get number of processors
    n_procs = mp.cpu_count()

    # Set up the multiprocessing pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use the pool to process the files in parallel
        results = pool.map(process_file, f3d_files)