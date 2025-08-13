# %% -*- coding: utf-8 -*-
"""
@author: maple

Code to compute some Poincare sections
"""

import numpy as np

import scipy.integrate
import scipy.interpolate
import scipy.sparse

from tqdm import tqdm

from netCDF4 import Dataset

from c1lgkt.fields.equilibrium import Equilibrium
from c1lgkt.fields.field_handlers import GaussHermiteFieldHandler, XgcZonalFieldHandler, GaussHermiteFunction
from c1lgkt.fields.geometry_handlers import XgcGeomHandler

import c1lgkt.particles.particle_motion as particle_motion
import c1lgkt.particles.particle_tools as particle_tools

import os

# %% Load the XGC data

drive_letter = 'D:'

eq = Equilibrium.from_eqdfile(drive_letter + R'\Documents\IFS\hmode_jet\D3D141451.eqd')

xgcdata = Dataset(drive_letter + R'\Documents\Globus\XGC1.nc')

geom_files = {
    'ele_filename': drive_letter + R'\Documents\IFS\hmode_jet\Seo.eqd.ele',
    'fdmat_filename': drive_letter + R'\Documents\IFS\hmode_jet\fdmat.pkl',
    'min_e_filename': drive_letter + R'\Documents\IFS\hmode_jet\min_E_mat.pkl'
}
geom = XgcGeomHandler(eq, xgcdata, **geom_files)

uph = np.load('./outputs/phase_vel.npz')['u_lstsq']

# %% Set up the ballooning interpolator

# Set up zonal interpolation function
tind = 400
zpot = xgcdata['pot00'][tind,:]
zpot_psi = xgcdata['psi00'][:]
interp_zpot = scipy.interpolate.CubicSpline(zpot_psi, zpot, extrapolate=True)
#zonalFields = XgcZonalFieldHandler(eq, xgcdata, 401)

# Set up ballooning mode interpolator
fit_results = np.load('./outputs/fit_results.npz', allow_pickle=True)
params_g, params_gh = fit_results['params_g'], fit_results['params_gh']


# %% Set up initial conditions

filelabel = 'psinn02_r100'
dpsin = 0.02

print('currently running: ' + filelabel)

# Set up the interpolator
mode = GaussHermiteFunction(params_g[:4], params_gh)
interp_balloon = [(39, mode)]

ballFields = GaussHermiteFieldHandler(geom, interp_zpot, interp_balloon)

## Set up the rotating frame
# NOTE: Be careful about unit conventions, XGC is in sec while we work in millisec
tind0 = tind

ksurf0 = np.argmin(-uph[tind0,180:220]*geom.q_surf[180:220])+180
print('ksurf0 = {}'.format(ksurf0))

omega_frame = -uph[tind0,ksurf0]*geom.q_surf[ksurf0]*1e-3
rotating_frame = particle_motion.RotatingFrameInfo(0, omega_frame, tind0)
t0 = rotating_frame.t0

## Choose which particle properties to use
pp = particle_motion.deut

## Set initial position
#r0 = 2.2259
r0 = eq.interp_router(geom.psi_surf[ksurf0]/eq.psix)
z0 = geom.zaxis
x0 = np.array([r0, 0.0, z0])

# Compute magnetic field at initial position
bv = eq.compute_bv(x0[0], x0[2])
modb = np.linalg.norm(bv)
bu = bv / modb

## Compute the rotation frequency and the mean parallel velocity
psi0 = eq.interp_psi.ev(x0[0], x0[2])
omega0 = -ballFields.interp_zpot(psi0, nu=1)*ballFields.scale_conversion()
vll_mean = eq.interp_ff(psi0) * omega0 / modb

## Prepare the initial conditions

# Number of particles in each dimension
nmu = 32
nvll = 32
nphi = 192
nump = nvll*nphi

#lagpts, lagweights = np.polynomial.laguerre.laggauss(nmu)
#hermpts, hermweights = np.polynomial.hermite_e.hermegauss(nvll)

# Reference particle energy
ev0 = 0.78
# Set magnetic moment based on vperp grid staggered relative to the XGC grid
vperp_grid = pp.vt * np.sqrt(ev0) * np.linspace(0, 4, nmu+1)
vperp_grid = (vperp_grid[1:] + vperp_grid[:-1])/2.0
mu_grid = pp.m * vperp_grid**2 / (2 * modb)

# Initial parallel velocity. Use a staggered grid relative to the XGC grid
vll_grid = vll_mean + pp.vt * np.sqrt(ev0) * np.linspace(-4, 4, nvll+1)
vll_grid = (vll_grid[1:] + vll_grid[:-1])/2.0
vll_start = np.repeat(vll_grid, nphi)

# Initial radius
r_start = np.ones(nump)*eq.interp_router(geom.psi_surf[ksurf0]/eq.psix - dpsin)

# Initial toroidal angle; use a low-discrepancy sequence
alpha = 2.0 / (np.sqrt(5) + 1)
varphi_grid = np.mod((np.arange(nphi)+1) * alpha, 1) * (2 * np.pi / 39)
varphi_start = np.tile(varphi_grid, nvll)


# %% Functions for detecting when a particle has left the domain

"""
The basic strategy is to run RK4 for some number steps, at which point I compute the angular momentum
to see if a particle has left the domain. If so, we find the timestep at which it left, then stop following
the particle. We compact the remaining particles, keeping track of the original index of each particle
"""

def compute_lphi(ysol):
    """
    Computes the angular momentum of ysol, optimized and adapted for this situation from
    the version in particle_tools
    """
    num_sol = ysol.shape[0] // 5

    if len(ysol.shape) > 1:
        r = ysol[0*num_sol:1*num_sol,:].flatten()
        z = ysol[2*num_sol:3*num_sol,:].flatten()
        vll = ysol[3*num_sol:4*num_sol,:].flatten()
    else:
        r = ysol[0*num_sol:1*num_sol]
        z = ysol[2*num_sol:3*num_sol]
        vll = ysol[3*num_sol:4*num_sol]

    # Magnetic stuff
    psi = eq.interp_psi.ev(r, z)
    bv = eq.compute_bv(r, z)
    modb = np.linalg.norm(bv, axis=0)

    lphi = pp.z * psi + pp.m * vll * r * bv[1,:] / modb

    if len(ysol.shape) > 1:
        return lphi.reshape((num_sol, -1))
    else:
        return lphi

def compute_escape_time(t, ysol, lphi0):
    # Note: num_sol might have changed relative to nump!
    num_sol = ysol.shape[0] // 5

    # Compute the normalized difference in the canonical angular momentum
    lphi = compute_lphi(ysol)
    dlphin = (lphi - lphi0[:,np.newaxis]) / eq.psix

    t_escape = np.empty(num_sol)

    for k in range(num_sol):
        # Escape condition is that the normalized change in angular momentum exceeds 2 times the original displacement
        ind_escape = np.nonzero(dlphin[k,:]/dpsin > 2)[0]

        # If we've escaped, store the escape time; else, return infinity
        if len(ind_escape) > 0:
            t_escape[k] = t[ind_escape[0]]
        else:
            t_escape[k] = np.inf

    return t_escape


# %% RK4 solve

output_dir = drive_letter + '/Documents/IFS/hmode_jet/outputs/'

# Check if the output directory exists, otherwise create it
if not os.path.exists(output_dir + 'transmissivity/{}'.format(filelabel)):
    os.makedirs(output_dir + 'transmissivity/{}'.format(filelabel))

# Set time window
n_dt_xgc = 1000

dt_xgc = (xgcdata['t'][tind0+1] - xgcdata['t'][tind0])*1e3
if pp.z > 0:
    t_span = [t0, t0 + dt_xgc*n_dt_xgc]
    nstep = n_dt_xgc  * 16
else:
    t_span = [t0, t0 + dt_xgc*n_dt_xgc]
    nstep = n_dt_xgc  * 500

# Number of steps before doing a checkpoint
ncheckpoint = 80

for kmu in [nmu-1]:
    # Initialize the values
    rk4_y_check = np.empty((5*nump,ncheckpoint))
    rk4_dy_check = np.empty((5*nump,ncheckpoint))
    rk4_t_check = np.empty(ncheckpoint)

    # Get the magnetic moment
    mu0 = mu_grid[kmu]

    # Set up the gyroaveraging and assign the fields
    ballFields.set_j_params(mu0, pp.m, pp.z)
    fields = ballFields

    # Assign initial conditions

    initial_conditions = np.empty(5*nump)

    for k in range(nump):
        initial_conditions[k + 0*nump] = r_start[k]
        initial_conditions[k + 1*nump] = varphi_start[k]
        initial_conditions[k + 2*nump] = z0
        initial_conditions[k + 3*nump] = vll_start[k]
        initial_conditions[k + 4*nump] = mu0


    rk4_y = initial_conditions

    dt = (t_span[1] - t_span[0]) / nstep

    f = particle_motion.f_driftkinetic
    args = (eq, pp, fields, rotating_frame)

    ## Arrays to store the intial angular momenta as well as escape times
    lphi0 = compute_lphi(initial_conditions)
    escape_times = np.full(nump, np.inf)

    # original_indices will change size over time as particles are removed
    original_indices = np.arange(nump)


    ## RK4 timestepper
    pbar = tqdm(range(nstep), dynamic_ncols=True)
    num_sol = nump
    pbar.set_description("kmu: {}, num_sol: {}".format(kmu, num_sol))
    for k in pbar:
        y0 = rk4_y
        tk = t_span[0] + dt*k

        k1 = f(tk, y0, *args)
        k2 = f(tk+dt/2, y0+k1*(dt/2), *args)
        k3 = f(tk+dt/2, y0+k2*(dt/2), *args)
        k4 = f(tk+dt, y0+k3*dt, *args)

        rk4_dy = (k1 + 2*k2 + 2*k3 + k4) / 6
        y1 = y0 + dt * rk4_dy

        # Save the values at the checkpoints. Note that the last point can be reconstructed using dt
        rk4_y_check[:,k % ncheckpoint] = y0
        rk4_dy_check[:,k % ncheckpoint] = rk4_dy
        rk4_t_check[k % ncheckpoint] = tk

        rk4_y = y1
        
        # At each checkpoint, rewrite the transmission file with the initial conditions and escape times
        if (k+1) % ncheckpoint == 0:
            # Check for updated escape times
            escape_times_check = compute_escape_time(rk4_t_check, rk4_y_check, lphi0[original_indices])
            escape_times[original_indices] = escape_times_check

            np.savez(output_dir + 'transmissivity/{}/{}_mu{:02d}.npz'.format(filelabel, filelabel, kmu),
                    initial_conditions=initial_conditions,
                    escape_times=escape_times,
                    mu_grid=mu_grid,
                    vll_grid=vll_grid,
                    varphi_grid=varphi_grid,
                    )

            # If any have escaped, compact the remaining solutions
            alive = np.logical_not(np.isfinite(escape_times_check))
            original_indices = original_indices[alive]
            num_sol_old = num_sol
            num_sol = len(original_indices)

            # Re-initialize the checkpoint arrays
            rk4_y_check = np.empty((5*num_sol,ncheckpoint))
            rk4_dy_check = np.empty((5*num_sol,ncheckpoint))
            rk4_t_check = np.empty(ncheckpoint)

            # Update the running solution
            rk4_y = np.empty(5*num_sol)
            for i in range(5):
                rk4_y[i*num_sol:(i+1)*num_sol] = y1[i*num_sol_old:(i+1)*num_sol_old][alive]

            # Update the progressbar with the number of remaining solutions
            pbar.set_description("kmu: {}, num_sol: {}".format(kmu, num_sol))
