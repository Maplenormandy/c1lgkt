# %% -*- coding: utf-8 -*-
"""
@author: maple

Code to compute some Poincare sections. This one starts a blob of initial conditions
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

filelabel = 'midplane_blob_deut_trapped_r100'

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

## Determine the initial values of the integrals

# Particle kinetic energy in keV and cos(pitch angle)
ev0 = 0.78
xi0 = np.sqrt(0.33)
#xi0 = np.sqrt(0.67)
# Set the initial parallel velocity
vll0 = vll_mean + pp.vt * xi0 * np.sqrt(ev0)
# Initial magnetic moment
mu0 = pp.m * (1-xi0**2) * (pp.vt * np.sqrt(ev0))**2 / 2 / modb

# Set up the gyroaveraging and assign the fields
ballFields.set_j_params(mu0, pp.m, pp.z)
fields = ballFields

# Compute integrals
ham, lphi = particle_tools.compute_integrals_dk(t0, np.concatenate((x0, [vll0, mu0])), eq, pp, fields, rotating_frame)

## Compute a set of initial conditions
nump = 192
#nump = 1

# Get a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(42)



#varphi_start = 0.0 + rng.normal(0, 0.0005, size=(nump,))
varphi0 = 0.0
r_start = r0-0.005 + rng.normal(0, 0.002, size=(nump,))
z_start = z0 + rng.normal(0, 0.002, size=(nump,))
#varphi_start = np.linspace(0,2*np.pi/39, num=nump, endpoint=False)
#r_start = np.linspace(r0-0.005, r0+0.005, num=nump)

initial_conditions = np.empty(5*nump)

for k in range(nump):
    kll, pll_mean = particle_tools.compute_parallel_energy(t0, r_start[k], z_start[k], varphi0, mu0, ham, lphi, eq, pp, fields, rotating_frame)

    if kll < 0:
        print("Warning: k={} has negative kinetic energy. Taking equal to zero".format(k))
        kll = 0

    vll = (pll_mean - 1 * np.sqrt(2 * pp.m * kll)) / pp.m

    initial_conditions[k + 0*nump] = r_start[k]
    initial_conditions[k + 1*nump] = varphi0
    initial_conditions[k + 2*nump] = z_start[k]
    initial_conditions[k + 3*nump] = vll
    initial_conditions[k + 4*nump] = mu0

    

# %% RK4 solve

output_dir = drive_letter + '/Documents/IFS/hmode_jet/outputs/'

if 'trapped' in filelabel:
    tmult = 1
else:
    tmult = 1

dt_xgc = (xgcdata['t'][tind0+1] - xgcdata['t'][tind0])*1e3
if pp.z > 0:
    t_span = [t0, t0 + dt_xgc*5000*tmult]
else:
    t_span = [t0, t0 + dt_xgc*80*tmult]
nstep = 80000*tmult
#nstep = 40000
ncheckpoint = 800

# Initialize the values
rk4_y_check = np.empty((5*nump,ncheckpoint))
rk4_dy_check = np.empty((5*nump,ncheckpoint))
rk4_t_check = np.empty(ncheckpoint)

rk4_y = initial_conditions

dt = (t_span[1] - t_span[0]) / nstep

f = particle_motion.f_driftkinetic
#args = (eq, pp, xgcFields)
args = (eq, pp, fields, rotating_frame)

# Check if the output directory exists, otherwise create it
if not os.path.exists(output_dir + 'sections/{}'.format(filelabel)):
    os.makedirs(output_dir + 'sections/{}'.format(filelabel))

## RK4 timestepper
for k in tqdm(range(nstep), dynamic_ncols=True):
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
    
    if (k+1) % ncheckpoint == 0:
        np.savez(output_dir + 'sections/{}/{:05d}.npz'.format(filelabel, k//ncheckpoint),
                 t=rk4_t_check,
                 y=rk4_y_check,
                 dy=rk4_dy_check)
