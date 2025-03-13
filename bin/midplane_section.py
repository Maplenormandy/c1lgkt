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

# %% Load the XGC data

eq = Equilibrium.from_eqdfile(R'D:\Documents\IFS\hmode_jet\D3D141451.eqd')

xgcdata = Dataset(R'D:\Documents\Globus\XGC1.nc')

geom_files = {
    'ele_filename': R'D:\Documents\IFS\hmode_jet\Seo.eqd.ele',
    'fdmat_filename': R'D:\Documents\IFS\hmode_jet\fdmat.pkl',
    'min_e_filename': R'D:\Documents\IFS\hmode_jet\min_E_mat.pkl'
}
geom = XgcGeomHandler(eq, xgcdata, **geom_files)

uph = np.load('./outputs/phase_vel.npz')['u_lstsq']

# %% Set up the ballooning interpolator

# Set up zonal interpolation function
tind = 401
zpot = xgcdata['pot00'][tind,:]
zpot_psi = xgcdata['psi00'][:]
interp_zpot = scipy.interpolate.CubicSpline(zpot_psi, zpot, extrapolate=True)
zonalFields = XgcZonalFieldHandler(eq, xgcdata, 401)

# Set up ballooning mode interpolator
fit_results = np.load('./outputs/fit_results.npz', allow_pickle=True)
params_g, params_gh = fit_results['params_g'], fit_results['params_gh']

# Set up the interpolator
mode = GaussHermiteFunction(params_g[:4], params_gh*0.5)
interp_balloon = [(39, mode)]

ballFields = GaussHermiteFieldHandler(geom, interp_zpot, interp_balloon)

# %% Set up initial conditions

filelabel = 'midplane_deut_test_r50'

print('currently running: ' + filelabel)


## Set up the rotating frame
# NOTE: Be careful about unit conventions, XGC is in sec while we work in millisec
tind0 = 401 #424, 386

omega_frame = -uph[tind0,196]*geom.q_surf[196]*1e-3
rotating_frame = particle_motion.RotatingFrameInfo(0, omega_frame, tind0)
t0 = rotating_frame.t0

## Choose which particle properties to use
pp = particle_motion.elec

## Set initial position
#r0 = 2.2259
r0 = eq.interp_router(geom.psi_surf[196]/eq.psix)
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

# Set the initial parallel velocity
vll0 = vll_mean + pp.vt * np.sqrt(0.33) * np.sqrt(0.5)
# Initial magnetic moment
mu0 = pp.m * (np.sqrt(0.67)*pp.vt * np.sqrt(0.5))**2 / 2 / modb


fields = ballFields


# Compute integrals
ham, lphi = particle_tools.compute_integrals_dk(t0, np.concatenate((x0, [vll0, mu0])), eq, pp, fields, rotating_frame)

## Compute a set of initial conditions
nump = 16
#nump = 1
varphi0 = np.linspace(0,2*np.pi/39, num=nump, endpoint=False)

initial_conditions = np.empty(5*nump)

for k in range(nump):
    kll, pll_mean = particle_tools.compute_parallel_energy(t0, r0, z0, varphi0[k], mu0, ham, lphi, eq, pp, fields, rotating_frame)

    if kll < 0:
        print("Warning: k={} has negative kinetic energy. Taking equal to zero".foramat(k))
        kll = 0

    vll = (pll_mean + np.choose(k%2, [1,-1]) * np.sqrt(2 * pp.m * kll)) / pp.m

    initial_conditions[k + 0*nump] = r0
    initial_conditions[k + 1*nump] = varphi0[k]
    initial_conditions[k + 2*nump] = z0
    initial_conditions[k + 3*nump] = vll
    initial_conditions[k + 4*nump] = mu0

    

# %% RK4 solve

output_dir = 'D:/Documents/IFS/hmode_jet/outputs/'

if 'trapped' in filelabel:
    tmult = 50
else:
    tmult = 10

dt_xgc = (xgcdata['t'][tind0+1] - xgcdata['t'][tind0])*1e3
if pp.z > 0:
    t_span = [t0, t0 + dt_xgc*5000*tmult]
else:
    t_span = [t0, t0 + dt_xgc*80*tmult]
nstep = 80000*tmult
ncheckpoint = 800

# Initialize the values
rk4_y = np.empty((5*nump,nstep+1))
rk4_dy = np.empty((5*nump,nstep))
rk4_t = np.linspace(t_span[0], t_span[1], nstep+1)
rk4_y[:,0] = initial_conditions

dt = (t_span[1] - t_span[0]) / nstep

f = particle_motion.f_driftkinetic
#args = (eq, pp, xgcFields)
args = (eq, pp, fields, rotating_frame)

# Load any values
num_saves = 0
for kc in range(num_saves):
    data = np.load(output_dir + 'sections/{}_{:05d}.npz'.format(filelabel, kc))
    rk4_t[kc*ncheckpoint:(kc+1)*ncheckpoint] = data['t']
    rk4_y[:,kc*ncheckpoint:(kc+1)*ncheckpoint] = data['y']
    rk4_dy[:,kc*ncheckpoint:(kc+1)*ncheckpoint] = data['dy']

k0 = max(num_saves*ncheckpoint - 1,0)

## RK4 timestepper
for k in tqdm(range(k0, nstep)):
    y0 = rk4_y[:,k]
    tk = rk4_t[k]

    k1 = f(tk, y0, *args)
    k2 = f(tk+dt/2, y0+k1*(dt/2), *args)
    k3 = f(tk+dt/2, y0+k2*(dt/2), *args)
    k4 = f(tk+dt, y0+k3*dt, *args)

    y1 = y0 + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    rk4_dy[:,k] = (k1 + 2*k2 + 2*k3 + k4) / 6
    rk4_y[:,k+1] = y1
    
    if k > 0 and k % ncheckpoint == 0:
        np.savez(output_dir + 'sections/{}_{:05d}.npz'.format(filelabel, k//ncheckpoint - 1),
                 t=rk4_t[k-ncheckpoint:k],
                 y=rk4_y[:,k-ncheckpoint:k],
                 dy=rk4_dy[:,k-ncheckpoint:k])
