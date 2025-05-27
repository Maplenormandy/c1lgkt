# %% -*- coding: utf-8 -*-
"""
@author: maple

Class for magnetic and profile equilibria. Based off of equilibrium.m provided by Hongxuan Zhu
"""

import numpy as np
import scipy.interpolate

# Below is a hack to load eqtools; 
#import sys
#sys.path.append(R'C:\Users\maple\OneDrive\Research\IFS\Scripts\libs\eqtools')
#from eqtools import eqdskreader, pfilereader # type: ignore

import matplotlib.pyplot as plt
import matplotlib as mpl

from typing import Type, TypeVar

# %%

# List of variables expected in the magnetic geometry. Used to generate some code
eqd_vars = 'mr mz mpsi rmin rmax zmin zmax rgrid zgrid raxis zaxis psix rx zx psi ff psirz wallrz lcfsrz'.split()
setter_code = '\n'.join('self.' + s + ' = kwargs["' + s + '"]' for s in eqd_vars)

# Some static typing stuff to help with type hinting
T = TypeVar('T', bound='Equilibrium')

class Equilibrium:
    """
    Class which is essentially responsible for holding EFIT and profile-level data
    """
    def __init__(self, **kwargs):
        """
        Instantiates an equilibrium with magnetic data initialized, but not profile data
        """
        # OLD CODE: Use reflection to set these variables out of laziness
        #for key in eqd_vars:
        #    setattr(self, key, kwargs[key])
        # NEW CODE: use the generated setter_code above to get the benefit of code completion
        self.mr = kwargs["mr"]
        self.mz = kwargs["mz"]
        self.mpsi = kwargs["mpsi"]
        self.rmin = kwargs["rmin"]
        self.rmax = kwargs["rmax"]
        self.zmin = kwargs["zmin"]
        self.zmax = kwargs["zmax"]
        self.rgrid = kwargs["rgrid"]
        self.zgrid = kwargs["zgrid"]
        self.raxis = kwargs["raxis"]
        self.zaxis = kwargs["zaxis"]
        self.psix = kwargs["psix"]
        self.rx = kwargs["rx"]
        self.zx = kwargs["zx"]
        self.psi = kwargs["psi"]
        self.ff = kwargs["ff"]
        self.psirz = kwargs["psirz"]
        self.wallrz = kwargs["wallrz"]
        self.lcfsrz = kwargs["lcfsrz"]
        
        ## Set up interpolator functions
        #self.psirz_derivs_ = fast_interpolators.compute_bicubic_spline_derivs(self.psirz.T)
        self.interp_ff = scipy.interpolate.CubicSpline(self.psi, self.ff, bc_type='clamped', extrapolate=True)
        
        # Functions for computing Rmid of a given psinorm
        psimid = self.psirz[self.mz//2,:] # Midplane psi
        ind_axis = np.argmin(psimid)
        self.interp_router = scipy.interpolate.CubicSpline(psimid[ind_axis+1:]/self.psix, self.rgrid[ind_axis+1:])
        self.interp_rinner = scipy.interpolate.CubicSpline(np.flip(psimid[:ind_axis])/self.psix, np.flip(self.rgrid[:ind_axis]))
        
        self.interp_psi = scipy.interpolate.RectBivariateSpline(self.rgrid, self.zgrid, self.psirz.T)
    
    #def interp_psi(self, r, z, nu):
        #return fast_interpolators.bicubic_hermite_interpolation(r, z, nu, self.psirz_derivs_, [self.rmin, self.rmax], [self.zmin, self.zmax])
    
    def compute_bv(self, r, z):
        """
        Compute the magnetic field vector
        """
        eq = self
        
        # Evaluate psi and its derivatives
        psi = eq.interp_psi.ev(r, z)
        drpsi = eq.interp_psi.ev(r, z, dx=1)
        dzpsi = eq.interp_psi.ev(r, z, dy=1)
        
        # Detect if a particle is outside the LCFS; if so, use different interpolation
        outside_lcfs = np.logical_or(psi > eq.psix, z < eq.zx)
        ff = np.choose(outside_lcfs, (eq.interp_ff(psi), eq.ff[-1]))
        
        return np.array([-dzpsi / r, -ff / r, drpsi / r])
    
    def compute_psi_and_ff(self, r, z):
        """
        Shorthand function for computing psi, ff, and its derivatives
        """
        # Evaluate psi and its derivatives
        psi = self.interp_psi.ev(r, z)
        psidr = self.interp_psi.ev(r, z, dx=1)
        psidz = self.interp_psi.ev(r, z, dy=1)
        psidrr = self.interp_psi.ev(r,z,dx=2)
        psidrz = self.interp_psi.ev(r,z,dx=1,dy=1)
        psidzz = self.interp_psi.ev(r,z,dy=2)

        # Detect if a particle is outside the LCFS; if so, use different interpolation
        outside_lcfs = np.logical_or(psi > self.psix, z < self.zx)
        ff = np.choose(outside_lcfs, (self.interp_ff(psi), self.ff[-1]))
        dff = np.choose(outside_lcfs, (self.interp_ff(psi, nu=1), 0))

        # Store the evaluations of the interpolations, to pass to other functions that will use it
        psi_ev = (psi, psidr, psidz, psidrr, psidrz, psidzz)
        ff_ev = (ff, dff)

        return psi_ev, ff_ev
    
    def compute_geom_terms(self, r, psi_ev, ff_ev):
        """
        Computes unit vector b, |B|, grad|B|, and curl(b) given psi and ff evaluations
        """
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev
        (ff, dff) = ff_ev
        nump = len(r)

        # B vector
        bv = np.array([-psidz / r, -ff / r, psidr / r])
        # |B|
        modb = np.linalg.norm(bv, axis=0)
        # B unit vector
        bu = bv / modb[np.newaxis,:]
        
        ## Evaluate grad|B| in the following manner:
        # grad|B| = (grad(R|B|) - B grad(R)) / R
        # grad(R|B|) = grad(sqrt(R**2 B**2)) = grad((RB)**2) / 2 / R|B|
        # grad((RB)**2) = grad(F(psi)**2 + |grad(psi)|**2)
        # grad(F(psi)**2) = 2 F'(psi) grad(psi)
        # grad(|grad(psi)|**2) = 2 Hess(psi) grad(psi)
        rmodb = r * modb
        gradpsi = np.array([psidr, np.zeros(nump), psidz])
        gradf2_half = (ff * dff)[np.newaxis,:] * gradpsi
        
        # Note that Hess(psi) is symmetric in Cylindrical coordinates.
        # Due to axisymmetry, we only need to evaluate the Hess(psi) in the R,Z plane
        gradgradpsi2_half = np.array([psidr * psidrr + psidz * psidrz, np.zeros(nump), psidr * psidrz + psidz * psidzz])
        gradrmodb = (gradf2_half + gradgradpsi2_half) / rmodb[np.newaxis,:]
        
        gradmodb = (gradrmodb - np.array([modb, np.zeros(nump), np.zeros(nump)]))/r[np.newaxis,:]

        ## Evalute curl(bhat) = curl(B/|B|) = (|B| curl(B) - grad|B| x B) / B**2
        curlb = np.array([dff * psidz / r, -(psidzz + psidrr) / r - 2*psidr / r**2, -dff * psidr / r])
        curlbu = (curlb - np.cross(gradmodb, bu, axis=0))/modb[np.newaxis,:]

        return bv, bu, modb, gradmodb, curlbu

    @classmethod
    def from_eqdfile(cls: Type[T], filename: str) -> T:
        """
        Loads a *.eqd file and returns an instance of an equilibrium class
        """
        
        with open(filename, 'r') as f:
            data = f.readlines()
            
            # Read and convert the line-by-line data at the top of the file
            mr, mz, mpsi = map(int,data[1].split())
            rmin, rmax, zmin, zmax = map(float,data[2].split())
            raxis, zaxis, baxis = map(float,data[3].split()) # Note: baxis is not used
            psix, rx, zx = map(float, data[4].split())
            
            rgrid = np.linspace(rmin, rmax, mr)
            zgrid = np.linspace(zmin, zmax, mz)

            # Take the lines corresponding to psi, join them into a single array, then parse them
            begin_read = 5
            end_read = begin_read+1+(mpsi-1)//4
            psi = np.array((' '.join(data[begin_read:end_read])).split(), dtype=float)
            # Update the read line, then proceed to the next block of data
            begin_read = end_read
            end_read = begin_read+1+(mpsi-1)//4
            ff = np.array((' '.join(data[begin_read:end_read])).split(), dtype=float) # Toroidal magnetic field?
            
            # Get 2d psi grid as a function of (R,z)
            begin_read = end_read
            end_read = begin_read+1+(mr*mz-1)//4
            psirz = np.array((' '.join(data[begin_read:end_read])).split(), dtype=float).reshape((mz,mr))
            
            # Get the shape of the wall
            begin_read = end_read
            mw = int(data[begin_read+1].strip())
            begin_read = begin_read + 2
            end_read = begin_read + 1 + (2*mw-1)//2
            wallrz = np.array((' '.join(data[begin_read:end_read])).split(), dtype=float).reshape((mw,2))
            
            # Get the shape of the LCFS
            begin_read = end_read
            ml = int(data[begin_read+1].strip())
            begin_read = begin_read + 2
            end_read = begin_read + 1 + (2*ml-1)//2
            lcfsrz = np.array((' '.join(data[begin_read:end_read])).split(), dtype=float).reshape((ml,2))
        
        # Use reflection to set these variables out of laziness
        local_vars = locals()
        eq_kwargs = {key: local_vars[key] for key in eqd_vars}
        return cls(**eq_kwargs)

    # Below is a method to create an equilibrium from a gfile, but it depends on markchil's eqtools:
    # https://github.com/PSFCPlasmaTools/eqtools . TODO: Figure out how to read gfiles to remove this dependency
    '''
    @classmethod
    def from_gfile(cls: Type[T], filename: str) -> T:
        """
        Loads an EFIT gfile and converts it to the same format as the eqd file loader
        """
        
        eq = eqdskreader.EqdskReader(gfile=filename, verbose=False)
        
        rgrid = eq.getRGrid()
        zgrid = eq.getZGrid()
        
        mr, mz, mpsi = len(rgrid), len(zgrid), len(eq.getF()[0,:])
        rmin, rmax, zmin, zmax = np.min(rgrid), np.max(rgrid), np.min(zgrid), np.max(zgrid)
        raxis, zaxis, = eq.getMagR()[0], eq.getMagZ()[0]
        
        # Note: Something is iffy about coordinate conventions, so multiply flux and F by -1
        # Shift psi so that psi on axis is the minimum
        psirz = eq.getFluxGrid()[0,:,:] - eq.getFluxAxis() * -1
    
        lcfsrz = np.array([eq.getRLCFS()[0,:], eq.getZLCFS()[0,:]]).T
        
        ## To compute rx and zx, look for the sharpest turn in the LCFS contour
        dlcfs = np.diff(lcfsrz, axis=0)
        # Unit vector pointing from one point in the LCFS contour to the next
        ulcfs = dlcfs / np.linalg.norm(dlcfs, axis=1)[:,np.newaxis]
        # Compute the dot product (i.e. cosine of angle) between unit vectors
        cangles = np.sum(ulcfs[1:,:] * ulcfs[:-1,:], axis=1)
        # Pick out the sharpest turn
        indx = np.argmin(np.abs(cangles))+1
        
        rx, zx = lcfsrz[indx,0], lcfsrz[indx,1]
        
        # Note: getFluxLCFS gives the wrong value, so recompute it
        psif = scipy.interpolate.RectBivariateSpline(rgrid, zgrid, psirz.T)
        psix = psif.ev(rx, zx)
        
        psi = np.linspace(0, 1, mpsi) * psix
        ff = eq.getF()[0,:] * -1
        
        wallrz = np.array(eq.getMachineCrossSectionFull()).T
        
        # Use reflection to set these variables out of laziness
        local_vars = locals()
        eq_kwargs = {key: local_vars[key] for key in eqd_vars}
        return cls(**eq_kwargs)
    '''
    
    def plot_magnetic_geometry(self, ax, monochrome=True, alpha=1.0):
        eq = self
        
        if monochrome:
            ax.contour(eq.rgrid, eq.zgrid, eq.psirz, levels=64, colors=['tab:gray'], linewidths=mpl.rcParams['lines.linewidth']*0.5, alpha=alpha)
            ax.plot(eq.wallrz[:,0], eq.wallrz[:,1], c='k')
            ax.plot(eq.lcfsrz[:,0], eq.lcfsrz[:,1], c='k')
            ax.set_aspect('equal')
        else:
            ax.contour(eq.rgrid, eq.zgrid, eq.psirz, levels=64, linewidths=mpl.rcParams['lines.linewidth']*0.5, alpha=alpha)
            ax.plot(eq.wallrz[:,0], eq.wallrz[:,1])
            ax.plot(eq.lcfsrz[:,0], eq.lcfsrz[:,1])
            ax.set_aspect('equal')
