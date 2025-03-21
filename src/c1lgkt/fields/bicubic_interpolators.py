# -*- coding: utf-8 -*-
"""
@author: maple

This file includes functions for performing interpolation on equispaced regular
grids that also returns derivatives.


NOTE: These interpolators seem to be fastest for an intermediate number of points???
"""

import numpy as np
from numpy.typing import ArrayLike
import scipy.linalg

from typing import List, Literal, Tuple

# %%

class BicubicInterpolator:
    """
    A class that performs bicubic interpolation on a regular (equispaced) grid, possibly including
    periodic boundary conditions.

    The basic strategy is to perform cubic Hermite interpolation in two dimensions,
    which requires knowing f, f_x, f_y, and f_xy at each grid point. This is where
    the boundary conditions come in.
    """

    def __init__(self, f: np.ndarray, lims: Tuple[ArrayLike], bc_type: List[Literal['natural', 'periodic']] = ['natural', 'natural']):
        """
        Initializes a cubic interpolator for f.

        TODO: Deal with complex numbers?

        Parameters
        ----------
        f : numpy array of shape (nx, ny)
            The data to interpolate. Notice that x is the first index and y is the second.
        lims : tuple of array_like
            The limits of the grid, ([xmin, xmax], [ymin, ymax])
        bc_type : list of str
            The boundary conditions to use. 'natural' means the second derivative is 0
            at the edges, while 'periodic' means the function is periodic. Note that unlike
            scipy's conventions, for periodic BCs f does not require the last point to be repeated
        """

        # Store the limits
        self.xlim, self.ylim = lims

        # Store the boundary conditions
        self.bc_type = bc_type
        self.bc_periodic = [b == 'periodic' for b in bc_type]

        ## Compute the derivatives here. Note that derivatives are with respect to
        # pixels, so dx = dy = 1.

        # Compute y derivatives first
        if bc_type[1] == 'natural':
            ybands = np.ones((2, f.shape[1]))
            ybands[0,:] = 4
            ybands[0,0] = 2
            ybands[0,-1] = 2

            y_lhs = np.empty(f.shape)
            y_lhs[:,1:-1] = 3*(f[:,2:] - f[:,:-2])
            y_lhs[:,0] = 3*(f[:,1] - f[:,0])
            y_lhs[:,-1] = 3*(f[:,-1] - f[:,-2])

            fy = scipy.linalg.solveh_banded(ybands, y_lhs.T, lower=True).T
        elif bc_type[1] == 'periodic':
            ky = np.fft.rfftfreq(f.shape[1], d=1.0)*2*np.pi
            f_ffty = np.fft.rfft(f, axis=1)

            fy = np.fft.irfft(f_ffty * (1j*ky)[np.newaxis,:], axis=1)
            

        # Next, compute x derivatives
        if bc_type[0] == 'natural':
            # Diagonals for the banded matrices used to compute the derivatives, following
            # the steps in https://mathworld.wolfram.com/CubicSpline.html
            xbands = np.ones((2, f.shape[0]))
            xbands[0,:] = 4
            xbands[0,0] = 2
            xbands[0,-1] = 2

            x_lhs = np.empty(f.shape)
            x_lhs[1:-1,:] = 3*(f[2:,:] - f[:-2,:])
            x_lhs[0,:] = 3*(f[1,:] - f[0,:])
            x_lhs[-1,:] = 3*(f[-1,:] - f[-2,:])

            fx = scipy.linalg.solveh_banded(xbands, x_lhs, lower=True)

            yx_lhs = np.empty(f.shape)
            yx_lhs[1:-1,:] = 3*(fy[2:,:] - fy[:-2,:])
            yx_lhs[0,:] = 3*(fy[1,:] - fy[0,:])
            yx_lhs[-1,:] = 3*(fy[-1,:] - fy[-2,:])

            fxy = scipy.linalg.solveh_banded(xbands, yx_lhs, lower=True)
        elif bc_type[0] == 'periodic':
            kx = np.fft.rfftfreq(f.shape[0], d=1.0)*2*np.pi

            f_fftx = np.fft.rfft(f, axis=0)
            fy_fftx = np.fft.rfft(fy, axis=0)

            fx = np.fft.irfft(f_fftx * (1j*kx)[:,np.newaxis], axis=0)
            fxy = np.fft.irfft(fy_fftx * (1j*kx)[:,np.newaxis], axis=0)

        # Store the derivatives
        self.f_derivs = (f, fx, fy, fxy)

        # Compute dx, dy
        if self.bc_periodic[0]:
            self.dx = (self.xlim[1] - self.xlim[0]) / f.shape[0]
        else:
            self.dx = (self.xlim[1] - self.xlim[0]) / (f.shape[0]-1)

        if self.bc_periodic[1]:
            self.dy = (self.ylim[1] - self.ylim[0]) / f.shape[1]
        else:
            self.dy = (self.ylim[1] - self.ylim[0]) / (f.shape[1]-1)

    def __call__(self, x: np.ndarray, y: np.ndarray, nu=0) -> np.ndarray | tuple[np.ndarray]:
        """
        This function computes f and its derivatives at the requested points xp using Bicubic Hermite interpolation.

        Parameters
        ----------
        x, y : numpy array
            Input points
        nu : int or tuple of ints
            The requested order of derivatives. For example, nu=0 returns just f, while
            nu=(1,2) returns a tuple of the first and second derivatives.

        Returns
        -------
        numpy array of shape (*,...) or a tuple of numpy arrays of shape (*,...)
            Returns either a single array or tuple shaped like nu. The shape of the
            array matches the shape of x and y, where * is either none for nu=0, (2,) for
            nu=1, and (2,2) for nu=2.
        """

        ## Unpack requisite info
        f, fx, fy, fxy = self.f_derivs
        xlim, ylim = self.xlim, self.ylim
        
        ## Flatten then normalize the evaluation points
        norm_xp = np.empty((2, np.size(x)))
        norm_xp[0,:] = (np.ravel(x) - xlim[0]) / (xlim[1]-xlim[0]) * f.shape[0]
        norm_xp[1,:] = (np.ravel(y) - ylim[0]) / (ylim[1]-ylim[0]) * f.shape[1]
        
        # Note: periodic boundary will be different
        dx, dy = self.dx, self.dy
        
        # Split into index and fractional part.
        # TODO: Right now we're defined on the half-closed interval [0,1) for non-periodic conditions but we could do [0,1] with some casework
        indp = np.floor(norm_xp).astype(int)
        fracp = norm_xp-indp
        cfracp = 1-fracp
        
        # The first and second index.
        ind0 = indp
        ind1 = indp+1

        if self.bc_periodic[0]:
            ind0[0,:] = ind0[0,:] % f.shape[0]
            ind1[0,:] = ind1[0,:] % f.shape[0]
        if self.bc_periodic[1]:
            ind0[1,:] = ind0[1,:] % f.shape[1]
            ind1[1,:] = ind1[1,:] % f.shape[1]

        
        ## Set up control points. Note Cubic Hermite interpolation can be expressed
        # in terms of a Bezier curve with control points:
        #     f(0), p(0) + p'(0)/3, p(1) - p'(1)/3, p(1)
        # see https://en.wikipedia.org/wiki/Cubic_Hermite_spline
        cpts = np.zeros((4,4,norm_xp.shape[1]))
        
        cpts[:2,:2,:] = f[ind0[0,:], ind0[1,:]][np.newaxis,np.newaxis,:]
        cpts[1,:2,:] += fx[ind0[0,:], ind0[1,:]][np.newaxis,:]/3
        cpts[:2,1,:] += fy[ind0[0,:], ind0[1,:]][np.newaxis,:]/3
        cpts[1,1,:] += fxy[ind0[0,:], ind0[1,:]]/9
        
        cpts[:2,2:,:] = f[ind0[0,:], ind1[1,:]][np.newaxis,np.newaxis,:]
        cpts[1,2:,:] += fx[ind0[0,:], ind1[1,:]][np.newaxis,:]/3
        cpts[:2,2,:] -= fy[ind0[0,:], ind1[1,:]][np.newaxis,:]/3
        cpts[1,2,:] -= fxy[ind0[0,:], ind1[1,:]]/9
        
        cpts[2:,:2,:] = f[ind1[0,:], ind0[1,:]][np.newaxis,np.newaxis,:]
        cpts[2,:2,:] -= fx[ind1[0,:], ind0[1,:]][np.newaxis,:]/3
        cpts[2:,1,:] += fy[ind1[0,:], ind0[1,:]][np.newaxis,:]/3
        cpts[2,1,:] -= fxy[ind1[0,:], ind0[1,:]]/9
        
        cpts[2:,2:,:] = f[ind1[0,:], ind1[1,:]][np.newaxis,np.newaxis,:]
        cpts[2,2:,:] -= fx[ind1[0,:], ind1[1,:]][np.newaxis,:]/3
        cpts[2:,2,:] -= fy[ind1[0,:], ind1[1,:]][np.newaxis,:]/3
        cpts[2,2,:] += fxy[ind1[0,:], ind1[1,:]]/9
        
        # Profiling code
        #print('assemble cpts:', time.perf_counter_ns() - startTime)
        #startTime = time.perf_counter_ns()
        
        ## Prepare arrays for storing results. eval_array[nu] is the nu-th derivative.
        eval_array = [None, None, None]
        
        if isinstance(nu, tuple):
            nu_tuple = nu
        else:
            nu_tuple = (nu,)
        
        ### Perform de Casteljau's algorithm. Note we can compute derivatives:
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
        
        ## Perform the first step of the reduction
        j = 1
        for k in range(4-j):
            cpts[k,:,:] = cfracp[0,:][np.newaxis,:]*cpts[k,:,:] + fracp[0,:][np.newaxis,:]*cpts[k+1,:,:]
            cpts[:,k,:] = cfracp[1,:][np.newaxis,:]*cpts[:,k,:] + fracp[1,:][np.newaxis,:]*cpts[:,k+1,:]

        if 2 in nu_tuple:
            # If the second derivative is requested, we start computing its info
            eval_array[2] = np.empty((2,2, norm_xp.shape[1]))
            
            # Compute helper points for the second derivative here
            cxxpts = cpts[2,:3,:] - 2*cpts[1,:3,:] + cpts[0,:3,:]
            cyypts = cpts[:3,2,:] - 2*cpts[:3,1,:] + cpts[:3,0,:]
            
            # Perform the next step of reduction on the second derivative points
            j = 2
            for k in range(4-j):
                cxxpts[k,:] = cfracp[0,:]*cxxpts[k,:] + fracp[0,:]*cxxpts[k+1,:]
                cyypts[k,:] = cfracp[1,:]*cyypts[k,:] + fracp[1,:]*cyypts[k+1,:]
            
            # Here, we fill in f_xx and f_yy by computing the final step of reduction
            eval_array[2][0,0,:] = (6/(dx**2)) * (cfracp[0,:]*cxxpts[0,:] + fracp[0,:]*cxxpts[1,:])
            eval_array[2][1,1,:] = (6/(dy**2)) * (cfracp[1,:]*cyypts[0,:] + fracp[1,:]*cyypts[1,:])
        
        ## Now we compute the next step of reduction on the original control points
        j = 2
        for k in range(4-j):
            cpts[k,:,:] = cfracp[0,:][np.newaxis,:]*cpts[k,:,:] + fracp[0,:][np.newaxis,:]*cpts[k+1,:,:]
            cpts[:,k,:] = cfracp[1,:][np.newaxis,:]*cpts[:,k,:] + fracp[1,:][np.newaxis,:]*cpts[:,k+1,:]

        if 1 in nu_tuple:
            # If the first derivative is requested, we compute its info
            eval_array[1] = np.empty((2, norm_xp.shape[1]))
            
            # Prepare derivative helpers
            cxpts = cpts[1,:2,:] - cpts[0,:2,:]
            cypts = cpts[:2,1,:] - cpts[:2,0,:]
        
            # Here, we are computing [f_x, f_y]
            eval_array[1][0,:] = (3/dx) * (cfracp[0,:]*cxpts[0,:] + fracp[0,:]*cxpts[1,:])
            eval_array[1][1,:] = (3/dy) * (cfracp[1,:]*cypts[0,:] + fracp[1,:]*cypts[1,:])
            
            # Finalize the array by reshaping it
            eval_array[1] = eval_array[1].reshape((2,) + x.shape)
        
        if 2 in nu_tuple:
            # Here, we compute the mixed derivative
            eval_array[2][0,1,:] = (9/(dx*dy)) * (cypts[1,:] - cypts[0,:])
            eval_array[2][1,0,:] = eval_array[2][0,1,:]
            
            eval_array[2] = eval_array[2].reshape((2,2) + x.shape)
            
        ## If the original function is requested, we perform the last step of reduction
        if 0 in nu_tuple:
            j = 3
            for k in range(4-j):
                cpts[k,:,:] = cfracp[0,:][np.newaxis,:]*cpts[k,:,:] + fracp[0,:][np.newaxis,:]*cpts[k+1,:,:]
                cpts[:,k,:] = cfracp[1,:][np.newaxis,:]*cpts[:,k,:] + fracp[1,:][np.newaxis,:]*cpts[:,k+1,:]
        
            eval_array[0] = cpts[0,0,:].reshape(x.shape)
        
        # Profiling code
        #print('compute algorithm:', time.perf_counter_ns() - startTime)
        #startTime = time.perf_counter_ns()
        
        if isinstance(nu, tuple):
            return tuple(eval_array[n] for n in nu)
        else:
            return eval_array[nu]
