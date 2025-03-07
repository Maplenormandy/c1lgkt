# -*- coding: utf-8 -*-
"""
@author: maple

This file includes functions for performing interpolation on equispaced regular
grids that also returns derivatives.


NOTE: These interpolators seem to be fastest for an intermediate number of points???
"""

import numpy as np
import scipy.linalg

# %%

def compute_bicubic_spline_derivs(f):
    """
    Computes the derivatives necessary for Bicubic spline interpolation using
    'natural' boundary conditions, i.e. second derivative = 0 at the edges.

    Parameters
    ----------
    f : numpy array of shape (nx, ny)
        Data to interpolate; matches scipy's conventions

    Returns
    -------
    tuple of numpy arrays of shape (nx, ny)
        A tuple (f, f_x, f_y, f_xy) of numpy arrays of shape (nx, ny) that is used
        in the bicubic interpolation. Note the derivatives are computed in pixel
        coordinates, so i.e. dx = dy = 1.
    """
    
    # Diagonals for the banded matrices used to compute the derivatives, following
    # the steps in https://mathworld.wolfram.com/CubicSpline.html
    xbands = np.ones((2, f.shape[0]))
    xbands[0,:] = 4
    xbands[0,0] = 2
    xbands[0,-1] = 2
    
    ybands = np.ones((2, f.shape[1]))
    ybands[0,:] = 4
    ybands[0,0] = 2
    ybands[0,-1] = 2
    
    y_lhs = np.zeros(f.shape)
    y_lhs[:,1:-1] = 3*(f[:,2:] - f[:,:-2])
    y_lhs[:,0] = 3*(f[:,1] - f[:,0])
    y_lhs[:,-1] = 3*(f[:,-1] - f[:,-2])
    
    fy = scipy.linalg.solveh_banded(ybands, y_lhs.T, lower=True).T
    
    x_lhs = np.zeros(f.shape)
    x_lhs[1:-1,:] = 3*(f[2:,:] - f[:-2,:])
    x_lhs[0,:] = 3*(f[1,:] - f[0,:])
    x_lhs[-1,:] = 3*(f[-1,:] - f[-2,:])
    
    fx = scipy.linalg.solveh_banded(xbands, x_lhs, lower=True)
    
    yx_lhs = np.zeros(f.shape)
    yx_lhs[1:-1,:] = 3*(fy[2:,:] - fy[:-2,:])
    yx_lhs[0,:] = 3*(fy[1,:] - fy[0,:])
    yx_lhs[-1,:] = 3*(fy[-1,:] - fy[-2,:])
    
    # Note: using f_x or f_y turns out to be equivalent
    fxy = scipy.linalg.solveh_banded(xbands, yx_lhs, lower=True)
    
    return tuple(map(np.asfortranarray, (f, fx, fy, fxy)))



def bicubic_hermite_interpolation(x, y, nu, f_derivs, xlim, ylim):
    """
    Using a tuple of f derivatives, this function computes f and its derivatives
    at the requested points xp using Bicubic Hermite interpolation.

    Parameters
    ----------
    x, y : numpy array
        Input points
    nu : int or tuple of ints
        The requested order of derivatives. For example, nu=0 returns just f, while
        nu=(1,2) returns a tuple of the first and second derivatives.
    f_derivs : tuple of numpy arrays of shape (nx, ny)
        Should be a tuple of (f, f_x, f_y, f_xy) on the gridpoints. The order of
        the arrays should match scipy's interpolation conventions, i.e. (nx ny)
    xlim, ylim : array_like
        xlim[0], xlim[1] gives xmin and xmax. Similarly, ylim[0], ylim[1] gives ymin and ymax

    Returns
    -------
    numpy array of shape (*,...) or a tuple of numpy arrays of shape (*,...)
        Returns either a single array or tuple shaped like nu. The shape of the
        array matches the shape of x and y, where * is either none for nu=0, (2,) for
        nu=1, and (2,2) for nu=2.
    """
    
    # Profiling code
    #startTime = time.perf_counter_ns()
    
    ## Unpack the derivatives
    f, fx, fy, fxy = f_derivs
    
    ## Flatten then normalize the evaluation points
    norm_xp = np.empty((2, np.size(x)))
    norm_xp[0,:] = (np.ravel(x) - xlim[0]) / (xlim[1]-xlim[0]) * f.shape[0]
    norm_xp[1,:] = (np.ravel(y) - ylim[0]) / (ylim[1]-ylim[0]) * f.shape[1]
    
    # Note: periodic boundary will be different
    dx = (xlim[1]-xlim[0]) / (f.shape[0]-1)
    dy = (ylim[1]-ylim[0]) / (f.shape[1]-1)
    
    # Split into index and fractional part
    indp = norm_xp.astype(int)
    indp[0,:] = np.clip(indp[0,:], 0, f.shape[0]-2)
    indp[1,:] = np.clip(indp[1,:], 0, f.shape[1]-2)
    fracp = norm_xp-indp
    cfracp = 1-fracp
    
    # The first and second index. Potentially could deal with periodicity here
    ind0 = indp
    ind1 = indp+1
    
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
