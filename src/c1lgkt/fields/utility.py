"""
A series of utility functions
"""

import numpy as np
import scipy.interpolate

def periodify(z, period=2*np.pi):
    """
    Takes a length m array and returns a length m+1 array such that r[0] == r[-1] mod period.
    Useful for the scipy interpolation functions, which require this format in order to do
    periodic interpolation.

    If period <= 0, then don't unwrap
    """
    if period > 0:
        return np.unwrap(np.concatenate((z, [z[0]])), period=period)
    else:
        return np.concatenate((z, [z[0]]))

def periodic_cubic_spline(theta, f) -> scipy.interpolate.CubicSpline:
    """
    A helper function for constructing periodic cubic splines.
    """
    return scipy.interpolate.CubicSpline(periodify(theta), periodify(f, period=0), bc_type='periodic')

def md(z, period=2*np.pi):
    """
    Maps z to [-period/2,period/2)
    """
    return np.mod(z + period/2, period) - period/2

def refine_max_position(t, f):
    """
    Finds the position of the absolute maximum of f and refines its location using a quadratic fit.

    Parameters:
    - f: 1D numpy array of function values.
    - t: 1D numpy array of corresponding positions (same length as f).

    Returns:
    - t_max_refined: Refined position of the maximum.
    - f_max_refined: Refined value of the maximum.
    """
    # Find the index of the absolute maximum
    max_idx = np.argmax(f)

    # Check if neighbors exist for quadratic fitting
    if max_idx == 0 or max_idx == len(f) - 1:
        # Cannot refine if the maximum is at the boundary
        return t[max_idx], f[max_idx]
    
    # Extract points for quadratic fitting
    f1, f2, f3 = f[max_idx - 1], f[max_idx], f[max_idx + 1]
    t1, t2, t3 = t[max_idx - 1], t[max_idx], t[max_idx + 1]
    
    # Fit a quadratic: f(t) = a * t^2 + b * t + c
    # Solve for a, b, c using the three points
    A = np.array([
        [t1**2, t1, 1],
        [t2**2, t2, 1],
        [t3**2, t3, 1]
    ])
    b = np.array([f1, f2, f3])
    a, b, c = np.linalg.solve(A, b)

    # Refined maximum position (vertex of the parabola)
    t_max_refined = -b / (2 * a)

    # Refined maximum value
    f_max_refined = a * t_max_refined**2 + b * t_max_refined + c

    return t_max_refined, f_max_refined