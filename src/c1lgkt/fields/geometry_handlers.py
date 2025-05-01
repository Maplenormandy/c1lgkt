# %% -*- coding: utf-8 -*-
"""
@author: maple

Classes for interpolating stuff on the XGC mesh, mostly matplotlib's cubic
triangle interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import tri

import scipy.interpolate

import pickle
from tqdm import tqdm

import netCDF4

from .equilibrium import Equilibrium

from typing import Literal, NamedTuple

from .utility import periodify, refine_max_position
from .bicubic_interpolators import BicubicInterpolator

import os

# %% Memoized version of minE DOF estimator

def barycentric_weights(x0, y0, x1, y1, x2, y2, x3, y3):
    """
    Barycentric weights used to compute linear interpolation on triangles
    """
    # Precompute factors used in the area calculation
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    
    # Compute the barycentric coordinates directly
    w1 = ((y2 - y3)*(x0 - x3) + (x3 - x2)*(y0 - y3)) / denom
    w2 = ((y3 - y1)*(x0 - x3) + (x1 - x3)*(y0 - y3)) / denom
    w3 = 1 - w1 - w2  # Since w3 = 1 - w1 - w2 for points inside the triangle

    return np.array([w1, w2, w3])

class _DOF_estimator_min_E_memoized(tri._triinterpolate._DOF_estimator):
    """
    Memoized version and sped-up version of the minE DOF estimator used in
    matplotlib.tri.CubicTriInterpolator. The two main optimizations are:
    1) Precomputing the sparse matrices used for the minE optimization, which take a fairly
       long time to assemble
    2) Capping the maximum number of iterations for the CG solver used to do the iterative
       optimization. Note that after a few (i.e. ~3) steps, the solution already looks pretty
       good.
    """

    ## Whether or not we've computed the matrices yet
    matrices_precomputed = False
    # Storage for matrix precomputation
    Kff_coo = None
    Kfc_elem = None
    Ff_indices = None

    # Default tolence and maximum number of iterations for CG
    tol = 1e-8
    maxiter = 3

    def __init__(self, Interpolator, dz):
        self._eccs = Interpolator._eccs

        ## Whether or not we've computed the dofs yet
        self.dof_computed = False
        # Storage for dofs
        self.tri_dof = None

        super().__init__(Interpolator, dz=dz)
        

    @staticmethod
    def compute_Kff_and_Kfc(reference_element, J, ecc, triangles):
        """
        Computes Kff and Kfc. This is a copy-paste of _ReducedHCT_Element.get_Kff_and_Ff
        with the parts depending on Uc removed.
        """
        ntri = np.size(ecc, 0)
        vec_range = np.arange(ntri, dtype=np.int32)
        c_indices = np.full(ntri, -1, dtype=np.int32)  # for unused dofs, -1
        f_dof = [1, 2, 4, 5, 7, 8]
        c_dof = [0, 3, 6]

        # vals, rows and cols indices in global dof numbering
        f_dof_indices = tri._triinterpolate._to_matrix_vectorized([[
            c_indices, triangles[:, 0]*2, triangles[:, 0]*2+1,
            c_indices, triangles[:, 1]*2, triangles[:, 1]*2+1,
            c_indices, triangles[:, 2]*2, triangles[:, 2]*2+1]])

        expand_indices = np.ones([ntri, 9, 1], dtype=np.int32)
        f_row_indices = tri._triinterpolate._transpose_vectorized(expand_indices @ f_dof_indices)
        f_col_indices = expand_indices @ f_dof_indices
        K_elem = reference_element.get_bending_matrices(J, ecc)

        # Extracting sub-matrices
        # Explanation & notations:
        # * Subscript f denotes 'free' degrees of freedom (i.e. dz/dx, dz/dx)
        # * Subscript c denotes 'condensated' (imposed) degrees of freedom
        #    (i.e. z at all nodes)
        # * F = [Ff, Fc] is the force vector
        # * U = [Uf, Uc] is the imposed dof vector
        #        [ Kff Kfc ]
        # * K =  [         ]  is the laplacian stiffness matrix
        #        [ Kcf Kff ]
        # * As F = K x U one gets straightforwardly: Ff = - Kfc x Uc

        # Computing Kff stiffness matrix in sparse coo format
        Kff_vals = np.ravel(K_elem[np.ix_(vec_range, f_dof, f_dof)])
        Kff_rows = np.ravel(f_row_indices[np.ix_(vec_range, f_dof, f_dof)])
        Kff_cols = np.ravel(f_col_indices[np.ix_(vec_range, f_dof, f_dof)])

        # Computing Ff force vector in sparse coo format
        Kfc_elem = K_elem[np.ix_(vec_range, f_dof, c_dof)]
        ###Uc_elem = np.expand_dims(Uc, axis=2)
        ###Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
        Ff_indices = f_dof_indices[np.ix_(vec_range, [0], f_dof)][:, 0, :]

        # Extracting Ff force vector in dense format
        # We have to sum duplicate indices -  using bincount
        ###Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
        return Kff_rows, Kff_cols, Kff_vals, Kfc_elem, Ff_indices

    def compute_dz(self, dz):
        """
        Elliptic solver for bending energy minimization.
        Copy-paste of _DOF_estimator_min_E.compute_dz(), except the stiffness matrix is memoized
        """

        # Initial guess for iterative PCG solver.
        (dzdx, dzdy) = dz
        dzdx = dzdx * self._unit_x
        dzdy = dzdy * self._unit_y
        dz_init = np.vstack([dzdx, dzdy]).T
        Uf0 = np.ravel(dz_init)

        # Check if matrices have been precomputed or not
        cls = self.__class__

        if cls.matrices_precomputed:
            Kff_coo = cls.Kff_coo
            Kfc_elem = cls.Kfc_elem
            Ff_indices = cls.Ff_indices

            Uc = self.z[self._triangles]

            # Compute the Ff vector
            Uc_elem = np.expand_dims(Uc, axis=2)
            Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
            Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))
        else:
            reference_element = tri._triinterpolate._ReducedHCT_Element()
            J = tri.CubicTriInterpolator._get_jacobian(self._tris_pts)
            eccs = self._eccs
            triangles = self._triangles
            Uc = self.z[self._triangles]

            # Building stiffness matrix and most of the force vector in coo format
            Kff_rows, Kff_cols, Kff_vals, Kfc_elem, Ff_indices = cls.compute_Kff_and_Kfc(
                reference_element, J, eccs, triangles)
            
            # Compute the Ff vector
            Uc_elem = np.expand_dims(Uc, axis=2)
            Ff_elem = -(Kfc_elem @ Uc_elem)[:, :, 0]
            Ff = np.bincount(np.ravel(Ff_indices), weights=np.ravel(Ff_elem))

            # Building sparse matrix and solving minimization problem
            # We could use scipy.sparse direct solver; however to avoid this
            # external dependency an implementation of a simple PCG solver with
            # a simple diagonal Jacobi preconditioner is implemented.
            
            n_dof = Ff.shape[0]
            Kff_coo = tri._triinterpolate._Sparse_Matrix_coo(Kff_vals, Kff_rows, Kff_cols,
                                        shape=(n_dof, n_dof))
            Kff_coo.compress_csc()

            # Store the precomputed matrices
            cls.Kff_coo = Kff_coo
            cls.Kfc_elem = Kfc_elem
            cls.Ff_indices = Ff_indices
            cls.matrices_precomputed = True

        Uf, err = tri._triinterpolate._cg(A=Kff_coo, b=Ff, x0=Uf0, tol=cls.tol, maxiter=cls.maxiter)
        # If the PCG did not converge, we return the best guess between Uf0
        # and Uf.
        err0 = np.linalg.norm(Kff_coo.dot(Uf0) - Ff)
        if err0 < err:
            print('convergence warning:', err0)

        # Building dz from Uf
        dz = np.empty([self._pts.shape[0], 2], dtype=np.float64)
        dz[:, 0] = Uf[::2]
        dz[:, 1] = Uf[1::2]
        return dz
    
    def compute_dof_from_df(self):
        """
        Memoize super's version of this function
        """
        if self.dof_computed:
            return self.tri_dof
        else:
            self.tri_dof = super().compute_dof_from_df()
            self.dof_computed = True
            return self.tri_dof
    
    @classmethod
    def save_matrices_to_file(cls, filename):
        with open(filename, 'wb') as f:
            pickle.dump((cls.Kff_coo, cls.Kfc_elem, cls.Ff_indices), f)

    @classmethod
    def load_matrices_from_file(cls, filename):
        with open(filename, 'rb') as f:
            cls.Kff_coo, cls.Kfc_elem, cls.Ff_indices = pickle.load(f)
            cls.matrices_precomputed = True

# %% CubicTriInterpolator that is able to use the memoized DOF estimator above

class CubicTriInterpolatorMemoized(tri.CubicTriInterpolator):
    """
    This is a modification of matplotlib.tri.CubicTriInterpolator which uses memoization
    and is able to use the custom DOF estimator _DOF_estimator_min_E_memoized.
    
    Note that while there a few minor optimizations in this class itself (avoiding some
    duplicate computations and memoizing some computations), the vast majority of the savings
    come from fixing the CG algorithm and avoiding the reassembly of the stiffness matrices
    in the DOF estimator.
    """
    # Some storage for memoized geometric information. These computations don't take very
    # long (~100ms), so it's not worth the hassle of saving/loading these from files
    geometry_precomputed = False
    # tuple of (compressed_triangles, compressed_x, compressed_y, tri_renum, node_renum)
    compressed_triangulation = None
    # tuple of (_unit_x, _unit_y, _pts, _tris_pts, _eccs)
    geometry_info = None

    def __init__(self, triangulation, z, kind='min_E', trifinder=None, dz=None):
        """
        This is a copy-paste of the CubicTriInterpolator, but with certain calls memoized.
        """
        tri.TriInterpolator.__init__(self, triangulation, z, trifinder)

        cls = self.__class__

        if cls.geometry_precomputed:
            (compressed_triangles, compressed_x, compressed_y, tri_renum, node_renum) = cls.compressed_triangulation
            self._triangles = compressed_triangles
            self._tri_renum = tri_renum
            valid_node = (node_renum != -1)
            self._z[node_renum[valid_node]] = self._z[valid_node]

            self._unit_x, self._unit_y, self._pts, self._tris_pts, self._eccs = cls.geometry_info
        else:
            self._triangulation.get_cpp_triangulation()

            tri_analyzer = tri.TriAnalyzer(self._triangulation)
            (compressed_triangles, compressed_x, compressed_y, tri_renum,
            node_renum) = tri_analyzer._get_compressed_triangulation()
            self._triangles = compressed_triangles
            self._tri_renum = tri_renum
            # Taking into account the node renumbering in self._z:
            valid_node = (node_renum != -1)
            self._z[node_renum[valid_node]] = self._z[valid_node]

            # Computing scale factors
            self._unit_x = np.ptp(compressed_x)
            self._unit_y = np.ptp(compressed_y)
            self._pts = np.column_stack([compressed_x / self._unit_x,
                                        compressed_y / self._unit_y])
            # Computing triangle points
            self._tris_pts = self._pts[self._triangles]
            # Computing eccentricities
            self._eccs = self._compute_tri_eccentricities(self._tris_pts)

            cls.compressed_triangulation = (compressed_triangles, compressed_x, compressed_y, tri_renum, node_renum)
            cls.geometry_info = self._unit_x, self._unit_y, self._pts, self._tris_pts, self._eccs
            cls.geometry_precomputed = True

        mpl._api.check_in_list(['user', 'geom', 'min_E'], kind=kind)
        self._dof = self._compute_dof(kind, dz=dz)
        # Loading HCT element
        self._ReferenceElement = tri._triinterpolate._ReducedHCT_Element()

    def _compute_dof(self, kind, dz=None):
        if kind == 'min_E':
            TE = _DOF_estimator_min_E_memoized(self, dz=dz)
            return TE.compute_dof_from_df()
        else:
            return super()._compute_dof(kind, dz)


# %% Geometry handler for XGC

class XgcGeomHandler:
    def __init__(self, eq: Equilibrium, xgcdata: netCDF4.Dataset, ele_filename: str, fdmat_filename: str = '', min_e_filename: str = '', theta0_mode: Literal['midplane', 'max_drive'] = 'midplane'):
        """
        Constructor for XGC data handler. Loads mesh data, triangle data, and sets up the
        finite difference matrices
        """
        self.eq = eq
        self.xgcdata = xgcdata

        ## Geometric and magnetic field info
        self.rz_node = self.xgcdata['rz'][:,:]
        self.psi_node = self.xgcdata['psi_node'][:]
        self.b_node = self.xgcdata['B'][:,:]

        self.nnode = self.psi_node.shape[0]

        ## Load mesh triangulation
        with open(ele_filename, 'r') as f:
            tridata = f.readlines()
            rawdata = list(list(map(int, l.split())) for l in tridata[1:])
            triangles = np.array(rawdata, dtype=int)[:,1:] - 1
        
        rz_tri = tri.Triangulation(self.rz_node[:,0], self.rz_node[:,1], triangles=triangles)
        # Needed for CubicTriInterpolator
        rz_tri.get_cpp_triangulation()
        self.triangles = rz_tri.triangles
        self.rz_tri = rz_tri

        self.ntri = self.triangles.shape[0]

        ## Either load or assemble the finite difference matrices, used to compute gradients
        #  on the mesh nodes

        # Check if the files exist
        if fdmat_filename != '' or not os.path.exists(fdmat_filename):
            with open(fdmat_filename, 'rb') as f:
                self.diff_r, self.diff_z = pickle.load(f)
        else:
            self.assemble_matrices()
            if not os.path.exists(fdmat_filename):
                self.save_matrices_to_file(fdmat_filename)


        ## Load the stiffness matrices needed to compute the minimum bending energy
        # cubic triangular interpolation
        if min_e_filename != '':
            _DOF_estimator_min_E_memoized.load_matrices_from_file(min_e_filename)

        ## Set up the straight field-line coordinates
        self.init_theta(theta0_mode=theta0_mode)

    def init_theta(self, theta0_mode: Literal['midplane', 'max_drive'] = 'midplane'):
        """
        Compute straight field line coordinates and save them to the class. Conventions are
        that theta=0 is along the outboard midplane, and theta is computed in increasing order
        in memory-order of the nodal data. Note that this means theta might not lie between
        [0, 2pi).

        theta0_mode = 'midplane' sets theta0 to be at the outboard midplane
        theta0_mode = 'max_drive' sets theta0 at the maximum of the binormal component of the grad B drift
        """
        ## Compute the node assignement to flux surfaces
        # Number of nodes inside the regular grid
        self.nnode_surf = np.argmax(self.psi_node > 0.331) + 1
        # Breakpoints in the node placement on flux surfaces
        self.breaks_surf = np.argwhere(np.diff(self.psi_node)[:self.nnode_surf] > 1e-4)[:,0] + 1
        # Total number of flux surfaces
        self.nsurf = self.breaks_surf.shape[0]-1

        ## Magnetic data needed to compute straight field line coords
        bp = np.linalg.norm(self.b_node[:,:2], axis=1)
        # Note: b_node is stored in a different coordinate system than I typically use
        f = self.b_node[:,2] / self.rz_node[:,0]
        # Coordinates of the magnetic axis, used to compute geometric theta
        # TODO: This differs from raxis and zaxis in eq. Should figure out something about it
        self.raxis = self.rz_node[0,0]
        self.zaxis = self.rz_node[0,1]

        ## psi and q of each flux surface
        self.psi_surf = np.empty(self.nsurf)
        self.q_surf = np.empty(self.nsurf)

        ## Straight field-line theta stored on the nodes
        self.theta_node = np.zeros(self.nnode)
        # Straight field-line theta minus the geometric theta of the nodes.
        # Useful for interpolation since it is smooth in (R,Z) coordinates.
        self.gdtheta_node = np.zeros(self.nnode)

        ## Holds unwrapped arrays of theta, useful for interpolation
        self.theta_surf = [None] * self.nsurf
        # Geometric theta, also useful for interpolation
        self.gtheta_surf = [None] * self.nsurf
        self.gdtheta_surf = [None] * self.nsurf

        for k in range(self.nsurf):
            # Simplify notation for indexing into the array of nodes
            surfslice = np.index_exp[self.breaks_surf[k]:self.breaks_surf[k+1]]

            self.psi_surf[k] = np.average(self.psi_node[surfslice])
            rz_surf = self.rz_node[self.breaks_surf[k]:self.breaks_surf[k+1],:]

            # Value of the integrand on nodes for computing straight field-line coordinates. See
            # https://xgc.pppl.gov/html/meshing_tutorial.html, section on `relation to flux-coordinates'
            integrand_node = f[surfslice] / bp[surfslice]

            # Value of the integrand on edges times dl
            integrand_edge = np.empty(len(integrand_node))
            integrand_edge[0] = 0.5 * (integrand_node[0] + integrand_node[-1]) * np.linalg.norm(rz_surf[0,:] - rz_surf[-1,:])
            integrand_edge[1:] = 0.5 * (integrand_node[1:] + integrand_node[:-1]) * np.linalg.norm(np.diff(rz_surf, axis=0), axis=1)

            # Compute q and an initial guess for theta; theta=0 corresponds arbtirarily to where the first
            # mesh node is located on the surface. We need to improve that guess
            self.q_surf[k] = np.sum(integrand_edge) / 2 / np.pi
            theta_raw = np.cumsum(integrand_edge) / self.q_surf[k]

            # Get the geometric theta
            gtheta = np.unwrap(np.arctan2(rz_surf[:,1]-self.zaxis, rz_surf[:,0]-self.raxis))

            ## Compute the theta0 offset
            theta_dgeom_raw = theta_raw - gtheta
            if theta0_mode == 'midplane':
                # If we're at the midplane, then theta0 should be at the outboard midplane
                theta0f = scipy.interpolate.CubicSpline(periodify(gtheta), periodify(theta_dgeom_raw), bc_type='periodic')
                theta0 = theta0f(0)
            elif theta0_mode == 'max_drive':
                # If we're in symmetric mode, compute the symmetry point
                r, z = rz_surf[:,0], rz_surf[:,1]
                nump = len(r)

                # Compute magnetic geometry stuff
                psi_ev, ff_ev = self.eq.compute_psi_and_ff(r, z)
                bv, bu, modb, gradmodb, curlbu = self.eq.compute_geom_terms(r, psi_ev, ff_ev)

                # Compute unit vectors in radial and binormal directions
                gradpsi = np.array([psi_ev[1], np.zeros(nump), psi_ev[2]])
                ru = gradpsi / np.linalg.norm(gradpsi, axis=0)
                yu = np.cross(ru, bu, axis=0)

                # Compute B x grad|B| / B^2 and its components in radial and binormal directions
                vd = np.cross(bu, gradmodb, axis=0) / modb
                vd_r = np.sum(vd * ru, axis=0)
                vd_y = np.sum(vd * yu, axis=0)

                # Resample vd onto a regular grid in theta_raw
                theta_samp = np.linspace(0, 2*np.pi, 1024, endpoint=False)
                vd_r_interp = scipy.interpolate.CubicSpline(periodify(theta_raw), periodify(vd_r, period=0), bc_type='periodic')
                vd_y_interp = scipy.interpolate.CubicSpline(periodify(theta_raw), periodify(vd_y, period=0), bc_type='periodic')
                vd_r_samp = vd_r_interp(theta_samp)
                vd_y_samp = vd_y_interp(theta_samp)

                # Compute the symmetry point. The basic idea is that vd_y should be even, and vd_r should be odd.
                # We take the inner product of vd = vd_y + 1j*vd_r with conj(vd(t - theta)), which is simply the
                # convolution of vd with conj(vd). The maximum of this convolution gives the best symmetry point.
                vd_signal = vd_y_samp + 1j*vd_r_samp
                vd_y_conv = np.fft.ifft(np.fft.fft(vd_y_samp) * np.fft.fft(vd_y_samp))
                vd_r_conv = np.fft.ifft(np.fft.fft(vd_r_samp) * np.fft.fft(-vd_r_samp))
                vd_conv = vd_y_conv

                #theta0 = refine_max_position(theta_samp, np.real(vd_conv))[0]
                theta0 = refine_max_position(theta_samp, vd_y_samp)[0]
                theta0 = np.mod(theta0+np.pi, 2*np.pi)-np.pi

            ## Assign the straight field-line thetas to nodes
            self.theta_node[surfslice] = np.unwrap(theta_raw - theta0)
            self.gdtheta_node[surfslice] = self.theta_node[surfslice] - gtheta

            ## Assign straight field-line thetas to surfaces
            #self.dtheta_surf[k] = integrand_edge / q_surf[k]
            self.theta_surf[k] = periodify(self.theta_node[surfslice])
            self.gtheta_surf[k] = periodify(gtheta)
            self.gdtheta_surf[k] = self.theta_surf[k] - self.gtheta_surf[k]

        ## Set up interpolation functions
        dr = self.diff_r @ self.gdtheta_node
        dz = self.diff_z @ self.gdtheta_node
        self.interp_gdtheta_mesh = CubicTriInterpolatorMemoized(self.rz_tri, self.gdtheta_node, kind='min_E', dz=(dr,dz))
        self.interp_gdtheta_mesh_linear = tri.LinearTriInterpolator(self.rz_tri, self.gdtheta_node)
        self.interp_q = scipy.interpolate.CubicSpline(self.psi_surf, self.q_surf)

        # Set up grids for Bicubic interpolation
        ksurf0, ksurf1 = 1, 240
        self.theta_samp = np.linspace(-np.pi, np.pi, 1024, endpoint=False)
        self.psi_samp = np.linspace(self.psi_surf[ksurf0], self.psi_surf[ksurf1], 512, endpoint=True)

        # First, resample theta on each flux surface to a regular grid
        gdtheta_samp = np.empty((ksurf1-ksurf0, len(self.theta_samp)))
        for k in range(ksurf0, ksurf1):
            interp_gdtheta = scipy.interpolate.CubicSpline(self.gtheta_surf[k], self.gdtheta_surf[k], bc_type='periodic')
            gdtheta_samp[k-ksurf0,:] = interp_gdtheta(self.theta_samp)
        
        # Then, resample psi on a regular grid
        self.gdtheta_grid = np.empty((len(self.psi_samp), len(self.theta_samp)))
        for j in range(len(self.theta_samp)):
            interp_gdtheta = scipy.interpolate.CubicSpline(self.psi_surf[ksurf0:ksurf1], gdtheta_samp[:,j], bc_type='natural')
            self.gdtheta_grid[:,j] = interp_gdtheta(self.psi_samp)

        # Set up the interpolator
        self.interp_gdtheta_grid = BicubicInterpolator(self.gdtheta_grid, ([self.psi_samp[0], self.psi_samp[-1]], [-np.pi, np.pi]), bc_type=['natural', 'periodic'])

        # Save the theta mode
        self.theta0_mode = theta0_mode

    def compute_theta(self, r, z):
        """
        Convenience function for computing psi and the straight field-line theta for
        given coordinates r, z.
        """
        gdtheta = self.interp_gdtheta_mesh(r, z)
        gtheta = np.arctan2(z - self.zaxis, r - self.raxis)
        theta = gtheta + gdtheta

        return theta

    def assemble_matrices(self):
        """
        Assembles sparse matrices for finite differences
        """
        ntri = self.ntri
        nnode = self.nnode
        triangles = self.triangles
        rz_node = self.rz_node

        # R, Z, component of the derivative inside triangles
        data_node2tri_r = np.empty(3*ntri)
        data_node2tri_z = np.empty(3*ntri)
        # Weight by which to apply to the derivative when accumulating back into triangles
        data_tri2node = np.empty(3*ntri)
        diag_tri2node_weight = np.zeros(nnode)

        # Row, column indices of mapping between triangles and nodes
        rowind_node2tri = np.empty(3*ntri)
        colind_node2tri = np.empty(3*ntri) # Column corresponds to input

        for k in tqdm(range(ntri), desc='Assembling matrices:'):
            # Set the row and column indices
            colind_node2tri[3*k:3*k+3] = triangles[k,:]
            rowind_node2tri[3*k:3*k+3] = k

            # Vectors pointing from one node to another
            e0 = rz_node[triangles[k,1],:] - rz_node[triangles[k,0],:]
            e1 = rz_node[triangles[k,2],:] - rz_node[triangles[k,1],:]

            ## Solve the equations e0 . grad(phi) = phi1 - phi0 and e1 . grad(phi) = phi2 - phi1
            # to get grad(phi) in terms of [phi0, phi1, phi2]
            edgemat = np.array([e0, e1])
            diffmat = np.array([[-1,1,0],[0,-1,1]])

            drzmat = np.linalg.solve(edgemat, diffmat)
            data_node2tri_r[3*k:3*k+3] = drzmat[0,:]
            data_node2tri_z[3*k:3*k+3] = drzmat[1,:]

            # Compute the area of the triangle
            area = np.abs(np.cross(e0, e1))
            data_tri2node[3*k:3*k+3] = area
            diag_tri2node_weight[triangles[k,:]] += area

    
        node2tri_r = scipy.sparse.csr_array((data_node2tri_r, (rowind_node2tri, colind_node2tri)), shape=(ntri,nnode))
        node2tri_z = scipy.sparse.csr_array((data_node2tri_z, (rowind_node2tri, colind_node2tri)), shape=(ntri,nnode))
        tri2node = scipy.sparse.csr_array((data_tri2node, (colind_node2tri, rowind_node2tri)), shape=(nnode,ntri))
        node_norm = scipy.sparse.dia_array((np.array([1.0/diag_tri2node_weight]), np.array([0])), shape=(nnode,nnode))

        self.diff_r = node_norm @ (tri2node @ node2tri_r)
        self.diff_z = node_norm @ (tri2node @ node2tri_z)

    def save_matrices_to_file(self, fdmat_filename: str, min_e_filename: str = ''):
        """
        Save assembled finite difference matrices
        """
        with open(fdmat_filename, 'wb') as f:
            pickle.dump((self.diff_r, self.diff_z), f)
        if min_e_filename != '':
            _DOF_estimator_min_E_memoized.save_matrices_to_file(min_e_filename)

    def assemble_jmat(self, mu, pm, pz):
        """
        Assemble sparse gyroaveraging matrices for a particular mu.

        mu is the given magnetic moment, pm is the particle mass, pz is the particle z
        """
        geom = self

        trifinder = geom.rz_tri.get_trifinder()

        ## Number of points to take for the gyroaverage
        ngyro = 32
        gyroangle = np.linspace(0,2*np.pi, num=ngyro, endpoint=False)

        ## Arrays to hold sparse gyroaverage matrix
        data_jmat = np.empty(3*ngyro*geom.nnode)
        rowind_jmat = np.empty(3*ngyro*geom.nnode, dtype=int)
        colind_jmat = np.empty(3*ngyro*geom.nnode, dtype=int)

        for knode in tqdm(range(geom.nnode)):
            # Position of node
            rz = geom.rz_node[knode,:]

            ## Magnetic info
            # NOTE: b_node is stored in (R, Z, phi) rather than (R, phi, Z) coordinates
            bv = geom.b_node[knode,:]
            modb = np.linalg.norm(bv)
            bu = bv / modb

            # Gyroradius
            rho = np.sqrt(2 * mu * modb * pm) / modb / np.abs(pz)

            ## Frame vectors
            # (b, e2, e3) form an orthonormal basis
            e2 = np.cross(bu, [0,1,0]) / np.linalg.norm(np.cross(bu, [0,1,0]))
            e3 = np.cross(bu, e2)
            # lr, lz are the dual basis vectors of (b, R, Z), used to compute the intersection
            # between the fluxtube and the R,Z plane
            lr = np.array([1, 0, -bu[0]/bu[2]])
            lz = np.array([0, 1, -bu[1]/bu[2]])

            # Compute the intersection gyroradius-sized flux tube with the (R,Z) plane
            rhovec = np.outer(np.cos(gyroangle) * rho, e2) + np.outer(np.sin(gyroangle) * rho, e3)
            r_gy_flux = rz[0] + rhovec @ lr
            z_gy_flux = rz[1] + rhovec @ lz

            # Compute the triangles touched by the gyroaverage points
            tris = trifinder(r_gy_flux, z_gy_flux)

            # Set row index (i.e. output node) of the sparse matrix
            rowind_jmat[knode*3*ngyro:(knode+1)*3*ngyro] = knode

            gyweight = 1.0 / np.sum(tris >= 0)

            for kg in range(ngyro):
                ktri = tris[kg]

                if ktri < 0:
                    # Lies outside of a triangle
                    colind_jmat[knode*3*ngyro+kg*3:knode*3*ngyro+(kg+1)*3] = knode
                    data_jmat[knode*3*ngyro+kg*3:knode*3*ngyro+(kg+1)*3] = 0
                else:
                    nodes = geom.rz_tri.triangles[ktri,:]
                    
                    # Set col index (i.e. input node) of the sparse matrix
                    colind_jmat[knode*3*ngyro+kg*3:knode*3*ngyro+(kg+1)*3] = nodes
                    data_jmat[knode*3*ngyro+kg*3:knode*3*ngyro+(kg+1)*3] = \
                            barycentric_weights(r_gy_flux[kg], z_gy_flux[kg],
                                                geom.rz_node[nodes[0],0], geom.rz_node[nodes[0],1],
                                                geom.rz_node[nodes[1],0], geom.rz_node[nodes[1],1],
                                                geom.rz_node[nodes[2],0], geom.rz_node[nodes[2],1]) * gyweight
                    

        jmat = scipy.sparse.csr_array((data_jmat, (rowind_jmat, colind_jmat)))

        return jmat