"""
Contains codes for setting up the initial conditions for particle analysis, and includes some implementations.
"""

import numpy as np

from typing import Protocol, NamedTuple

from ..fields.equilibrium import Equilibrium
from ..fields.geometry_handlers import XgcGeomHandler
from ..fields.field_handlers import XgcZonalFieldHandler, XgcFieldHandler
from . import particle_tools
from .particle_motion import ParticleParams, RotatingFrameInfo

# %%

class IntegralTuple(NamedTuple):
    """
    A named tuple to hold the integrals of motion for a particle.
    Contains the rotating frame info, Hamiltonian, angular momentum, and magnetic moment.
    """
    frame: RotatingFrameInfo
    ham: float
    lphi: float
    mu: float

class AnalysisData(Protocol):
    """
    Protocol for analysis data that contains initial values for the particle motion.
    """
    def get_reference_data(self) -> tuple[float, np.ndarray]:
        """
        Get the reference position of the particle. Should return a tuple of the form:
        t0, [r0, phi0, z0, vll0, mu0]
        """
        ...

    def get_reference_integrals(self) -> IntegralTuple:
        """
        Get the reference integrals of motion for the particle. Should return a tuple of the form:
        (frame, ham0, lphi0, mu0)
        """
        ...

    def get_name(self) -> str:
        """
        Get the name of the analysis data.
        """
        ...

class MidplaneAnalysisData(AnalysisData):
    def __init__(self, name: str, ksurf: int, ev0: float, xi0: float, pp: ParticleParams, geom: XgcGeomHandler, zonalFields: XgcZonalFieldHandler, frame: RotatingFrameInfo):
        """
        Initialize the midplane analysis data with the initial values for the particle motion.

        Parameters
        ----------
        ksurf0 : int
            The index of the surface where the particle is initialized.
        ev0 : float
            The initial kinetic energy of the particle in keV.
        xi0 : float
            The initial cosine of the pitch angle of the particle. Should take values [-1, 1], with 1 being parallel to the magnetic field
        """

        ## Store initial data
        self.ev0 = ev0
        self.xi0 = xi0
        self.ksurf0 = ksurf
        self.geom = geom
        self.pp = pp
        self.zonalFields = zonalFields
        self.frame = frame
        self.name = name

        # Unpack equilibrium for convenience
        eq = geom.eq

        ## Set initial position, which in this case is (approximately) the outboard midplane
        r0 = eq.interp_router(geom.psi_surf[ksurf]/eq.psix)
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

        # Particle kinetic energy in keV and cos(pitch angle)
        # Set the initial parallel velocity
        vll0 = vll_mean + pp.vt * xi0 * np.sqrt(ev0)
        # Initial magnetic moment
        mu0 = pp.m * (1-xi0**2) * (pp.vt * np.sqrt(ev0))**2 / 2 / modb

        ## Store initial data
        self.t0 = frame.t0
        self.x0 = x0
        self.vll0 = vll0
        self.mu0 = mu0


    def get_reference_data(self) -> tuple[float, np.ndarray]:
        """
        Get the reference data for the midplane analysis.
        """
        return self.t0, np.concatenate((self.x0, [self.vll0, self.mu0]))
    
    def get_reference_integrals(self) -> IntegralTuple:
        """
        Get the reference integrals of motion for the midplane analysis.
        """
        # Compute initial value of the integrals
        ham0, lphi0 = particle_tools.compute_integrals_dk(self.t0, np.concatenate((self.x0, [self.vll0, self.mu0])), self.geom.eq, self.pp, self.zonalFields, self.frame)

        return IntegralTuple(self.frame, ham0, lphi0, self.mu0)

    def get_name(self) -> str:
        """
        Get the name of the midplane analysis data.
        """
        return self.name