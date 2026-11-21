#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUXONIUM MODEL — SOLVEUR SPECTRAL
===================================

Hamiltonien adimensionné en fréquence (GHz) :

    H/h = 4 EC n²
          + EL/2 (phi - phi_ext)²
          - EJ cos(phi)

Convention :
    n = -i d/dphi
    phi_ext = 2*pi*Phi_ext/Phi0

Unités :
    EJ, EC, EL       : GHz
    phi, phi_ext     : radians
    énergies propres : GHz
    T1, T_phi        : secondes

Méthode :
    - discrétisation sur grille uniforme ;
    - conditions aux limites de Dirichlet ;
    - Laplacien à trois diagonales ;
    - diagonalisation symétrique tridiagonale ;
    - normalisation des états propres par intégration discrète.

Limites numériques :
    - n_g non nul non implémenté ;
    - dérivée discrète centrée pour l'opérateur de charge ;
    - les éléments de matrice de charge dépendent de cette
      discrétisation et des conditions aux limites.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Optional

import numpy as np
from scipy.linalg import eigh_tridiagonal


# ============================================================
# CONSTANTES PHYSIQUES
# ============================================================

@dataclass(frozen=True)
class PhysicalConstants:
    e: float = 1.602176634e-19
    h: float = 6.62607015e-34
    hbar: float = 1.054571817e-34
    phi0: float = 2.067833848e-15
    k_B: float = 1.380649e-23
    R_Q: float = 6.4532e3


CONSTANTS = PhysicalConstants()


# ============================================================
# PARAMÈTRES MATÉRIELS
# ============================================================

@dataclass
class FluxoniumHardwareParams:

    E_J_GHz: float
    E_C_GHz: float
    E_L_GHz: float

    temperature_mK: float = 20.0

    Q_cap: float = 1e6
    Q_ind: float = 1e6

    x_qp: float = 0.0
    A_flux_uPhi0: float = 0.0

    def __post_init__(self):

        for name in ("E_J_GHz", "E_C_GHz", "E_L_GHz"):
            value = float(getattr(self, name))

            if not isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} doit être fini et strictement positif."
                )

        if not isfinite(self.temperature_mK) or self.temperature_mK < 0:
            raise ValueError("temperature_mK doit être positif ou nul.")

        if not isfinite(self.Q_cap) or self.Q_cap <= 0:
            raise ValueError("Q_cap doit être strictement positif.")

        if not isfinite(self.Q_ind) or self.Q_ind <= 0:
            raise ValueError("Q_ind doit être strictement positif.")

        if not isfinite(self.x_qp) or self.x_qp < 0:
            raise ValueError("x_qp doit être positif ou nul.")

        if not isfinite(self.A_flux_uPhi0) or self.A_flux_uPhi0 < 0:
            raise ValueError("A_flux_uPhi0 doit être positif ou nul.")

    @property
    def EJ_J(self) -> float:
        return self.E_J_GHz * 1e9 * CONSTANTS.h

    @property
    def EC_J(self) -> float:
        return self.E_C_GHz * 1e9 * CONSTANTS.h

    @property
    def EL_J(self) -> float:
        return self.E_L_GHz * 1e9 * CONSTANTS.h

    @property
    def capacitance_F(self) -> float:
        return CONSTANTS.e**2 / (2.0 * self.EC_J)

    @property
    def inductance_H(self) -> float:
        return CONSTANTS.phi0**2 / (4.0 * pi**2 * self.EL_J)

    @property
    def critical_current_A(self) -> float:
        return self.EJ_J * 2.0 * pi / CONSTANTS.phi0

    @property
    def impedance_ohm(self) -> float:
        return np.sqrt(
            self.inductance_H / self.capacitance_F
        )


# ============================================================
# RÉSULTAT SPECTRAL
# ============================================================

@dataclass
class FluxoniumResult:

    phi: np.ndarray
    energies_GHz: np.ndarray
    states: np.ndarray

    dphi: float

    matrix_elements: dict

    phi_ext: float

    @property
    def frequencies_GHz(self) -> np.ndarray:
        return np.diff(self.energies_GHz)

    @property
    def f01_GHz(self) -> float:
        return float(
            self.energies_GHz[1] - self.energies_GHz[0]
        )


# ============================================================
# VALIDATION DES TEMPS DE COHÉRENCE
# ============================================================

class DecoherenceAnalyzer:

    @staticmethod
    def validate_coherence_times(
        T1_s: float,
        T_phi_s: float,
    ) -> dict:

        T1 = float(T1_s)
        T_phi = float(T_phi_s)

        if not isfinite(T1) or T1 <= 0:
            raise ValueError(
                "T1_s doit être fini et strictement positif."
            )

        if np.isnan(T_phi) or T_phi <= 0:
            raise ValueError(
                "T_phi_s doit être strictement positif. "
                "+inf est autorisé."
            )

        gamma_1 = 1.0 / T1

        gamma_phi = (
            0.0 if np.isinf(T_phi)
            else 1.0 / T_phi
        )

        gamma_2 = 0.5 * gamma_1 + gamma_phi

        T2 = 1.0 / gamma_2

        return {
            "T1_s": T1,
            "T_phi_s": T_phi,
            "T2_s": float(T2),
            "gamma_1_per_s": float(gamma_1),
            "gamma_phi_per_s": float(gamma_phi),
            "gamma_2_per_s": float(gamma_2),
        }


# ============================================================
# SOLVEUR HAMILTONIEN
# ============================================================

class FluxoniumHamiltonianSolver:

    def __init__(
        self,
        hardware: FluxoniumHardwareParams,
        n_grid: int = 500,
        phi_bound: float = 5.0 * pi,
        n_states: int = 6,
    ):

        if not isinstance(hardware, FluxoniumHardwareParams):
            raise TypeError(
                "hardware doit être FluxoniumHardwareParams."
            )

        if isinstance(n_grid, bool) or int(n_grid) != n_grid:
            raise ValueError("n_grid doit être un entier.")

        if n_grid < 20:
            raise ValueError("n_grid doit être >= 20.")

        if not isfinite(phi_bound) or phi_bound <= 0:
            raise ValueError("phi_bound doit être strictement positif.")

        if isinstance(n_states, bool) or int(n_states) != n_states:
            raise ValueError("n_states doit être un entier.")

        if n_states < 3 or n_states >= n_grid:
            raise ValueError(
                "n_states doit être compris entre 3 et n_grid - 1."
            )

        self.hardware = hardware
        self.n_grid = int(n_grid)
        self.phi_bound = float(phi_bound)
        self.n_states = int(n_states)

        # Grille intérieure : les extrémités sont imposées à zéro.
        self.phi = np.linspace(
            -self.phi_bound,
            self.phi_bound,
            self.n_grid + 2,
            dtype=float,
        )[1:-1]

        self.dphi = float(self.phi[1] - self.phi[0])

        self.D1 = self._build_first_derivative()
        self.D2 = self._build_second_derivative()

        # Convention n = -i d/dphi.
        self.n_op = -1j * self.D1

    # --------------------------------------------------------
    # OPÉRATEURS DISCRETS
    # --------------------------------------------------------

    def _build_first_derivative(self) -> np.ndarray:

        n = self.n_grid
        h = self.dphi

        D1 = np.zeros((n, n), dtype=float)

        indices = np.arange(n - 1)

        D1[indices, indices + 1] = 1.0 / (2.0 * h)
        D1[indices + 1, indices] = -1.0 / (2.0 * h)

        return D1

    def _build_second_derivative(self) -> np.ndarray:

        n = self.n_grid
        h2 = self.dphi**2

        D2 = np.zeros((n, n), dtype=float)

        indices = np.arange(n)

        D2[indices, indices] = -2.0 / h2

        off = np.arange(n - 1)

        D2[off, off + 1] = 1.0 / h2
        D2[off + 1, off] = 1.0 / h2

        return D2

    # --------------------------------------------------------
    # POTENTIEL
    # --------------------------------------------------------

    def potential(
        self,
        phi_ext: float,
    ) -> np.ndarray:

        if not isfinite(phi_ext):
            raise ValueError("phi_ext doit être fini.")

        EJ = self.hardware.E_J_GHz
        EL = self.hardware.E_L_GHz

        return (
            0.5 * EL * (self.phi - phi_ext)**2
            - EJ * np.cos(self.phi)
        )

    # --------------------------------------------------------
    # DIAGONALISATION
    # --------------------------------------------------------

    def solve(
        self,
        phi_ext: float = pi,
        n_states: Optional[int] = None,
    ) -> FluxoniumResult:

        if not isfinite(phi_ext):
            raise ValueError("phi_ext doit être fini.")

        count = self.n_states if n_states is None else int(n_states)

        if count < 2 or count >= self.n_grid:
            raise ValueError(
                "n_states doit être compris entre 2 et n_grid - 1."
            )

        EC = self.hardware.E_C_GHz

        # H = -4 EC D2 + V
        #
        # Diagonale de -4 EC D2 :
        #       8 EC / dphi²
        #
        # Hors-diagonale :
        #      -4 EC / dphi²

        kinetic_diag = 8.0 * EC / self.dphi**2
        kinetic_off = -4.0 * EC / self.dphi**2

        V = self.potential(phi_ext)

        diagonal = kinetic_diag + V

        off_diagonal = np.full(
            self.n_grid - 1,
            kinetic_off,
            dtype=float,
        )

        energies, states = eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(0, count - 1),
            check_finite=True,
        )

        # Normalisation intégrale :
        # sum(|psi|²) * dphi = 1
        states = states / np.sqrt(self.dphi)

        matrix_elements = self.compute_matrix_elements(states)

        return FluxoniumResult(
            phi=self.phi.copy(),
            energies_GHz=np.asarray(energies, dtype=float),
            states=np.asarray(states, dtype=float),
            dphi=self.dphi,
            matrix_elements=matrix_elements,
            phi_ext=float(phi_ext),
        )

    # --------------------------------------------------------
    # ÉLÉMENTS DE MATRICE
    # --------------------------------------------------------

    def compute_matrix_elements(
        self,
        states: np.ndarray,
    ) -> dict:

        states = np.asarray(states)

        if states.ndim != 2 or states.shape[0] != self.n_grid:
            raise ValueError(
                "states doit avoir la forme (n_grid, n_states)."
            )

        if not np.all(np.isfinite(states)):
            raise ValueError("Les états doivent être finis.")

        # Élément de matrice de phi :
        #
        # <i|phi|j> = somme(phi * psi_i* * psi_j) dphi

        phi_elements = (
            states.conj().T
            @ (self.phi[:, None] * states)
        ) * self.dphi

        # Élément de matrice de charge :
        #
        # <i|n|j> = somme(psi_i* * n_op psi_j) dphi
        #
        # La dérivée centrée et les conditions aux limites
        # ne reproduisent pas exactement le commutateur canonique.

        n_states = self.n_op @ states

        charge_elements = (
            states.conj().T @ n_states
        ) * self.dphi

        return {
            "phi": phi_elements,
            "n": charge_elements,
            "phi_01": complex(phi_elements[0, 1]),
            "n_01": complex(charge_elements[0, 1]),
        }

    # --------------------------------------------------------
    # BALAYAGE EN FLUX
    # --------------------------------------------------------

    def sweep_flux(
        self,
        phi_ext_values: np.ndarray,
        n_states: Optional[int] = None,
    ) -> dict:

        phases = np.asarray(
            phi_ext_values,
            dtype=float,
        )

        if phases.ndim != 1 or phases.size == 0:
            raise ValueError(
                "phi_ext_values doit être un tableau 1D non vide."
            )

        if not np.all(np.isfinite(phases)):
            raise ValueError(
                "Les phases doivent toutes être finies."
            )

        count = self.n_states if n_states is None else int(n_states)

        if count < 3 or count >= self.n_grid:
            raise ValueError(
                "Le balayage exige au moins 3 états."
            )

        energies = np.empty(
            (phases.size, count),
            dtype=float,
        )

        for index, phase in enumerate(phases):

            result = self.solve(
                phi_ext=float(phase),
                n_states=count,
            )

            energies[index] = result.energies_GHz

        transitions = np.diff(energies, axis=1)

        return {
            "phi_ext": phases.copy(),
            "energies_GHz": energies,
            "transitions_GHz": transitions,
            "f01_GHz": transitions[:, 0],
            "f12_GHz": transitions[:, 1],
        }


__all__ = [
    "PhysicalConstants",
    "FluxoniumHardwareParams",
    "FluxoniumResult",
    "DecoherenceAnalyzer",
    "FluxoniumHamiltonianSolver",
]
