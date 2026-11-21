
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUXONIUM MODEL
===============

Solveur spectral par différences finies dans la base de phase.

Convention :
    H / h = 4 EC n² + EL/2 (phi - phi_ext)² - EJ cos(phi)

Unités :
    EJ, EC, EL : GHz
    phi, phi_ext : radians (coordonnées de phase réduites)
    energies : GHz

Le module calcule :
    - spectre du Fluxonium ;
    - états propres ;
    - éléments de matrice de phi et n ;
    - fréquence f01 et anharmonicité ;
    - dérivées numériques en flux.

Il ne prétend pas déduire T1 ou T_phi du Hamiltonien statique.
Ces temps exigent des modèles de dissipation et des densités spectrales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class PhysicalConstants:
    h: float = 6.62607015e-34
    hbar: float = 1.054571817e-34
    e: float = 1.602176634e-19
    k_B: float = 1.380649e-23
    Phi_0: float = 2.067833848e-15
    phi_0: float = 3.291059785e-16
    R_Q: float = 6453.20175


CONST = PhysicalConstants()


@dataclass
class FluxoniumHardwareParams:
    name: str = "Fluxonium"
    E_J_GHz: float = 4.0
    E_C_GHz: float = 1.0
    E_L_GHz: float = 0.5
    temperature_mK: float = 15.0
    Q_cap: float = 1.5e6
    Q_ind: float = 8.0e5
    x_qp: float = 1.0e-7
    A_flux_uPhi0: float = 1.5

    L_nH: float = field(init=False)
    C_fF: float = field(init=False)
    I_c_nA: float = field(init=False)
    Z_char_Ohms: float = field(init=False)

    def __post_init__(self):
        for name in ("E_J_GHz", "E_C_GHz", "E_L_GHz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} doit être strictement positif.")

        EJ_J = self.E_J_GHz * 1e9 * CONST.h
        EC_J = self.E_C_GHz * 1e9 * CONST.h
        EL_J = self.E_L_GHz * 1e9 * CONST.h

        self.L_nH = (CONST.phi_0 ** 2 / EL_J) * 1e9
        self.C_fF = (CONST.e ** 2 / (2.0 * EC_J)) * 1e15
        self.I_c_nA = (EJ_J / CONST.phi_0) * 1e9
        self.Z_char_Ohms = math.sqrt(
            (self.L_nH * 1e-9) / (self.C_fF * 1e-15)
        )


class FluxoniumHamiltonianSolver:
    """
    Solveur du Hamiltonien Fluxonium.

    La grille contient uniquement les points intérieurs.
    Les conditions aux limites de Dirichlet sont ainsi imposées
    aux extrémités du domaine.

    Le Laplacien utilise un stencil centré d'ordre 2, symétrique.
    L'opérateur de charge est construit avec une dérivée centrée
    antisymétrique, donc n_op est hermitien.
    """

    def __init__(
        self,
        params: FluxoniumHardwareParams,
        n_grid: int = 500,
        phi_bound: float = 5.0 * np.pi,
        n_states: int = 6,
    ):
        if n_grid < 20:
            raise ValueError("n_grid doit être supérieur ou égal à 20.")
        if not math.isfinite(phi_bound) or phi_bound <= 0.0:
            raise ValueError("phi_bound doit être strictement positif.")
        if not 2 <= n_states < n_grid:
            raise ValueError("n_states doit être compris entre 2 et n_grid-1.")

        self.params = params
        self.n_grid = int(n_grid)
        self.phi_bound = float(phi_bound)
        self.n_states = int(n_states)

        # Points intérieurs : les deux frontières sont fixées à zéro.
        full_grid = np.linspace(
            -self.phi_bound,
            self.phi_bound,
            self.n_grid + 2,
        )
        self.phi = full_grid[1:-1]
        self.dphi = float(full_grid[1] - full_grid[0])

        self._build_operators()

    def _build_operators(self) -> None:
        N = self.n_grid
        h = self.dphi

        # Laplacien tridiagonal symétrique.
        D2 = np.zeros((N, N), dtype=np.float64)
        idx = np.arange(N)

        D2[idx, idx] = -2.0
        D2[idx[1:], idx[:-1]] = 1.0
        D2[idx[:-1], idx[1:]] = 1.0
        D2 /= h ** 2

        # Dérivée première centrée.
        # La matrice est antisymétrique réelle.
        D1 = np.zeros((N, N), dtype=np.float64)
        D1[idx[:-1], idx[1:]] = 1.0
        D1[idx[1:], idx[:-1]] = -1.0
        D1 /= 2.0 * h

        self.D1 = D1
        self.D2 = D2
        self.n_op = -1j * D1

        if not np.allclose(
            self.n_op,
            self.n_op.conj().T,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("L'opérateur n n'est pas hermitien.")

        if not np.allclose(
            self.D2,
            self.D2.T,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("Le Laplacien n'est pas symétrique.")

    def potential(
        self,
        phi_ext: float,
        E_L_override: Optional[float] = None,
    ) -> np.ndarray:
        if not math.isfinite(float(phi_ext)):
            raise ValueError("phi_ext doit être fini.")

        EL = (
            self.params.E_L_GHz
            if E_L_override is None
            else float(E_L_override)
        )

        if not math.isfinite(EL) or EL <= 0.0:
            raise ValueError("E_L_override doit être positif.")

        return (
            0.5 * EL * (self.phi - phi_ext) ** 2
            - self.params.E_J_GHz * np.cos(self.phi)
        )

    def solve(
        self,
        phi_ext: float = np.pi,
        n_g: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Résout le problème spectral.

        n_g non nul n'est pas activé dans cette discrétisation :
        sa prise en charge exige un traitement cohérent de la
        condition de jauge et des conditions aux limites.
        """
        if abs(float(n_g)) > 1e-14:
            raise NotImplementedError(
                "n_g non nul n'est pas pris en charge par cette "
                "discrétisation à conditions de Dirichlet."
            )

        EC = self.params.E_C_GHz
        V = self.potential(phi_ext)

        H = -4.0 * EC * self.D2 + np.diag(V)

        # Contrôle de l'hermiticité avant diagonalisation.
        if not np.allclose(H, H.T, atol=1e-11, rtol=0.0):
            raise RuntimeError("Hamiltonien non symétrique.")

        energies, states = eigh(
            H,
            subset_by_index=[0, self.n_states - 1],
            check_finite=True,
        )

        # Normalisation intégrale : ∫ |psi|² dphi = 1.
        for k in range(self.n_states):
            norm = math.sqrt(
                float(np.sum(np.abs(states[:, k]) ** 2) * self.dphi)
            )

            if norm <= 0.0 or not math.isfinite(norm):
                raise RuntimeError(f"État propre {k} mal normalisé.")

            states[:, k] /= norm

            # Fixe la phase globale pour stabiliser les comparaisons.
            idx_max = int(np.argmax(np.abs(states[:, k])))
            amplitude = states[idx_max, k]

            if abs(amplitude) > 0.0:
                states[:, k] *= np.exp(-1j * np.angle(amplitude))

        return {
            "energies": np.asarray(energies, dtype=float),
            "states": states,
            "potential": V,
            "phi": self.phi.copy(),
        }

    def compute_matrix_elements(
        self,
        states: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        states = np.asarray(states, dtype=np.complex128)

        if states.ndim != 2 or states.shape[0] != self.n_grid:
            raise ValueError("Dimension des états incompatible avec la grille.")

        k = states.shape[1]
        phi_mat = np.zeros((k, k), dtype=np.complex128)
        n_mat = np.zeros((k, k), dtype=np.complex128)

        phi_diag = np.diag(self.phi)

        for i in range(k):
            bra = states[:, i].conj()
            for j in range(k):
                ket = states[:, j]

                phi_mat[i, j] = (
                    np.vdot(states[:, i], phi_diag @ ket)
                    * self.dphi
                )
                n_mat[i, j] = (
                    np.vdot(states[:, i], self.n_op @ ket)
                    * self.dphi
                )

        return {
            "phi": phi_mat,
            "n": n_mat,
        }

    def sweep_flux(
        self,
        n_points: int = 121,
        flux_min: float = 0.0,
        flux_max: float = 2.0 * np.pi,
    ) -> Dict[str, np.ndarray]:
        if n_points < 3:
            raise ValueError("n_points doit être supérieur ou égal à 3.")
        if flux_max <= flux_min:
            raise ValueError("flux_max doit dépasser flux_min.")

        flux_grid = np.linspace(flux_min, flux_max, n_points)
        energies_all = np.zeros((n_points, self.n_states))
        phi_01 = np.zeros(n_points)

        for idx, flux in enumerate(flux_grid):
            result = self.solve(phi_ext=float(flux))
            energies_all[idx] = result["energies"]

            mats = self.compute_matrix_elements(result["states"])
            phi_01[idx] = abs(mats["phi"][0, 1])

        f01 = energies_all[:, 1] - energies_all[:, 0]
        f12 = energies_all[:, 2] - energies_all[:, 1]

        df01 = np.gradient(f01, flux_grid, edge_order=2)
        d2f01 = np.gradient(df01, flux_grid, edge_order=2)

        return {
            "flux": flux_grid,
            "energies": energies_all,
            "f01": f01,
            "f12": f12,
            "alpha": f12 - f01,
            "df01_dphi_ext": df01,
            "d2f01_dphi_ext2": d2f01,
            "phi_01": phi_01,
        }


class DecoherenceAnalyzer:
    """
    Gestion des temps de cohérence fournis par un modèle externe.

    Le Hamiltonien statique ne suffit pas à déterminer T1 ou T_phi.
    Cette classe ne fabrique donc pas de taux de relaxation.
    """

    @staticmethod
    def validate_coherence_times(
        T1_s: float,
        T_phi_s: float,
    ) -> Dict[str, float]:
        T1_s = float(T1_s)
        T_phi_s = float(T_phi_s)

        if not math.isfinite(T1_s) or T1_s <= 0.0:
            raise ValueError("T1_s doit être fini et strictement positif.")

        if math.isnan(T_phi_s) or T_phi_s <= 0.0:
            raise ValueError("T_phi_s doit être positif.")

        gamma_1 = 1.0 / T1_s
        gamma_phi = (
            0.0 if math.isinf(T_phi_s) else 1.0 / T_phi_s
        )

        gamma_2 = 0.5 * gamma_1 + gamma_phi
        T2_s = 1.0 / gamma_2

        return {
            "T1_s": T1_s,
            "T_phi_s": T_phi_s,
            "T2_s": T2_s,
            "gamma_1_s": gamma_1,
            "gamma_phi_s": gamma_phi,
            "gamma_2_s": gamma_2,
        }
