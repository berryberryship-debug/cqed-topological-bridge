#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module : fluxonium_model.py
Modélisation spectrale, analyse de sensibilité au bruit (charge/flux)
et évaluation des temps de cohérence (T1, T_phi, T2) d'un qubit Fluxonium.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
    name: str = "Fluxonium_Protege_Pi"
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
        E_J_J = self.E_J_GHz * 1e9 * CONST.h
        E_C_J = self.E_C_GHz * 1e9 * CONST.h
        E_L_J = self.E_L_GHz * 1e9 * CONST.h

        self.L_nH = ((CONST.phi_0 ** 2) / E_L_J) * 1e9
        self.C_fF = ((CONST.e ** 2) / (2.0 * E_C_J)) * 1e15
        self.I_c_nA = (E_J_J / CONST.phi_0) * 1e9
        self.Z_char_Ohms = math.sqrt((self.L_nH * 1e-9) / (self.C_fF * 1e-15))


class FluxoniumHamiltonianSolver:
    """Solveur par différences finies d'ordre 4 dans la base de phase supraconductrice."""

    def __init__(
        self,
        params: FluxoniumHardwareParams,
        n_grid: int = 500,
        phi_bound: float = 5.0 * np.pi,
        n_states: int = 6
    ):
        self.params = params
        self.n_grid = n_grid
        self.phi_bound = phi_bound
        self.n_states = n_states
        self.phi = np.linspace(-phi_bound, phi_bound, n_grid)
        self.dphi = self.phi[1] - self.phi[0]
        self._build_operators()

    def _build_operators(self) -> None:
        N, h = self.n_grid, self.dphi
        d1 = np.zeros((N, N), dtype=np.complex128)
        d2 = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            d2[i, i] = -30.0
            if i - 1 >= 0:
                d1[i, i - 1], d2[i, i - 1] = -8.0, 16.0
            if i + 1 < N:
                d1[i, i + 1], d2[i, i + 1] = 8.0, 16.0
            if i - 2 >= 0:
                d1[i, i - 2], d2[i, i - 2] = 1.0, -1.0
            if i + 2 < N:
                d1[i, i + 2], d2[i, i + 2] = -1.0, -1.0

        self.n_op = -1j * (d1 / (12.0 * h))
        self.D2 = d2 / (12.0 * (h ** 2))

    def potential(self, phi_ext: float, E_L_override: Optional[float] = None) -> np.ndarray:
        E_L = self.params.E_L_GHz if E_L_override is None else E_L_override
        return 0.5 * E_L * (self.phi ** 2) - self.params.E_J_GHz * np.cos(self.phi - phi_ext)

    def solve(self, phi_ext: float = np.pi, n_g: float = 0.0) -> Dict[str, np.ndarray]:
        E_C = self.params.E_C_GHz
        V = self.potential(phi_ext)

        if abs(n_g) < 1e-14:
            H = -4.0 * E_C * self.D2 + np.diag(V)
        else:
            n_shift = self.n_op - n_g * np.eye(self.n_grid, dtype=np.complex128)
            H = 4.0 * E_C * (n_shift @ n_shift) + np.diag(V.astype(np.complex128))

        eigvals, eigvecs = eigh(H, subset_by_index=[0, self.n_states - 1])
        for k in range(self.n_states):
            norm = np.sqrt(np.sum(np.abs(eigvecs[:, k]) ** 2) * self.dphi)
            eigvecs[:, k] /= norm
            idx_max = np.argmax(np.abs(eigvecs[:, k]))
            eigvecs[:, k] /= eigvecs[idx_max, k] / np.abs(eigvecs[idx_max, k])

        return {
            "energies": np.real(eigvals),
            "states": eigvecs,
            "potential": V,
            "phi": self.phi
        }

    def compute_matrix_elements(self, states: np.ndarray) -> Dict[str, np.ndarray]:
        k = states.shape[1]
        phi_mat = np.zeros((k, k), dtype=np.complex128)
        n_mat = np.zeros((k, k), dtype=np.complex128)

        for i in range(k):
            bra = np.conj(states[:, i]) * self.dphi
            for j in range(k):
                ket = states[:, j]
                phi_mat[i, j] = np.sum(bra * (self.phi * ket))
                n_mat[i, j] = np.sum(bra * (self.n_op @ ket))

        return {"phi": phi_mat, "n": n_mat}

    def sweep_flux(self, n_points: int = 121) -> Dict[str, np.ndarray]:
        flux_grid = np.linspace(0.0, 2.0 * np.pi, n_points)
        energies_all = np.zeros((n_points, self.n_states))
        phi_01 = np.zeros(n_points)

        for idx, f_ext in enumerate(flux_grid):
            res = self.solve(phi_ext=f_ext)
            energies_all[idx, :] = res["energies"]
            mats = self.compute_matrix_elements(res["states"])
            phi_01[idx] = np.abs(mats["phi"][0, 1])

        f01 = energies_all[:, 1] - energies_all[:, 0]
        f12 = energies_all[:, 2] - energies_all[:, 1]
        df01 = np.gradient(f01, flux_grid)
        d2f01 = np.gradient(df01, flux_grid)

        return {
            "flux": flux_grid,
            "energies": energies_all,
            "f01": f01,
            "f12": f12,
            "alpha": f12 - f01,
            "df01_dphi": df01,
            "d2f01_dphi2": d2f01,
            "phi_01": phi_01
        }


class DecoherenceAnalyzer:
    """Calcul des taux de relaxation (T1) et de déphasage pur (T_phi)."""

    def __init__(self, params: FluxoniumHardwareParams):
        self.params = params

    def _coth_thermal(self, freq_GHz: float) -> float:
        f_Hz = max(abs(freq_GHz) * 1e9, 1e6)
        T_K = max(self.params.temperature_mK * 1e-3, 1e-3)
        x = (CONST.h * f_Hz) / (2.0 * CONST.k_B * T_K)
        return 1.0 if x > 50.0 else 1.0 / math.tanh(x)

    def relaxation_rates(self, f01_GHz: float, phi_01: float) -> Dict[str, float]:
        omega_01 = 2.0 * np.pi * max(f01_GHz, 0.01) * 1e9
        E_L_J = self.params.E_L_GHz * 1e9 * CONST.h
        E_C_J = self.params.E_C_GHz * 1e9 * CONST.h
        coth_th = self._coth_thermal(f01_GHz)

        gamma_cap = (CONST.hbar * (omega_01 ** 2)) / (4.0 * E_C_J * self.params.Q_cap) * (phi_01 ** 2) * coth_th
        gamma_ind = (E_L_J / (CONST.hbar * self.params.Q_ind)) * (phi_01 ** 2) * coth_th
        gamma_qp = self.params.x_qp * (16.0 * self.params.E_J_GHz * 1e9 / np.pi) * math.sqrt(88.0 / max(f01_GHz, 0.05)) * (phi_01 ** 2) * 0.25

        gamma_tot = gamma_cap + gamma_ind + gamma_qp
        return {
            "gamma_cap_Hz": gamma_cap,
            "gamma_ind_Hz": gamma_ind,
            "gamma_qp_Hz": gamma_qp,
            "gamma_total_Hz": gamma_tot,
            "T1_us": (1.0 / max(gamma_tot, 1.0)) * 1e6
        }

    def dephasing_rate(self, df01: float, d2f01: float) -> Dict[str, float]:
        A_rad = (self.params.A_flux_uPhi0 * 1e-6) * (2.0 * np.pi)
        D1_Hz, D2_Hz = abs(df01) * 1e9, abs(d2f01) * 1e9
        gamma_phi = 2.0 * np.pi * D1_Hz * A_rad * math.sqrt(3.5) + np.pi * D2_Hz * (A_rad ** 2) * 3.5
        return {
            "gamma_phi_Hz": gamma_phi,
            "T_phi_us": (1.0 / max(gamma_phi, 1.0)) * 1e6
        }
