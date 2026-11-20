#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module : kerr_cat_simulator.py
Simulation d'un qubit bosonique Kerr-Cat (stabilisation d'états chats de Schrödinger
par pompage paramétrique à deux photons) :
  - Construction de l'Hamiltonien de Kerr pompé dans l'espace de Fock tronqué
  - Extraction des états chats pair/impair |C_alpha^+>, |C_alpha^-> et de l'écart tunnel
  - Calcul de la fonction de quasi-probabilité de Wigner W(x, p)
  - Évaluation du biais de bruit (suppression exponentielle des bit-flips vs phase-flips)
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from scipy.linalg import eigh, expm
from scipy.integrate import solve_ivp


@dataclass
class KerrCatParams:
    """Paramètres physiques d'un résonateur non linéaire Kerr-Cat (en MHz)."""
    name: str = "KerrCat_SNAIL_Resonator"
    K_MHz: float = 6.5                 # Non-linéarité Kerr auto-induite K / (2*pi)
    eps2_MHz: float = 26.0             # Amplitude de pompe à deux photons eps_2 / (2*pi)
    kappa_1_kHz: float = 2.5           # Taux de perte à un photon kappa_1 / (2*pi)
    kappa_2_kHz: float = 15.0          # Taux de dissipation à deux photons kappa_2 / (2*pi)
    n_fock: int = 45                   # Dimension de l'espace de Hilbert tronqué

    @property
    def alpha_amplitude(self) -> float:
        """Amplitude cohérente théorique alpha = sqrt(eps_2 / K)."""
        return math.sqrt(max(self.eps2_MHz / max(self.K_MHz, 1e-9), 0.0))

    @property
    def mean_photon_number(self) -> float:
        """Nombre moyen de photons |alpha|^2."""
        return self.eps2_MHz / max(self.K_MHz, 1e-9)


class KerrCatSimulator:
    """Solveur quantique dans la base de Fock pour l'architecture Kerr-Cat."""

    def __init__(self, params: KerrCatParams):
        self.params = params
        self.dim = params.n_fock
        self._build_fock_operators()

    def _build_fock_operators(self) -> None:
        """Construit les opérateurs d'annihilation a, de création a_dag et de parité P."""
        n_vals = np.arange(1, self.dim, dtype=np.float64)
        self.a = np.diag(np.sqrt(n_vals), k=1).astype(np.complex128)
        self.a_dag = self.a.T.conj()
        self.num_op = self.a_dag @ self.a
        self.parity_op = np.diag([(-1.0) ** k for k in range(self.dim)]).astype(np.complex128)

    def build_hamiltonian(self, detuning_MHz: float = 0.0) -> np.ndarray:
        """
        Construit H / h (en MHz) :
            H = Delta * a^dag a - K * a^dag^2 a^2 + eps_2 * (a^dag^2 + a^2)
        """
        a2 = self.a @ self.a
        adag2 = self.a_dag @ self.a_dag
        H = (
            detuning_MHz * self.num_op
            - self.params.K_MHz * (adag2 @ a2)
            + self.params.eps2_MHz * (adag2 + a2)
        )
        return H

    def solve_spectrum(self, detuning_MHz: float = 0.0, n_states: int = 8) -> Dict[str, np.ndarray]:
        """Diagonalise l'Hamiltonien et classe les états propres selon leur parité photonique."""
        H = self.build_hamiltonian(detuning_MHz=detuning_MHz)
        eigvals, eigvecs = eigh(H, subset_by_index=[0, n_states - 1])

        # Les états chats |C_alpha^+> et |C_alpha^-> sont les états les plus élevés
        # du spectre de -K a^dag^2 a^2 + eps_2 (a^dag^2 + a^2) (maximisant l'énergie -H)
        # En inversant le signe de H, les deux états dégénérés forment le fondamental.
        eigvals_inv, eigvecs_inv = eigh(-H, subset_by_index=[0, n_states - 1])
        cat_energies = -eigvals_inv

        parities = np.array([
            np.real(np.vdot(eigvecs_inv[:, i], self.parity_op @ eigvecs_inv[:, i]))
            for i in range(n_states)
        ])

        # Écart tunnel résiduel entre les deux états chats |C+> et |C->
        tunnel_splitting_MHz = abs(cat_energies[0] - cat_energies[1])
        # Écart énergétique (gap) vers les états excités hors du sous-espace chat
        excited_gap_MHz = abs(cat_energies[1] - cat_energies[2])

        return {
            "cat_energies_MHz": cat_energies,
            "cat_states": eigvecs_inv,
            "parities": parities,
            "tunnel_splitting_MHz": tunnel_splitting_MHz,
            "excited_gap_MHz": excited_gap_MHz
        }

    def coherent_state(self, alpha: complex) -> np.ndarray:
        """Génère un état cohérent |alpha> tronqué dans la base de Fock."""
        coeffs = np.zeros(self.dim, dtype=np.complex128)
        prefactor = math.exp(-0.5 * (abs(alpha) ** 2))
        term = 1.0 + 0.0j
        coeffs[0] = prefactor
        for n in range(1, self.dim):
            term *= alpha / math.sqrt(n)
            coeffs[n] = prefactor * term
        norm = np.linalg.norm(coeffs)
        return coeffs / norm if norm > 0 else coeffs

    def wigner_function(
        self,
        rho: np.ndarray,
        x_vec: np.ndarray,
        p_vec: np.ndarray
    ) -> np.ndarray:
        """
        Calcule la fonction de Wigner W(x, p) sur une grille de phase 2D :
            W(x, p) = (2 / pi) * Tr[ D(-beta) * rho * D(beta) * P ]
        avec beta = (x + i*p) / sqrt(2).
        """
        W = np.zeros((len(p_vec), len(x_vec)), dtype=np.float64)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        for ix, x in enumerate(x_vec):
            for ip, p in enumerate(p_vec):
                beta = (x + 1j * p) * inv_sqrt2
                D_beta = expm(beta * self.a_dag - np.conj(beta) * self.a)
                rho_disp = D_beta.conj().T @ rho @ D_beta
                W[ip, ix] = (2.0 / np.pi) * np.real(np.trace(rho_disp @ self.parity_op))
        return W

    def simulate_noise_bias(self, t_max_us: float = 50.0, n_steps: int = 120) -> Dict[str, np.ndarray]:
        """
        Simule l'évolution dissipative sous perte à un photon (kappa_1) pour mesurer
        la dissymétrie entre :
          - T_phase_flip (décohérence entre |+alpha> et |-alpha>, linéaire en |alpha|^2)
          - T_bit_flip   (basculement entre |+alpha> et |-alpha>, exponentiellement supprimé)
        """
        H = self.build_hamiltonian() * 2.0 * np.pi  # rad / us
        kappa_1_us = self.params.kappa_1_kHz * 1e-3 * 2.0 * np.pi
        L1 = math.sqrt(kappa_1_us) * self.a
        L1_dag = L1.conj().T
        L1_dag_L1 = L1_dag @ L1

        alpha = self.params.alpha_amplitude
        psi_plus = self.coherent_state(alpha)
        psi_minus = self.coherent_state(-alpha)

        # État initial |+alpha><+alpha| pour mesurer le temps de bit-flip
        rho0_bit = np.outer(psi_plus, psi_plus.conj())

        def rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
            rho = rho_vec.reshape((self.dim, self.dim))
            drho = -1j * (H @ rho - rho @ H)
            drho += L1 @ rho @ L1_dag - 0.5 * (L1_dag_L1 @ rho + rho @ L1_dag_L1)
            return drho.reshape(-1)

        t_eval = np.linspace(0.0, t_max_us, n_steps)
        sol = solve_ivp(rhs, (0.0, t_max_us), rho0_bit.reshape(-1), t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

        pop_plus = np.zeros(n_steps)
        pop_minus = np.zeros(n_steps)
        for idx in range(n_steps):
            rho_t = sol.y[:, idx].reshape((self.dim, self.dim))
            pop_plus[idx] = np.real(np.vdot(psi_plus, rho_t @ psi_plus))
            pop_minus[idx] = np.real(np.vdot(psi_minus, rho_t @ psi_minus))

        # Estimation des taux caractéristiques :
        # Gamma_Z ~ 2 * kappa_1 * |alpha|^2  vs  Gamma_X ~ kappa_1 * |alpha|^2 * exp(-2 |alpha|^2)
        n_bar = self.params.mean_photon_number
        gamma_Z = 2.0 * kappa_1_us * n_bar
        gamma_X = kappa_1_us * n_bar * math.exp(-2.0 * n_bar)
        bias_ratio = gamma_Z / max(gamma_X, 1e-15)

        return {
            "time_us": t_eval,
            "pop_plus_alpha": pop_plus,
            "pop_minus_alpha": pop_minus,
            "T_Z_phase_us": 1.0 / max(gamma_Z, 1e-12),
            "T_X_bit_us": 1.0 / max(gamma_X, 1e-12),
            "noise_bias_ratio": bias_ratio
        }
