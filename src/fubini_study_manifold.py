#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module : fubini_study_manifold.py
Calcul de la métrique quantique de Fubini-Study g_{Phi,Phi}, de la distance
géodésique de Bures et projection sur le réseau de 21 qubits Fluxonium.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.linalg import eigh


@dataclass
class FubiniStudyConfig:
    E_J_GHz: float = 4.0
    E_C_GHz: float = 1.0
    E_L_GHz: float = 0.5
    n_grid: int = 400
    phi_bound: float = 5.0 * np.pi
    dphi_ext: float = 1e-4


class FluxoniumQuantumGeometry:
    """Évalue la géométrie différentielle de l'espace projectif du Fluxonium."""

    def __init__(self, cfg: FubiniStudyConfig):
        self.cfg = cfg
        self.phi = np.linspace(-cfg.phi_bound, cfg.phi_bound, cfg.n_grid)
        self.h = self.phi[1] - self.phi[0]
        self.T_kin = self._build_kinetic()

    def _build_kinetic(self) -> np.ndarray:
        N = self.cfg.n_grid
        d2 = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            d2[i, i] = -30.0
            if i - 1 >= 0:
                d2[i, i - 1] = 16.0
            if i + 1 < N:
                d2[i, i + 1] = 16.0
            if i - 2 >= 0:
                d2[i, i - 2] = -1.0
            if i + 2 < N:
                d2[i, i + 2] = -1.0
        return -4.0 * self.cfg.E_C_GHz * (d2 / (12.0 * (self.h ** 2)))

    def ground_state(self, phi_ext: float) -> Tuple[float, np.ndarray]:
        """Retourne (E0, |psi0(phi_ext)>) normalisé."""
        V = 0.5 * self.cfg.E_L_GHz * (self.phi ** 2) - self.cfg.E_J_GHz * np.cos(self.phi - phi_ext)
        vals, vecs = eigh(self.T_kin + np.diag(V), subset_by_index=[0, 1])
        psi0 = vecs[:, 0]
        psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * self.h)
        return float(vals[0]), psi0

    def metric_component(self, phi_ext: float) -> float:
        """Calcule g_{Phi,Phi}(phi_ext) par différence finie symétrique."""
        eps = self.cfg.dphi_ext
        _, psi_minus = self.ground_state(phi_ext - eps)
        _, psi_plus = self.ground_state(phi_ext + eps)
        overlap = abs(np.sum(np.conj(psi_minus) * psi_plus) * self.h)
        overlap = min(1.0, overlap)
        return float((1.0 - overlap ** 2) / ((2.0 * eps) ** 2))

    def bures_geodesic_distance(self, phi_a: float, phi_b: float) -> float:
        """Distance géodésique exacte D_FS = arccos(|<psi(a)|psi(b)>|) sur CP^N."""
        _, psi_a = self.ground_state(phi_a)
        _, psi_b = self.ground_state(phi_b)
        overlap = min(1.0, abs(np.sum(np.conj(psi_a) * psi_b) * self.h))
        return float(np.arccos(overlap))

    def project_onto_21q_lattice(
        self,
        gradient_flux: float = 0.04,
        sigma_flux_uPhi0: float = 1.5
    ) -> Dict[str, np.ndarray]:
        """
        Projette la métrique g_{Phi,Phi} sur les 21 nœuds (2 patchs d=3 + 3 ancillaires)
        et en déduit la susceptibilité géométrique au déphasage.
        """
        n_qubits = 21
        x_coords = np.linspace(-1.0, 1.0, n_qubits)
        delta_phi = gradient_flux * np.exp(-0.5 * (x_coords / 0.6) ** 2) * np.sin(np.pi * x_coords)
        phi_nodes = np.pi + delta_phi

        g_nodes = np.array([self.metric_component(p) for p in phi_nodes])
        d_bures = np.array([self.bures_geodesic_distance(np.pi, p) for p in phi_nodes])

        # Variance de phase géométrique induite par le bruit de flux
        sigma_rad = (sigma_flux_uPhi0 * 1e-6) * (2.0 * np.pi)
        geometric_dephasing_rate_Hz = g_nodes * (sigma_rad ** 2) * 1e9

        return {
            "phi_nodes": phi_nodes,
            "g_PhiPhi_nodes": g_nodes,
            "bures_distance_from_pi": d_bures,
            "geometric_dephasing_Hz": geometric_dephasing_rate_Hz
        }


if __name__ == "__main__":
    geom = FluxoniumQuantumGeometry(FubiniStudyConfig())
    proj = geom.project_onto_21q_lattice(gradient_flux=0.04)

    print("--- Projection Métrique Fubini-Study sur Réseau 21 Qubits ---")
    print(f"  g_PhiPhi au centre (noeud 10, Phi=pi) : {proj['g_PhiPhi_nodes'][10]:.4f} rad^-2")
    print(f"  g_PhiPhi max sur le réseau            : {np.max(proj['g_PhiPhi_nodes']):.4f} rad^-2")
    print(f"  Distance géodésique max à pi (D_FS)   : {np.max(proj['bures_distance_from_pi']):.5f} rad")
    print(f"  Taux déphasage géométrique max        : {np.max(proj['geometric_dephasing_Hz']):.4f} Hz")
