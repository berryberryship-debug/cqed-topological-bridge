#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module : fubini_study_manifold.py

Calcul de la métrique quantique de Fubini-Study g_{Phi,Phi},
de la distance de Fubini-Study (angle de Bures) entre états
fondamentaux du fluxonium, et projection sur un réseau de
21 qubits Fluxonium.

Distinction terminologique (rigoureuse)
---------------------------------------
- Distance de Fubini-Study (angle de Bures) :
      D_FS(a, b) = arccos(|<psi(a)|psi(b)>|)
  C'est la distance géodésique sur l'espace projectif CP^{N-1}
  pour des états purs. C'est ce que calcule ce module.

- Distance de Bures (au sens strict) :
      D_B(a, b) = sqrt(2 * (1 - |<psi(a)|psi(b)>|))
  Ce n'est PAS ce que calcule ce module.

La métrique g_{Phi,Phi} est la partie réelle du tenseur
géométrique quantique (QGT) : elle mesure la sensibilité de
l'état fondamental aux variations du paramètre phi_ext.

Note d'architecture
-------------------
Ce module est AUTONOME : il ne dépend ni de fluxonium_hamiltonian.py
ni de fluxonium_model.py. Il possède son propre solveur (grille de
phase + eigh), ce qui en fait un module de caractérisation
géométrique séparé, conformément à l'architecture découplée.

Auteur : [Votre nom]
Licence : MIT
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.linalg import eigh


@dataclass
class FubiniStudyConfig:
    """Paramètres du fluxonium et de la grille numérique."""

    E_J_GHz: float = 4.0
    E_C_GHz: float = 1.0
    E_L_GHz: float = 0.5
    n_grid: int = 400
    phi_bound: float = 5.0 * np.pi
    dphi_ext: float = 1e-4


class FluxoniumQuantumGeometry:
    """
    Évalue la géométrie différentielle de l'espace projectif du Fluxonium.

    Métrique de Fubini-Study (différence symétrique) :
        g_{Phi,Phi}(phi) = (1 - |<psi(phi-eps)|psi(phi+eps)>|²) / (2 eps)²

    Distance de Fubini-Study (angle de Bures) :
        D_FS(a, b) = arccos(|<psi(a)|psi(b)>|)
    """

    def __init__(self, cfg: FubiniStudyConfig):
        self.cfg = cfg
        self.phi = np.linspace(
            -cfg.phi_bound, cfg.phi_bound, cfg.n_grid
        )
        self.h = self.phi[1] - self.phi[0]
        self.T_kin = self._build_kinetic()

    def _build_kinetic(self) -> np.ndarray:
        """
        Construit l'opérateur cinétique -4 EC d²/dphi²
        avec un stencil à 5 points (ordre 4).

        Coefficients : (-1, 16, -30, 16, -1) / (12 h²)
        """
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
        """
        Retourne (E0, |psi0(phi_ext)>) normalisé sur la grille.

        Potentiel : V(phi) = 0.5 EL phi² - EJ cos(phi - phi_ext)
        """
        V = (
            0.5 * self.cfg.E_L_GHz * (self.phi ** 2)
            - self.cfg.E_J_GHz * np.cos(self.phi - phi_ext)
        )
        vals, vecs = eigh(
            self.T_kin + np.diag(V), subset_by_index=[0, 1]
        )
        psi0 = vecs[:, 0]
        psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * self.h)
        return float(vals[0]), psi0

    def metric_component(self, phi_ext: float) -> float:
        """
        Calcule g_{Phi,Phi}(phi_ext) par différence finie symétrique.

        C'est la partie réelle du tenseur géométrique quantique,
        aussi appelée métrique de Fubini-Study.
        """
        eps = self.cfg.dphi_ext
        _, psi_minus = self.ground_state(phi_ext - eps)
        _, psi_plus = self.ground_state(phi_ext + eps)
        overlap = abs(np.sum(np.conj(psi_minus) * psi_plus) * self.h)
        overlap = min(1.0, overlap)
        return float((1.0 - overlap ** 2) / ((2.0 * eps) ** 2))

    def fubini_study_distance(self, phi_a: float, phi_b: float) -> float:
        """
        Distance de Fubini-Study (angle de Bures) entre deux états
        fondamentaux :

            D_FS(a, b) = arccos(|<psi(a)|psi(b)>|)

        C'est la distance géodésique sur CP^{N-1} pour des états purs.
        """
        _, psi_a = self.ground_state(phi_a)
        _, psi_b = self.ground_state(phi_b)
        overlap = min(1.0, abs(np.sum(np.conj(psi_a) * psi_b) * self.h))
        return float(np.arccos(overlap))

    # Alias rétrocompatible (ancien nom, sémantique identique)
    def bures_geodesic_distance(self, phi_a: float, phi_b: float) -> float:
        """
        Alias de fubini_study_distance.

        Note : le nom historique 'bures_geodesic_distance' est conservé
        pour compatibilité, mais la quantité calculée est bien la
        distance de Fubini-Study (angle de Bures), pas la distance
        de Bures au sens strict.
        """
        return self.fubini_study_distance(phi_a, phi_b)

    def project_onto_21q_lattice(
        self,
        gradient_flux: float = 0.04,
        sigma_flux_uPhi0: float = 1.5,
    ) -> Dict[str, np.ndarray]:
        """
        Projette la métrique g_{Phi,Phi} sur les 21 nœuds
        (2 patchs d=3 + 3 ancillaires) et en déduit la
        susceptibilité géométrique au déphasage.

        Le profil de flux est gaussien-centré, modulé par un sinus.
        """
        n_qubits = 21
        x_coords = np.linspace(-1.0, 1.0, n_qubits)
        delta_phi = (
            gradient_flux
            * np.exp(-0.5 * (x_coords / 0.6) ** 2)
            * np.sin(np.pi * x_coords)
        )
        phi_nodes = np.pi + delta_phi

        g_nodes = np.array(
            [self.metric_component(p) for p in phi_nodes]
        )
        d_fs = np.array(
            [self.fubini_study_distance(np.pi, p) for p in phi_nodes]
        )

        # Variance de phase géométrique induite par le bruit de flux
        sigma_rad = (sigma_flux_uPhi0 * 1e-6) * (2.0 * np.pi)
        geometric_dephasing_rate_Hz = g_nodes * (sigma_rad ** 2) * 1e9

        return {
            "phi_nodes": phi_nodes,
            "g_PhiPhi_nodes": g_nodes,
            "fubini_study_distance_from_pi": d_fs,
            "geometric_dephasing_Hz": geometric_dephasing_rate_Hz,
        }


# ----------------------------------------------------------------------
# Vérification physique
# ----------------------------------------------------------------------

def verify_sweet_spot_minimum():
    """
    Vérifie que g_{Phi,Phi} est minimale au sweet spot (phi_ext = pi).

    C'est une propriété physique attendue : au point doux, l'état
    fondamental est insensible au premier ordre au flux, donc la
    métrique est minimale.
    """
    geom = FluxoniumQuantumGeometry(FubiniStudyConfig())
    g_sweet = geom.metric_component(np.pi)
    g_off = geom.metric_component(np.pi * 0.95)
    assert g_sweet < g_off, (
        f"g au sweet spot ({g_sweet:.6g}) devrait être inférieur "
        f"à g hors sweet spot ({g_of
