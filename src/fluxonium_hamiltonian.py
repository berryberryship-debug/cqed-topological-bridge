#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUXONIUM HAMILTONIAN — COMPATIBILITY LAYER
===========================================

Façade compatible avec fluxonium_model.py.

Conventions :
    EJ, EC, EL : GHz
    phi_ext    : phase réduite, en radians
                 phi_ext = 2*pi*Phi_ext/Phi0

Hamiltonien :
    H = 4 EC n² + EL/2 (phi - phi_ext)² - EJ cos(phi)

Important :
    Le solveur spectral calcule les niveaux et les états propres.
    Il ne déduit pas T1 ni T_phi à partir du Hamiltonien statique.

    T1_s et T_phi_s doivent être fournis explicitement pour obtenir
    les taux de cohérence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Optional

import numpy as np

from fluxonium_model import (
    FluxoniumHardwareParams,
    FluxoniumHamiltonianSolver,
    DecoherenceAnalyzer,
)


@dataclass(frozen=True)
class FluxoniumParams:
    """Paramètres du circuit et de la discrétisation."""

    EJ: float = 4.0
    EC: float = 1.0
    EL: float = 0.5
    phi_ext: float = pi

    n_grid: int = 500
    phi_bound: float = 5.0 * pi
    n_states: int = 6

    # Temps de cohérence fournis par mesure ou modèle externe.
    T1_s: Optional[float] = None
    T_phi_s: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("EJ", "EC", "EL"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} doit être fini et strictement positif.")

        if not isfinite(self.phi_ext):
            raise ValueError("phi_ext doit être fini.")

        if isinstance(self.n_grid, bool) or self.n_grid < 20:
            raise ValueError("n_grid doit être un entier >= 20.")

        if isinstance(self.n_states, bool):
            raise ValueError("n_states doit être un entier.")

        if self.n_states < 3 or self.n_states >= self.n_grid:
            raise ValueError("n_states doit être compris entre 3 et n_grid - 1.")

        if not isfinite(self.phi_bound) or self.phi_bound <= 0:
            raise ValueError("phi_bound doit être fini et strictement positif.")

        if self.T1_s is not None:
            if not isfinite(self.T1_s) or self.T1_s <= 0:
                raise ValueError("T1_s doit être fini et strictement positif.")

        if self.T_phi_s is not None:
            if np.isnan(self.T_phi_s) or self.T_phi_s <= 0:
                raise ValueError(
                    "T_phi_s doit être strictement positif. "
                    "Une valeur +inf est autorisée."
                )


class FluxoniumHamiltonian:
    """
    Interface de haut niveau pour le solveur Fluxonium.

    Méthodes principales :
        solve()
        f01()
        df01_dphi()
        df01_dphi_phi0()
        coherence_rates()
        matrix_elements()
        sweep_flux()
    """

    def __init__(self, params: FluxoniumParams):
        if not isinstance(params, FluxoniumParams):
            raise TypeError("params doit être une instance de FluxoniumParams.")

        self.params = params

        self.hardware = FluxoniumHardwareParams(
            E_J_GHz=params.EJ,
            E_C_GHz=params.EC,
            E_L_GHz=params.EL,
        )

        self.solver = FluxoniumHamiltonianSolver(
            self.hardware,
            n_grid=params.n_grid,
            phi_bound=params.phi_bound,
            n_states=params.n_states,
        )

        self._last_result = None

    def solve(self, phi_ext: Optional[float] = None):
        """
        Résout le Hamiltonien pour une phase externe donnée.

        phi_ext est exprimé en radians.
        """
        phase = self.params.phi_ext if phi_ext is None else float(phi_ext)

        if not isfinite(phase):
            raise ValueError("phi_ext doit être fini.")

        result = self.solver.solve(phi_ext=phase)
        self._last_result = result
        return result

    def energies(self, phi_ext: Optional[float] = None) -> np.ndarray:
        """Retourne les énergies propres en GHz."""
        result = self.solve(phi_ext)
        return np.asarray(result.energies_GHz, dtype=float)

    def f01(self, phi_ext: Optional[float] = None) -> float:
        """Fréquence de transition 0→1 en GHz."""
        energies = self.energies(phi_ext)

        if energies.size < 2:
            raise RuntimeError("Deux niveaux propres sont nécessaires.")

        return float(energies[1] - energies[0])

    def df01_dphi(
        self,
        phi_ext: Optional[float] = None,
        step: float = 1e-4,
    ) -> float:
        """
        Dérivée centrale de f01 par rapport à phi_ext.

        Unité :
            GHz/rad

        phi_ext est la phase réduite 2*pi*Phi_ext/Phi0.
        """
        if not isfinite(step) or step <= 0:
            raise ValueError("step doit être fini et strictement positif.")

        phase = self.params.phi_ext if phi_ext is None else float(phi_ext)

        if not isfinite(phase):
            raise ValueError("phi_ext doit être fini.")

        f_plus = self.f01(phase + step)
        f_minus = self.f01(phase - step)

        return float((f_plus - f_minus) / (2.0 * step))

    def df01_dphi_phi0(
        self,
        phi_ext: Optional[float] = None,
        step: float = 1e-4,
    ) -> float:
        """
        Dérivée par rapport au flux réduit Phi/Phi0.

        Unité :
            GHz/Phi0

        Puisque phi_ext = 2*pi*Phi/Phi0 :
            df/d(Phi/Phi0) = 2*pi * df/d(phi_ext)
        """
        return float(2.0 * pi * self.df01_dphi(phi_ext, step))

    def coherence_rates(self) -> dict:
        """
        Valide et expose les temps de cohérence fournis explicitement.

        Aucun T1 ni T_phi n'est inféré du Hamiltonien statique.
        """
        if self.params.T1_s is None:
            raise ValueError(
                "T1_s absent : fournir une valeur issue d'une mesure "
                "ou d'un modèle de dissipation."
            )

        if self.params.T_phi_s is None:
            raise ValueError(
                "T_phi_s absent : fournir une valeur issue d'une mesure "
                "ou d'un modèle de dissipation."
            )

        return DecoherenceAnalyzer.validate_coherence_times(
            T1_s=self.params.T1_s,
            T_phi_s=self.params.T_phi_s,
        )

    def matrix_elements(self) -> dict:
        """
        Retourne les éléments de matrice calculés par le solveur.

        La convention de normalisation et les unités sont celles
        documentées dans fluxonium_model.py.
        """
        result = self.solve()

        if not hasattr(result, "matrix_elements"):
            raise AttributeError(
                "Le résultat du solveur ne fournit pas matrix_elements."
            )

        return result.matrix_elements

    def sweep_flux(
        self,
        phi_ext_values: np.ndarray,
        n_states: Optional[int] = None,
    ) -> dict:
        """
        Effectue un balayage spectral en phase externe.

        Les phases sont exprimées en radians.
        """
        phases = np.asarray(phi_ext_values, dtype=float)

        if phases.ndim != 1 or phases.size == 0:
            raise ValueError("phi_ext_values doit être un tableau 1D non vide.")

        if not np.all(np.isfinite(phases)):
            raise ValueError("Toutes les phases doivent être finies.")

        count = self.params.n_states if n_states is None else int(n_states)

        if count < 3 or count >= self.params.n_grid:
            raise ValueError("n_states doit être compris entre 3 et n_grid - 1.")

        return self.solver.sweep_flux(
            phi_ext_values=phases,
            n_states=count,
        )


__all__ = [
    "FluxoniumParams",
    "FluxoniumHamiltonian",
]
