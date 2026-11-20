"""
error_correction.py

Estimation du taux d'erreur logique pour codes correcteurs
à partir des paramètres physiques mesurés (T1, T2, fidélité de porte).

Modèles disponibles :
- Code de surface : p_L ≈ A * (p / p_th)^((d+1)/2)
- Code de répétition biaisé : somme binomiale exacte

Architecture découplée
----------------------
Ce module ne prend que des NOMBRES (T1, T2, fidélité de porte,
durée de porte). Il ne dépend d'aucun autre module du projet
(fluxonium_hamiltonian, noise_channel, etc.). Le pipeline
d'assemblage (run_pipeline.py) fait le collage.

Auteur : [Votre nom]
Licence : MIT
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np


# ----------------------------------------------------------------------
# Paramètres physiques
# ----------------------------------------------------------------------

@dataclass
class PhysicalErrorRates:
    """Taux d'erreur physiques extraits de la simulation."""

    t1: float  # secondes
    t2: float  # secondes
    gate_fidelity: float  # entre 0 et 1
    gate_time: float  # secondes

    def __post_init__(self):
        if self.t1 <= 0 or self.t2 <= 0:
            raise ValueError("T1 et T2 doivent être positifs.")
        if not 0 < self.gate_fidelity <= 1:
            raise ValueError("gate_fidelity doit être dans (0, 1].")
        if self.gate_time <= 0:
            raise ValueError("gate_time doit être positif.")

    @property
    def relaxation_error(self) -> float:
        """Probabilité d'erreur de relaxation pendant une porte."""
        return 1.0 - np.exp(-self.gate_time / self.t1)

    @property
    def dephasing_error(self) -> float:
        """Probabilité d'erreur de déphasage pendant une porte."""
        return 1.0 - np.exp(-self.gate_time / self.t2)

    @property
    def gate_error(self) -> float:
        """Erreur totale par porte = 1 - fidélité."""
        return 1.0 - self.gate_fidelity

    @property
    def total_physical_error(self) -> float:
        """
        Somme des contributions (approximation indépendante).

        Note : cette somme suppose l'indépendance des trois canaux,
        ce qui est une construction du modèle, pas une propriété
        fondamentale.
        """
        return self.relaxation_error + self.dephasing_error + self.gate_error


# ----------------------------------------------------------------------
# Codes correcteurs
# ----------------------------------------------------------------------

def logical_error_rate_surface_code(
    physical_error: float,
    code_distance: int,
    threshold: float = 0.01,
    prefactor: float = 0.1,
) -> float:
    """
    Estime le taux d'erreur logique pour un code de surface.

    p_L ≈ A * (p / p_th)^((d+1)/2)

    Valide seulement si p < p_th.
    """
    if code_distance < 1:
        raise ValueError("code_distance doit être au moins 1.")
    if physical_error >= threshold:
        return 1.0  # au-dessus du seuil, pas de correction
    exponent = (code_distance + 1) / 2.0
    return prefactor * (physical_error / threshold) ** exponent


def logical_error_rate_repetition_code(
    physical_error: float,
    n_repetitions: int,
) -> float:
    """
    Taux d'erreur logique pour un code de répétition simple.

    p_L = somme des k impairs C(n,k) p^k (1-p)^(n-k)
    Approximation dominante : premier terme impair.
    """
    if n_repetitions < 1:
        raise ValueError("n_repetitions doit être au moins 1.")
    p_L = (
        n_repetitions
        * physical_error
        * (1.0 - physical_error) ** (n_repetitions - 1)
    )
    return float(p_L)


def error_budget(
    rates: PhysicalErrorRates,
    code_distance: int = 3,
    code_type: str = "surface",
) -> Dict[str, float]:
    """
    Produit un budget d'erreur complet.

    Retourne un dictionnaire avec toutes les contributions.
    """
    p_phys = rates.total_physical_error

    if code_type == "surface":
        p_logical = logical_error_rate_surface_code(
            p_phys, code_distance
        )
    elif code_type == "repetition":
        p_logical = logical_error_rate_repetition_code(
            p_phys, code_distance
        )
    else:
        raise ValueError(
            "code_type doit être 'surface' ou 'repetition'."
        )

    return {
        "relaxation_error": rates.relaxation_error,
        "dephasing_error": rates.dephasing_error,
        "gate_error": rates.gate_error,
        "total_physical_error": p_phys,
        "code_distance": code_distance,
        "code_type": code_type,
        "logical_error_rate": p_logical,
        "below_threshold": p_phys < 0.01,
    }


# ----------------------------------------------------------------------
# Pipeline complet (découplé)
# ----------------------------------------------------------------------

def run_full_pipeline(
    T1: float,
    T2: float,
    gate_fidelity: float,
    gate_time: float,
    code_distance: int = 3,
    code_type: str = "surface",
) -> Dict[str, float]:
    """
    Pipeline complet de budget d'erreur.

    Prend des NOMBRES (T1, T2, fidélité de porte, durée de porte),
    jamais un objet modèle. Respecte l'architecture découplée.

    Retourne un dictionnaire avec toutes les contributions.
    """
    rates = PhysicalErrorRates(
        t1=T1,
        t2=T2,
        gate_fidelity=gate_fidelity,
        gate_time=gate_time,
    )
    return error_budget(
        rates, code_distance=code_distance, code_type=code_type
    )


# ----------------------------------------------------------------------
# Point d'entrée
# ----------------------------------------------------------------------

if __name__ == "__main__":
    budget = run_full_pipeline(
        T1=7.19e-6,
        T2=14.38e-6,
        gate_fidelity=0.99,
        gate_time=20e-9,
        code_distance=3,
        code_type="surface",
    )
    print(json.dumps(budget, indent=2))
