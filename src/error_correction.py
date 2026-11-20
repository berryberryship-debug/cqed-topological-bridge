#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module : error_correction.py
Simulation Monte Carlo et analytique de correction d'erreurs quantiques (QEC)
adaptée aux architectures à bruit biaisé (Fluxonium protégé / Kerr-Cat) :
  - Canal de bruit asymétrique (Pauli biaisé : p_Z >> p_X, p_Y)
  - Décodeur de code de répétition de phase à distance d (syndrome X_i X_{i+1})
  - Simulation Monte Carlo de seuil et d'échelle de suppression d'erreurs logiques
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class BiasedNoiseChannel:
    """Modèle de canal de Pauli asymétrique avec biais eta = p_Z / (p_X + p_Y)."""
    p_total: float = 0.02              # Probabilité totale d'erreur par cycle
    bias_eta: float = 100.0            # Biais de phase eta = p_Z / (p_X + p_Y)

    @property
    def probabilities(self) -> Tuple[float, float, float, float]:
        """Retourne (p_I, p_X, p_Y, p_Z)."""
        p_xy = self.p_total / (1.0 + self.bias_eta)
        p_Z = self.p_total - p_xy
        p_X = 0.5 * p_xy
        p_Y = 0.5 * p_xy
        p_I = max(0.0, 1.0 - self.p_total)
        return p_I, p_X, p_Y, p_Z


class PhaseFlipRepetitionCode:
    """
    Code de répétition de phase à distance impaire d :
      - Encode |0_L> = |+>^{otimes d}, |1_L> = |->^{otimes d}
      - Stabilisateurs : S_i = X_i X_{i+1} pour i = 0, ..., d-2
      - Corrige jusqu'à t = (d - 1) // 2 erreurs Z simultanées
      - Toute erreur X ou Y non détectée provoque un basculement logique résiduel
    """

    def __init__(self, distance: int):
        if distance < 3 or distance % 2 == 0:
            raise ValueError("La distance d du code de répétition doit être impaire et >= 3.")
        self.d = distance
        self.t_correctable = (distance - 1) // 2

    def measure_syndrome(self, z_errors: np.ndarray) -> np.ndarray:
        """Calcule le vecteur syndrome s_i = z_i XOR z_{i+1} de longueur d - 1."""
        return np.bitwise_xor(z_errors[:-1], z_errors[1:])

    def decode_minimum_weight(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Décodeur exact de poids minimal en 1D :
        Compare les deux configurations d'erreurs Z compatibles avec le syndrome
        et sélectionne celle impliquant le plus petit nombre de qubits physiques.
        """
        candidate_0 = np.zeros(self.d, dtype=np.int8)
        for i in range(self.d - 1):
            candidate_0[i + 1] = candidate_0[i] ^ syndrome[i]

        candidate_1 = 1 - candidate_0
        if np.sum(candidate_0) <= np.sum(candidate_1):
            return candidate_0
        return candidate_1

    def run_monte_carlo(
        self,
        channel: BiasedNoiseChannel,
        n_shots: int = 50_000,
        seed: int = 42
    ) -> Dict[str, float]:
        """
        Exécute n_shots cycles d'injection de bruit et de correction :
          - Échec logique Z si la parité globale de (z_err XOR z_corr) vaut 1
          - Échec logique X si un nombre impair d'erreurs X/Y survient
        """
        rng = np.random.default_rng(seed)
        p_I, p_X, p_Y, p_Z = channel.probabilities

        # Tirage vectorisé des erreurs Pauli : 0=I, 1=X, 2=Y, 3=Z
        draws = rng.choice(
            4,
            size=(n_shots, self.d),
            p=[p_I, p_X, p_Y, p_Z]
        )

        # Décomposition en composantes X (codes 1 et 2) et Z (codes 2 et 3)
        x_errors = ((draws == 1) | (draws == 2)).astype(np.int8)
        z_errors = ((draws == 2) | (draws == 3)).astype(np.int8)

        logical_z_failures = 0
        logical_x_failures = 0

        for shot in range(n_shots):
            z_vec = z_errors[shot]
            syn = self.measure_syndrome(z_vec)
            z_corr = self.decode_minimum_weight(syn)
            residual_z = np.bitwise_xor(z_vec, z_corr)

            if np.sum(residual_z) % 2 == 1:
                logical_z_failures += 1
            if np.sum(x_errors[shot]) % 2 == 1:
                logical_x_failures += 1

        p_L_Z = logical_z_failures / n_shots
        p_L_X = logical_x_failures / n_shots
        p_L_tot = 1.0 - (1.0 - p_L_Z) * (1.0 - p_L_X)

        return {
            "distance": self.d,
            "n_shots": n_shots,
            "p_phys_total": channel.p_total,
            "bias_eta": channel.bias_eta,
            "p_logical_Z": p_L_Z,
            "p_logical_X": p_L_X,
            "p_logical_total": p_L_tot
        }

    def analytical_logical_error(self, channel: BiasedNoiseChannel) -> float:
        """Formule combinatoire exacte du taux d'erreur logique Z + X."""
        _, p_X, p_Y, p_Z = channel.probabilities
        p_z_eff = p_Z + p_Y
        p_x_eff = p_X + p_Y

        # Probabilité d'avoir au moins t + 1 erreurs Z
        p_L_Z = sum(
            math.comb(self.d, k) * (p_z_eff ** k) * ((1.0 - p_z_eff) ** (self.d - k))
            for k in range(self.t_correctable + 1, self.d + 1)
        )
        # Probabilité d'avoir un nombre impair d'erreurs X non corrigées
        p_L_X = 0.5 * (1.0 - ((1.0 - 2.0 * p_x_eff) ** self.d))
        return 1.0 - (1.0 - p_L_Z) * (1.0 - p_L_X)
