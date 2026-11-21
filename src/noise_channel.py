#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NOISE CHANNEL — FLUXONIUM
=========================

Canaux effectifs :
    - relaxation (amplitude damping) ;
    - déphasage pur ;
    - composition des canaux ;
    - représentation Pauli-twirled ;
    - conversion entre représentation PTM et probabilités de Pauli ;
    - contribution phénoménologique du bruit de flux.

Conventions :
    T1, T_phi, tau : secondes
    S_phi          : Phi0²/GHz
    df01_dphi      : GHz/Phi0

Attention :
    La contribution du bruit de flux est un modèle phénoménologique.
    Elle dépend de la convention spectrale adoptée pour S_phi.
    Elle n'est pas activée implicitement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, isinf, isnan, sqrt
from typing import Optional

import numpy as np


_EPS = 1e-12


def _require_finite_positive(name: str, value: float) -> float:
    """Valide une valeur réelle finie et strictement positive."""
    value = float(value)

    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} doit être fini et strictement positif.")

    return value


def _require_positive_or_inf(name: str, value: float) -> float:
    """Autorise +inf, mais rejette NaN, zéro et les valeurs négatives."""
    value = float(value)

    if isnan(value) or value <= 0.0:
        raise ValueError(
            f"{name} doit être strictement positif ; +inf est autorisé."
        )

    return value


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Valide et nettoie un vecteur de probabilités."""
    p = np.asarray(probabilities, dtype=float)

    if p.shape != (4,):
        raise ValueError("Le vecteur de probabilités doit avoir 4 éléments.")

    if not np.all(np.isfinite(p)):
        raise ValueError("Les probabilités doivent être finies.")

    if np.any(p < -_EPS) or np.any(p > 1.0 + _EPS):
        raise ValueError("Probabilité hors de l'intervalle [0, 1].")

    p = np.clip(p, 0.0, 1.0)

    total = float(np.sum(p))

    if not np.isclose(total, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(
            f"La somme des probabilités doit être égale à 1 ; somme={total}."
        )

    return p


def relaxation_dephasing_rates(
    T1_s: float,
    T_phi_s: float,
) -> dict:
    """
    Convertit les temps de cohérence en taux.

    gamma_1   = 1/T1
    gamma_phi = 1/T_phi
    gamma_2   = gamma_1/2 + gamma_phi

    T_phi = +inf correspond à l'absence de déphasage pur.
    """
    T1 = _require_finite_positive("T1_s", T1_s)
    T_phi = _require_positive_or_inf("T_phi_s", T_phi_s)

    gamma_1 = 1.0 / T1
    gamma_phi = 0.0 if isinf(T_phi) else 1.0 / T_phi
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


def amplitude_damping_pauli_twirling(gamma: float) -> np.ndarray:
    """
    Probabilités de la twirl de Pauli du canal d'amortissement
    d'amplitude.

    gamma = 1 - exp(-tau/T1)

    Ordre retourné :
        [I, X, Y, Z]
    """
    gamma = float(gamma)

    if not isfinite(gamma) or gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma doit appartenir à [0, 1].")

    root = sqrt(max(0.0, 1.0 - gamma))

    probabilities = np.array(
        [
            ((1.0 + root) ** 2) / 4.0,
            gamma / 4.0,
            gamma / 4.0,
            ((1.0 - root) ** 2) / 4.0,
        ],
        dtype=float,
    )

    return _validate_probabilities(probabilities)


def pure_dephasing_pauli_twirling(
    tau_s: float,
    T_phi_s: float,
) -> np.ndarray:
    """
    Probabilités du canal de déphasage pur.

    Ordre retourné :
        [I, X, Y, Z]

    Le canal est :
        rho_01 -> exp(-tau/T_phi) * rho_01
    """
    tau = _require_finite_positive("tau_s", tau_s)
    T_phi = _require_positive_or_inf("T_phi_s", T_phi_s)

    decay = 1.0 if isinf(T_phi) else exp(-tau / T_phi)

    p_z = 0.5 * (1.0 - decay)
    p_i = 1.0 - p_z

    return _validate_probabilities(
        np.array([p_i, 0.0, 0.0, p_z], dtype=float)
    )


def probabilities_to_ptm_diag(probabilities: np.ndarray) -> np.ndarray:
    """
    Convertit les probabilités de Pauli en diagonale PTM.

    Pour un canal Pauli :
        PTM = diag(1, lambda_x, lambda_y, lambda_z)
    """
    p = _validate_probabilities(probabilities)
    p_i, p_x, p_y, p_z = p

    return np.array(
        [
            1.0,
            p_i + p_x - p_y - p_z,
            p_i - p_x + p_y - p_z,
            p_i - p_x - p_y + p_z,
        ],
        dtype=float,
    )


def probs_from_ptm_diag(ptm_diag: np.ndarray) -> np.ndarray:
    """Reconstruit les probabilités de Pauli depuis une diagonale PTM."""
    lam = np.asarray(ptm_diag, dtype=float)

    if lam.shape != (4,):
        raise ValueError("La diagonale PTM doit avoir 4 éléments.")

    if not np.all(np.isfinite(lam)):
        raise ValueError("La diagonale PTM doit être finie.")

    if not np.isclose(lam[0], 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Le premier élément PTM doit être égal à 1.")

    _, lx, ly, lz = lam

    probabilities = np.array(
        [
            (1.0 + lx + ly + lz) / 4.0,
            (1.0 + lx - ly - lz) / 4.0,
            (1.0 - lx + ly - lz) / 4.0,
            (1.0 - lx - ly + lz) / 4.0,
        ],
        dtype=float,
    )

    return _validate_probabilities(probabilities)


def compose_pauli_channels(
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
) -> np.ndarray:
    """
    Compose deux canaux Pauli indépendants.

    La composition est effectuée par multiplication des diagonales PTM.
    """
    p_a = _validate_probabilities(probabilities_a)
    p_b = _validate_probabilities(probabilities_b)

    ptm_a = probabilities_to_ptm_diag(p_a)
    ptm_b = probabilities_to_ptm_diag(p_b)

    return probs_from_ptm_diag(ptm_a * ptm_b)


def flux_noise_dephasing_rate(
    df01_dphi_GHz_per_Phi0: float,
    S_phi_Phi0_squared_per_GHz: float,
) -> float:
    """
    Estimation phénoménologique du taux de déphasage dû au bruit de flux.

    Unités :
        df01_dphi : GHz/Phi0
        S_phi     : Phi0²/GHz
        résultat  : GHz

    Convention utilisée :
        gamma_phi_GHz = 0.5 * (df01/dPhi)^2 * S_phi

    Cette relation dépend de la convention de densité spectrale.
    Elle ne constitue pas une formule universelle de décohérence.
    """
    derivative = _require_finite_positive(
        "df01_dphi_GHz_per_Phi0",
        abs(float(df01_dphi_GHz_per_Phi0)),
    )

    spectral_density = _require_finite_positive(
        "S_phi_Phi0_squared_per_GHz",
        S_phi_Phi0_squared_per_GHz,
    )

    return float(0.5 * derivative**2 * spectral_density)


@dataclass(frozen=True)
class ChannelConfig:
    """Configuration d'un canal effectif sur une durée tau."""

    T1: float
    T_phi: float
    tau: float

    df01_dphi: Optional[float] = None
    S_phi: Optional[float] = None
    use_flux_noise: bool = False

    def __post_init__(self) -> None:
        _require_finite_positive("T1", self.T1)
        _require_positive_or_inf("T_phi", self.T_phi)
        _require_finite_positive("tau", self.tau)

        if self.use_flux_noise:
            if self.df01_dphi is None:
                raise ValueError(
                    "df01_dphi est requis lorsque use_flux_noise=True."
                )

            if self.S_phi is None:
                raise ValueError(
                    "S_phi est requis lorsque use_flux_noise=True."
                )

            _require_finite_positive(
                "abs(df01_dphi)",
                abs(float(self.df01_dphi)),
            )
            _require_finite_positive("S_phi", self.S_phi)

        elif self.df01_dphi is not None:
            if not isfinite(float(self.df01_dphi)):
                raise ValueError("df01_dphi doit être fini.")

        if self.S_phi is not None:
            _require_finite_positive("S_phi", self.S_phi)

    def to_dict(self) -> dict:
        """Retourne la configuration sous forme de dictionnaire."""
        return asdict(self)


def build_pauli_channel(config: ChannelConfig) -> dict:
    """
    Construit le canal effectif relaxation + déphasage.

    Si use_flux_noise=True, le taux phénoménologique du bruit de flux
    est ajouté au taux de déphasage pur.

    Retourne :
        - probabilités I, X, Y, Z ;
        - diagonale PTM ;
        - probabilités des sous-canaux ;
        - taux effectifs utilisés.
    """
    if not isinstance(config, ChannelConfig):
        raise TypeError("config doit être une instance de ChannelConfig.")

    gamma_1 = 1.0 / config.T1
    gamma_phi = 0.0 if isinf(config.T_phi) else 1.0 / config.T_phi

    gamma_flux = 0.0

    if config.use_flux_noise:
        gamma_flux_GHz = flux_noise_dephasing_rate(
            df01_dphi_GHz_per_Phi0=config.df01_dphi,
            S_phi_Phi0_squared_per_GHz=config.S_phi,
        )

        # Conversion GHz -> s^-1.
        gamma_flux = gamma_flux_GHz * 1e9

    T_phi_effective = (
        float("inf")
        if gamma_phi + gamma_flux == 0.0
        else 1.0 / (gamma_phi + gamma_flux)
    )

    gamma_relaxation = -np.expm1(-config.tau / config.T1)

    p_relaxation = amplitude_damping_pauli_twirling(
        gamma_relaxation
    )

    p_dephasing = pure_dephasing_pauli_twirling(
        tau_s=config.tau,
        T_phi_s=T_phi_effective,
    )

    probabilities = compose_pauli_channels(
        p_relaxation,
        p_dephasing,
    )

    ptm_diag = probabilities_to_ptm_diag(probabilities)

    return {
        "probabilities": {
            "I": float(probabilities[0]),
            "X": float(probabilities[1]),
            "Y": float(probabilities[2]),
            "Z": float(probabilities[3]),
        },
        "probability_array": probabilities,
        "ptm_diagonal": ptm_diag,
        "relaxation_probabilities": p_relaxation,
        "dephasing_probabilities": p_dephasing,
        "rates": {
            "gamma_1_per_s": float(gamma_1),
            "gamma_phi_per_s": float(gamma_phi),
            "gamma_flux_per_s": float(gamma_flux),
            "gamma_phi_effective_per_s": float(
                gamma_phi + gamma_flux
            ),
        },
        "T_phi_effective_s": float(T_phi_effective),
        "tau_s": float(config.tau),
    }


def effective_error_rate(
    channel: dict,
    convention: str = "1-pI",
) -> float:
    """
    Calcule un taux d'erreur effectif à partir du canal.

    Conventions :
        "1-pI"  : probabilité totale d'une erreur Pauli non identité.
        "pX+pY" : contribution des composantes X et Y.
        "pZ"    : contribution de la composante Z.
    """
    if "probabilities" not in channel:
        raise ValueError("Le canal ne contient pas de probabilités.")

    p = channel["probabilities"]

    p_i = float(p["I"])
    p_x = float(p["X"])
    p_y = float(p["Y"])
    p_z = float(p["Z"])

    if convention == "1-pI":
        result = 1.0 - p_i
    elif convention == "pX+pY":
        result = p_x + p_y
    elif convention == "pZ":
        result = p_z
    else:
        raise ValueError(
            "Convention inconnue. Utiliser '1-pI', 'pX+pY' ou 'pZ'."
        )

    return float(result)


__all__ = [
    "ChannelConfig",
    "relaxation_dephasing_rates",
    "amplitude_damping_pauli_twirling",
    "pure_dephasing_pauli_twirling",
    "probabilities_to_ptm_diag",
    "probs_from_ptm_diag",
    "compose_pauli_channels",
    "flux_noise_dephasing_rate",
    "build_pauli_channel",
    "effective_error_rate",
]
