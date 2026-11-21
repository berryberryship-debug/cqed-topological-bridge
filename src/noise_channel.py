
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOISE CHANNEL — Pauli twirling et composition PTM.

Convention :
- T1, T_phi, tau : secondes.
- Pauli probabilities : sans dimension.
- df01_dphi : GHz / Phi0.
- S_phi : Phi0² / GHz, selon la convention spectrale adoptée ici.

IMPORTANT
---------
Le bruit de flux n'est pas converti automatiquement en T_phi.
Une densité spectrale doit être accompagnée de sa convention exacte
(unilatérale/bilatérale, fréquence angulaire ou ordinaire, bande).
Le paramètre use_flux_noise active explicitement l'approximation
phénoménologique documentée dans flux_noise_to_dephasing_rate().
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

PAULI_KEYS = ("I", "X", "Y", "Z")


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} doit être une valeur finie.")


def _validate_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    missing = set(PAULI_KEYS) - set(probs)
    if missing:
        raise ValueError(f"Probabilités Pauli manquantes : {sorted(missing)}")

    result = {k: float(probs[k]) for k in PAULI_KEYS}

    for key, value in result.items():
        _require_finite(key, value)
        if value < -1e-12 or value > 1.0 + 1e-12:
            raise ValueError(f"{key}={value} hors de [0, 1].")

    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"Canal Pauli non normalisé : somme={total:.16g}"
        )

    return {k: max(0.0, min(1.0, v)) for k, v in result.items()}


# ---------------------------------------------------------------------
# Canaux élémentaires
# ---------------------------------------------------------------------

def amplitude_damping_twirl(gamma: float) -> Dict[str, float]:
    """
    Twirling Pauli exact du canal amplitude damping standard.

    Kraus :
        K0 = diag(1, sqrt(1-gamma))
        K1 = sqrt(gamma) |0><1|

    Probabilités exactes :
        pI = (1 + sqrt(1-gamma))² / 4
        pX = pY = gamma / 4
        pZ = (1 - sqrt(1-gamma))² / 4
    """
    gamma = float(gamma)
    _require_finite("gamma", gamma)

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma doit appartenir à [0, 1].")

    root = math.sqrt(1.0 - gamma)

    return {
        "I": ((1.0 + root) ** 2) / 4.0,
        "X": gamma / 4.0,
        "Y": gamma / 4.0,
        "Z": ((1.0 - root) ** 2) / 4.0,
    }


def dephasing_twirl(lambda_z: float) -> Dict[str, float]:
    """Canal de déphasage pur : identité avec probabilité 1-lambda_z,
    Pauli Z avec probabilité lambda_z.
    """
    lambda_z = float(lambda_z)
    _require_finite("lambda_z", lambda_z)

    if not 0.0 <= lambda_z <= 1.0:
        raise ValueError("lambda_z doit appartenir à [0, 1].")

    return {
        "I": 1.0 - lambda_z,
        "X": 0.0,
        "Y": 0.0,
        "Z": lambda_z,
    }


def relaxation_dephasing_rates(
    T1: float,
    T_phi: float,
    tau: float,
) -> Tuple[float, float]:
    """Convertit les temps en paramètres de canaux."""
    for name, value in (("T1", T1), ("T_phi", T_phi), ("tau", tau)):
        _require_finite(name, value)
        if value <= 0.0:
            raise ValueError(f"{name} doit être strictement positif.")

    gamma_1 = -math.expm1(-tau / T1)
    lambda_z = -0.5 * math.expm1(-tau / T_phi)

    return gamma_1, lambda_z


# ---------------------------------------------------------------------
# Pauli transfer matrix
# ---------------------------------------------------------------------

def pauli_transfer_matrix_diag(
    probs: Dict[str, float],
) -> Tuple[float, float, float]:
    """Diagonale non triviale du PTM d'un canal Pauli."""
    p = _validate_probabilities(probs)

    return (
        p["I"] + p["X"] - p["Y"] - p["Z"],
        p["I"] - p["X"] + p["Y"] - p["Z"],
        p["I"] - p["X"] - p["Y"] + p["Z"],
    )


def probs_from_ptm_diag(
    Rxx: float,
    Ryy: float,
    Rzz: float,
) -> Dict[str, float]:
    """Inverse du PTM diagonal, sans écrêtage silencieux."""
    for name, value in (
        ("Rxx", Rxx), ("Ryy", Ryy), ("Rzz", Rzz)
    ):
        _require_finite(name, value)

    p = {
        "I": (1.0 + Rxx + Ryy + Rzz) / 4.0,
        "X": (1.0 + Rxx - Ryy - Rzz) / 4.0,
        "Y": (1.0 - Rxx + Ryy - Rzz) / 4.0,
        "Z": (1.0 - Rxx - Ryy + Rzz) / 4.0,
    }

    # Les petites erreurs numériques sont tolérées.
    # Une probabilité négative significative provoque une erreur.
    for key, value in p.items():
        if value < -1e-10 or value > 1.0 + 1e-10:
            raise ValueError(
                f"PTM non physique : p_{key}={value:.12g}"
            )

    p = {k: max(0.0, min(1.0, v)) for k, v in p.items()}
    total = sum(p.values())

    if not math.isclose(total, 1.0, abs_tol=1e-10):
        raise ValueError(
            f"Reconstruction PTM non normalisée : somme={total}"
        )

    return p


def compose_pauli_channels(
    channel_A: Dict[str, float],
    channel_B: Dict[str, float],
) -> Dict[str, float]:
    """Composition C = B après A, par multiplication des PTM."""
    ax, ay, az = pauli_transfer_matrix_diag(channel_A)
    bx, by, bz = pauli_transfer_matrix_diag(channel_B)

    return probs_from_ptm_diag(
        ax * bx,
        ay * by,
        az * bz,
    )


def pauli_twirl_channel(
    T1: float,
    T_phi: float,
    tau: float,
) -> Dict[str, float]:
    """Composition du twirling amplitude damping et déphasage pur."""
    gamma_1, lambda_z = relaxation_dephasing_rates(
        T1, T_phi, tau
    )

    channel_ad = amplitude_damping_twirl(gamma_1)
    channel_phi = dephasing_twirl(lambda_z)

    return compose_pauli_channels(channel_ad, channel_phi)


# ---------------------------------------------------------------------
# Bruit de flux — approximation explicite
# ---------------------------------------------------------------------

def flux_noise_to_dephasing_rate(
    df01_dphi: float,
    S_phi: float,
) -> float:
    """
    Approximation phénoménologique :

        gamma_phi_GHz = 0.5 * (df01/dPhi)^2 * S_phi

    Unités déclarées :
        df01_dphi : GHz / Phi0
        S_phi     : Phi0² / GHz
        résultat  : GHz

    Conversion numérique :
        gamma_phi_s = gamma_phi_GHz * 1e9

    Cette convention ne remplace pas une dérivation à partir d'une
    densité spectrale précisément définie. Le facteur dépend notamment
    de la convention unilatérale/bilatérale et de la variable spectrale.
    """
    df01_dphi = float(df01_dphi)
    S_phi = float(S_phi)

    _require_finite("df01_dphi", df01_dphi)
    _require_finite("S_phi", S_phi)

    if S_phi < 0.0:
        raise ValueError("S_phi ne peut pas être négatif.")

    gamma_phi_GHz = 0.5 * (df01_dphi ** 2) * S_phi
    return gamma_phi_GHz * 1e9


def dephasing_rate_to_T_phi(gamma_phi_s: float) -> float:
    """Convertit un taux en s^-1 en temps T_phi en secondes."""
    gamma_phi_s = float(gamma_phi_s)
    _require_finite("gamma_phi_s", gamma_phi_s)

    if gamma_phi_s < 0.0:
        raise ValueError("Le taux ne peut pas être négatif.")
    if gamma_phi_s == 0.0:
        return float("inf")

    return 1.0 / gamma_phi_s


# ---------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------

def apply_readout_error(
    pauli_probs: Dict[str, float],
    p_readout: float,
) -> Dict[str, float]:
    """Conserve le canal quantique ; l'erreur de lecture est séparée."""
    p_readout = float(p_readout)
    _require_finite("p_readout", p_readout)

    if not 0.0 <= p_readout <= 1.0:
        raise ValueError("p_readout doit appartenir à [0, 1].")

    if p_readout > 0.0:
        LOGGER.info(
            "p_readout=%.6g est une erreur classique séparée du canal.",
            p_readout,
        )

    return _validate_probabilities(pauli_probs)


# ---------------------------------------------------------------------
# Configuration et pipeline
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelConfig:
    T1: float
    T_phi: float
    tau: float
    df01_dphi: float = 0.0
    S_phi: float = 0.0
    p_readout: float = 0.0
    use_flux_noise: bool = False

    def __post_init__(self):
        for name in ("T1", "T_phi", "tau"):
            value = float(getattr(self, name))
            _require_finite(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} doit être strictement positif.")

        _require_finite("df01_dphi", self.df01_dphi)
        _require_finite("S_phi", self.S_phi)
        _require_finite("p_readout", self.p_readout)

        if self.S_phi < 0.0:
            raise ValueError("S_phi ne peut pas être négatif.")
        if not 0.0 <= self.p_readout <= 1.0:
            raise ValueError("p_readout doit appartenir à [0, 1].")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_pauli_channel(
    config: ChannelConfig,
) -> Dict[str, float]:
    """Construit le canal Pauli à partir de la configuration."""
    T_phi = config.T_phi

    if config.use_flux_noise:
        if config.S_phi == 0.0:
            raise ValueError(
                "use_flux_noise=True exige S_phi > 0."
            )
        gamma_phi_s = flux_noise_to_dephasing_rate(
            config.df01_dphi,
            config.S_phi,
        )
        T_phi = dephasing_rate_to_T_phi(gamma_phi_s)

    channel = pauli_twirl_channel(
        T1=config.T1,
        T_phi=T_phi,
        tau=config.tau,
    )

    return apply_readout_error(channel, config.p_readout)


def effective_error_rate(
    pauli_probs: Dict[str, float],
) -> float:
    """Probabilité totale d'une erreur non identité."""
    p = _validate_probabilities(pauli_probs)
    return 1.0 - p["I"]


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def run_all_tests() -> None:
    # Normalisation amplitude damping.
    for gamma in (0.0, 1e-9, 0.1, 0.5, 1.0):
        p = amplitude_damping_twirl(gamma)
        assert abs(sum(p.values()) - 1.0) < 1e-12

    # Limite faible gamma : pZ est d'ordre gamma².
    gamma = 1e-5
    p = amplitude_damping_twirl(gamma)
    assert abs(p["X"] - gamma / 4.0) < 1e-14
    assert abs(p["Y"] - gamma / 4.0) < 1e-14
    assert p["Z"] < 1e-9

    # Canal de déphasage.
    p = dephasing_twirl(0.05)
    assert abs(sum(p.values()) - 1.0) < 1e-12

    # Composition et associativité.
    A = amplitude_damping_twirl(0.1)
    B = dephasing_twirl(0.05)
    C = amplitude_damping_twirl(0.2)

    left = compose_pauli_channels(
        compose_pauli_channels(A, B), C
    )
    right = compose_pauli_channels(
        A, compose_pauli_channels(B, C)
    )

    for key in PAULI_KEYS:
        assert abs(left[key] - right[key]) < 1e-12

    # Inversion PTM.
    original = {"I": 0.7, "X": 0.1, "Y": 0.1, "Z": 0.1}
    reconstructed = probs_from_ptm_diag(
        *pauli_transfer_matrix_diag(original)
    )

    for key in PAULI_KEYS:
        assert abs(original[key] - reconstructed[key]) < 1e-12

    # Temps très court.
    channel = pauli_twirl_channel(
        T1=1e-3,
        T_phi=1e-3,
        tau=1e-15,
    )
    assert channel["I"] > 0.999999
    assert effective_error_rate(channel) < 1e-6

    # Configuration.
    config = ChannelConfig(
        T1=7e-6,
        T_phi=1e-3,
        tau=20e-9,
    )
    assert abs(sum(build_pauli_channel(config).values()) - 1.0) < 1e-10

    print("Tous les tests noise_channel.py sont passés.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canal de bruit et Pauli twirling."
    )
    parser.add_argument("--T1", type=float, default=7.19e-6)
    parser.add_argument("--T_phi", type=float, default=45960e-6)
    parser.add_argument("--tau", type=float, default=20e-9)
    parser.add_argument("--df01_dphi", type=float, default=0.0)
    parser.add_argument("--S_phi", type=float, default=0.0)
    parser.add_argument("--use_flux_noise", action="store_true")
    parser.add_argument("--p_readout", type=float, default=0.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test:
        run_all_tests()
        return

    config = ChannelConfig(
        T1=args.T1,
        T_phi=args.T_phi,
        tau=args.tau,
        df01_dphi=args.df01_dphi,
        S_phi=args.S_phi,
        p_readout=args.p_readout,
        use_flux_noise=args.use_flux_noise,
    )

    channel = build_pauli_channel(config)
    result = {
        **channel,
        "effective_error_rate": effective_error_rate(channel),
        "configuration": config.to_dict(),
    }

    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Résultat sauvegardé : %s", output_path)


if __name__ == "__main__":
    main()
