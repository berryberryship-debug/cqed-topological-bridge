"""
noise_channel.py

Canal de bruit et réduction Pauli-twirled.

Rôle dans le pont
-----------------
C'est la couche II-III de l'architecture :

    Couche I (physique) : T1, T_phi, d f01/d Phi, S_Phi(omega)
                ↓
    Couche II (canal)  : Gamma_1, Gamma_phi → canal E_tau
                ↓
    Couche III (QEC)   : Pauli twirling → p_X, p_Y, p_Z

Ce module est DÉCOUPLÉ : il ne prend que des nombres (T1, T_phi,
tau, d f01/d Phi, S_Phi), jamais un objet FluxoniumModel.

Modèle physique
---------------
1. Amplitude damping (relaxation T1) :
   canal de Kraus, twirlé en canal de Pauli :
       p_X = p_Y = gamma_1 / 4
       p_Z = gamma_1 / 2
       p_I = 1 - gamma_1
   avec gamma_1 = 1 - exp(-tau / T1).

2. Déphasage pur (T_phi) :
   canal de Pauli pur Z :
       p_Z = (1 - exp(-tau / T_phi)) / 2
       p_X = p_Y = 0

3. Composition exacte des deux canaux de Pauli via la matrice
   de transfert de Pauli (PTM), pas une approximation additive.

4. Bruit de flux : Gamma_phi = 0.5 * (d f01/d Phi)² * S_Phi
   (approximation de bruit blanc ; hook pour 1/f documenté).

Auteur : [Votre nom]
Licence : MIT
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------

def clip01(x: float) -> float:
    """Contraint une valeur dans [0, 1]."""
    return float(np.clip(x, 0.0, 1.0))


def normalize_pauli_probs(
    probs: Dict[str, float],
) -> Dict[str, float]:
    """
    Normalise un dictionnaire de probabilités Pauli
    pour que la somme fasse exactement 1.
    """
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-6:
        LOGGER.warning(
            "Somme des probabilités = %.6f, normalisation appliquée.",
            total,
        )
    return {k: v / total for k, v in probs.items()}


# ----------------------------------------------------------------------
# Canaux élémentaires
# ----------------------------------------------------------------------

def amplitude_damping_twirl(gamma: float) -> Dict[str, float]:
    """
    Pauli twirl exact du canal d'amplitude damping.

    Pour un canal AD de paramètre gamma (probabilité de saut),
    le twirl de Pauli donne :

        p_I = 1 - gamma
        p_X = p_Y = gamma / 4
        p_Z = gamma / 2

    Vérification : p_I + p_X + p_Y + p_Z = 1.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma doit être dans [0, 1].")
    return {
        "I": 1.0 - gamma,
        "X": gamma / 4.0,
        "Y": gamma / 4.0,
        "Z": gamma / 2.0,
    }


def dephasing_twirl(lambda_z: float) -> Dict[str, float]:
    """
    Canal de déphasage pur (Pauli Z uniquement).

    p_Z = lambda_z, p_X = p_Y = 0, p_I = 1 - lambda_z.

    Pour un déphasage avec temps T_phi et durée tau :
        lambda_z = (1 - exp(-tau / T_phi)) / 2
    """
    if not 0.0 <= lambda_z <= 1.0:
        raise ValueError("lambda_z doit être dans [0, 1].")
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
    """
    Convertit (T1, T_phi, tau) en (gamma_1, lambda_z).

    gamma_1  = 1 - exp(-tau / T1)
    lambda_z = (1 - exp(-tau / T_phi)) / 2
    """
    if T1 <= 0 or T_phi <= 0 or tau <= 0:
        raise ValueError("T1, T_phi et tau doivent être positifs.")
    gamma_1 = 1.0 - np.exp(-tau / T1)
    lambda_z = (1.0 - np.exp(-tau / T_phi)) / 2.0
    return float(gamma_1), float(lambda_z)


# ----------------------------------------------------------------------
# Composition exacte via PTM
# ----------------------------------------------------------------------

def pauli_transfer_matrix_diag(
    probs: Dict[str, float],
) -> Tuple[float, float, float]:
    """
    Retourne les 3 diagonales non triviales du PTM d'un canal de Pauli.

    R_XX = pI + pX - pY - pZ
    R_YY = pI - pX + pY - pZ
    R_ZZ = pI - pX - pY + pZ
    """
    pI = probs["I"]
    pX = probs["X"]
    pY = probs["Y"]
    pZ = probs["Z"]
    return (
        pI + pX - pY - pZ,  # R_XX
        pI - pX + pY - pZ,  # R_YY
        pI - pX - pY + pZ,  # R_ZZ
    )


def probs_from_ptm_diag(
    Rxx: float,
    Ryy: float,
    Rzz: float,
) -> Dict[str, float]:
    """
    Inverse du PTM : reconstruit les probabilités Pauli
    à partir des 3 diagonales.
    """
    pI = (1.0 + Rxx + Ryy + Rzz) / 4.0
    pX = (1.0 + Rxx - Ryy - Rzz) / 4.0
    pY = (1.0 - Rxx + Ryy - Rzz) / 4.0
    pZ = (1.0 - Rxx - Ryy + Rzz) / 4.0
    return {
        "I": clip01(pI),
        "X": clip01(pX),
        "Y": clip01(pY),
        "Z": clip01(pZ),
    }


def compose_pauli_channels(
    channel_A: Dict[str, float],
    channel_B: Dict[str, float],
) -> Dict[str, float]:
    """
    Compose exactement deux canaux de Pauli : C = B ∘ A.

    La composition correspond au produit des diagonales du PTM.
    C'est exact, pas une approximation additive.
    """
    Rxx_A, Ryy_A, Rzz_A = pauli_transfer_matrix_diag(channel_A)
    Rxx_B, Ryy_B, Rzz_B = pauli_transfer_matrix_diag(channel_B)
    return probs_from_ptm_diag(
        Rxx_A * Rxx_B,
        Ryy_A * Ryy_B,
        Rzz_A * Rzz_B,
    )


def pauli_twirl_channel(
    T1: float,
    T_phi: float,
    tau: float,
) -> Dict[str, float]:
    """
    Construit le canal de Pauli twirled complet pour une durée tau.

    Étapes :
    1. gamma_1, lambda_z depuis (T1, T_phi, tau).
    2. Canal AD twirled.
    3. Canal de déphasage pur.
    4. Composition exacte via PTM.
    """
    gamma_1, lambda_z = relaxation_dephasing_rates(T1, T_phi, tau)
    channel_AD = amplitude_damping_twirl(gamma_1)
    channel_dep = dephasing_twirl(lambda_z)
    return compose_pauli_channels(channel_AD, channel_dep)


# ----------------------------------------------------------------------
# Bruit de flux → déphasage
# ----------------------------------------------------------------------

def flux_noise_to_dephasing_rate(
    df01_dphi: float,
    S_phi: float,
    omega_cutoff: Optional[float] = None,
) -> float:
    """
    Convertit une densité spectrale de bruit de flux en taux de déphasage.

    Approximation de bruit blanc :
        Gamma_phi = 0.5 * (d f01/d Phi)² * S_Phi

    Unités :
    - df01_dphi en GHz / Phi0
    - S_phi en Phi0² / GHz (densité spectrale à fréquence nulle)
    - Gamma_phi en GHz (à convertir en s⁻¹ par 2 pi si besoin)

    Hook 1/f : si omega_cutoff est fourni, on applique une correction
    logarithmique (documentée, non utilisée par défaut).
    """
    if df01_dphi == 0.0:
        return 0.0
    gamma_phi = 0.5 * (df01_dphi ** 2) * S_phi
    if omega_cutoff is not None:
        # Correction 1/f documentée (non active par défaut)
        LOGGER.info(
            "Hook 1/f fourni (omega_cutoff=%.3e) mais non appliqué "
            "dans cette version.", omega_cutoff,
        )
    return float(gamma_phi)


def dephasing_rate_to_T_phi(
    gamma_phi: float,
) -> float:
    """Convertit un taux de déphasage en T_phi (mêmes unités de temps)."""
    if gamma_phi <= 0:
        return float("inf")
    return 1.0 / gamma_phi


# ----------------------------------------------------------------------
# Erreur de lecture
# ----------------------------------------------------------------------

def apply_readout_error(
    pauli_probs: Dict[str, float],
    p_readout: float,
) -> Dict[str, float]:
    """
    Applique une erreur de lecture classique (bit-flip) après le canal.

    Modèle : la mesure confond |0> et |1> avec probabilité p_readout.
    Cela ne change pas les probabilités Pauli du canal quantique,
    mais doit être pris en compte séparément dans le décodeur QEC.
    Cette fonction retourne simplement le canal inchangé, avec un
    avertissement si p_readout > 0.
    """
    if p_readout > 0:
        LOGGER.info(
            "Erreur de lecture p=%.4f : à traiter séparément "
            "dans le décodeur QEC, pas dans le canal quantique.",
            p_readout,
        )
    return pauli_probs


# ----------------------------------------------------------------------
# Pipeline complet
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelConfig:
    """Configuration du canal de bruit."""

    T1: float          # secondes
    T_phi: float       # secondes
    tau: float         # secondes (durée du cycle)
    df01_dphi: float = 0.0   # GHz / Phi0
    S_phi: float = 0.0       # Phi0² / GHz
    p_readout: float = 0.0   # probabilité d'erreur de lecture

    def __post_init__(self):
        if self.T1 <= 0 or self.T_phi <= 0 or self.tau <= 0:
            raise ValueError("T1, T_phi et tau doivent être positifs.")
        if self.p_readout < 0 or self.p_readout > 1:
            raise ValueError("p_readout doit être dans [0, 1].")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def build_pauli_channel(
    config: ChannelConfig,
) -> Dict[str, float]:
    """
    Construit le canal de Pauli twirled complet à partir de la config.

    Si df01_dphi et S_phi sont fournis, T_phi est recalculé à partir
    du bruit de flux (priorité au modèle physique sur le T_phi direct).
    """
    T_phi = config.T_phi
    if config.df01_dphi != 0.0 and config.S_phi != 0.0:
        gamma_phi = flux_noise_to_dephasing_rate(
            config.df01_dphi, config.S_phi
        )
        T_phi = dephasing_rate_to_T_phi(gamma_phi)
        LOGGER.info(
            "T_phi recalculé depuis S_Phi : %.6g s", T_phi
        )

    channel = pauli_twirl_channel(
        config.T1, T_phi, config.tau
    )
    return apply_readout_error(channel, config.p_readout)


def effective_error_rate(
    pauli_probs: Dict[str, float],
) -> float:
    """Probabilité d'erreur totale = 1 - p_I."""
    return clip01(1.0 - pauli_probs["I"])


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_amplitude_damping_twirl_normalization():
    """Le twirl de l'AD doit sommer à 1."""
    probs = amplitude_damping_twirl(0.1)
    assert abs(sum(probs.values()) - 1.0) < 1e-12


def test_dephasing_twirl_normalization():
    """Le canal de déphasage doit sommer à 1."""
    probs = dephasing_twirl(0.05)
    assert abs(sum(probs.values()) - 1.0) < 1e-12


def test_composition_associative():
    """La composition via PTM doit être associative."""
    A = amplitude_damping_twirl(0.1)
    B = dephasing_twirl(0.05)
    C = amplitude_damping_twirl(0.2)

    AB_C = compose_pauli_channels(
        compose_pauli_channels(A, B), C
    )
    A_BC = compose_pauli_channels(
        A, compose_pauli_channels(B, C)
    )
    for k in ("I", "X", "Y", "Z"):
        assert abs(AB_C[k] - A_BC[k]) < 1e-12


def test_zero_tau_identity():
    """Pour tau -> 0, le canal doit tendre vers l'identité."""
    channel = pauli_twirl_channel(
        T1=1e-3, T_phi=1e-3, tau=1e-15
    )
    assert channel["I"] > 0.9999
    assert channel["X"] < 1e-6
    assert channel["Y"] < 1e-6
    assert channel["Z"] < 1e-6


def test_ptm_round_trip():
    """Le PTM doit être réversible."""
    probs = {"I": 0.7, "X": 0.1, "Y": 0.1, "Z": 0.1}
    Rxx, Ryy, Rzz = pauli_transfer_matrix_diag(probs)
    recovered = probs_from_ptm_diag(Rxx, Ryy, Rzz)
    for k in ("I", "X", "Y", "Z"):
        assert abs(probs[k] - recovered[k]) < 1e-12


def test_flux_noise_zero_derivative():
    """Si df01/dphi = 0 (sweet spot), Gamma_phi = 0."""
    gamma = flux_noise_to_dephasing_rate(0.0, 1e-6)
    assert gamma == 0.0


def run_all_tests():
    """Lance tous les tests."""
    test_amplitude_damping_twirl_normalization()
    test_dephasing_twirl_normalization()
    test_composition_associative()
    test_zero_tau_identity()
    test_ptm_round_trip()
    test_flux_noise_zero_derivative()
    LOGGER.info("Tous les tests sont passés.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Canal de bruit et Pauli twirling."
    )
    parser.add_argument("--T1", type=float, default=7.19e-6,
                        help="T1 en secondes.")
    parser.add_argument("--T_phi", type=float, default=45960e-6,
                        help="T_phi en secondes.")
    parser.add_argument("--tau", type=float, default=20e-9,
                        help="Durée du cycle en secondes.")
    parser.add_argument("--df01_dphi", type=float, default=0.0,
                        help="d f01/d Phi en GHz/Phi0.")
    parser.add_argument("--S_phi", type=float, default=0.0,
                        help="S_Phi en Phi0²/GHz.")
    parser.add_argument("--p_readout", type=float, default=0.0,
                        help="Probabilité d'erreur de lecture.")
    parser.add_argument("--output", type=str, default=None,
                        help="Fichier JSON de sortie.")
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
    )
    channel = build_pauli_channel(config)
    result = {
        **channel,
        "effective_error_rate": effective_error_rate(channel),
    }
    print(json.dumps(result, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        LOGGER.info("Résultat sauvegardé : %s", args.output)


if __name__ == "__main__":
    main()
