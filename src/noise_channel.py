
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOISE CHANNEL — Pauli twirling et composition PTM.

Unités :
    T1, T_phi, tau : secondes
    df01_dphi : GHz / Phi0
    S_phi : Phi0² / GHz (convention phénoménologique)
    Probabilités : sans dimension

Le bruit de flux est désactivé par défaut.
Sa convention spectrale doit être justifiée avant toute interprétation
physique quantitative.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

LOGGER = logging.getLogger(__name__)
PAULI_KEYS = ("I", "X", "Y", "Z")
_TOL = 1e-12


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} doit être fini.")
    return value


def _positive_time(
    name: str,
    value: float,
    allow_infinity: bool = False,
) -> float:
    value = float(value)

    if math.isnan(value) or value <= 0.0:
        raise ValueError(f"{name} doit être strictement positif.")

    if math.isinf(value) and not (
        allow_infinity and value > 0.0
    ):
        raise ValueError(f"{name} doit être fini.")

    return value


def _validate_probabilities(
    probs: Dict[str, float],
) -> Dict[str, float]:
    missing = set(PAULI_KEYS) - set(probs)
    if missing:
        raise ValueError(
            f"Probabilités manquantes : {sorted(missing)}"
        )

    result = {k: float(probs[k]) for k in PAULI_KEYS}

    for key, value in result.items():
        _finite(key, value)
        if value < -_TOL or value > 1.0 + _TOL:
            raise ValueError(
                f"Probabilité {key} hors de [0,1] : {value}"
            )

    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"Canal non normalisé : somme={total:.16g}"
        )

    # Correction des seules erreurs d'arrondi, suivie d'une
    # renormalisation explicite.
    result = {k: max(0.0, v) for k, v in result.items()}
    total = sum(result.values())

    if total <= 0.0:
        raise ValueError("Somme des probabilités nulle.")

    result = {k: v / total for k, v in result.items()}
    return result


def amplitude_damping_twirl(gamma: float) -> Dict[str, float]:
    """
    Pauli twirling exact du canal amplitude damping standard.

    K0 = diag(1, sqrt(1-gamma))
    K1 = sqrt(gamma) |0><1|
    """
    gamma = _finite("gamma", gamma)

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma doit appartenir à [0,1].")

    root = math.sqrt(1.0 - gamma)

    return _validate_probabilities({
        "I": ((1.0 + root) ** 2) / 4.0,
        "X": gamma / 4.0,
        "Y": gamma / 4.0,
        "Z": ((1.0 - root) ** 2) / 4.0,
    })


def dephasing_twirl(lambda_z: float) -> Dict[str, float]:
    lambda_z = _finite("lambda_z", lambda_z)

    if not 0.0 <= lambda_z <= 1.0:
        raise ValueError("lambda_z doit appartenir à [0,1].")

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
    """Retourne gamma_1 et lambda_z pour une durée tau."""
    T1 = _positive_time("T1", T1)
    T_phi = _positive_time(
        "T_phi", T_phi, allow_infinity=True
    )
    tau = _positive_time("tau", tau)

    gamma_1 = -math.expm1(-tau / T1)

    if math.isinf(T_phi):
        lambda_z = 0.0
    else:
        lambda_z = -0.5 * math.expm1(-tau / T_phi)

    return gamma_1, lambda_z


def pauli_transfer_matrix_diag(
    probs: Dict[str, float],
) -> Tuple[float, float, float]:
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
    Rxx = _finite("Rxx", Rxx)
    Ryy = _finite("Ryy", Ryy)
    Rzz = _finite("Rzz", Rzz)

    probs = {
        "I": (1.0 + Rxx + Ryy + Rzz) / 4.0,
        "X": (1.0 + Rxx - Ryy - Rzz) / 4.0,
        "Y": (1.0 - Rxx + Ryy - Rzz) / 4.0,
        "Z": (1.0 - Rxx - Ryy + Rzz) / 4.0,
    }

    for key, value in probs.items():
        if value < -1e-10 or value > 1.0 + 1e-10:
            raise ValueError(
                f"PTM non physique : p_{key}={value:.12g}"
            )

    return _validate_probabilities(probs)


def compose_pauli_channels(
    channel_A: Dict[str, float],
    channel_B: Dict[str, float],
) -> Dict[str, float]:
    """Composition C = B après A."""
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
    gamma_1, lambda_z = relaxation_dephasing_rates(
        T1, T_phi, tau
    )

    return compose_pauli_channels(
        amplitude_damping_twirl(gamma_1),
        dephasing_twirl(lambda_z),
    )


def flux_noise_to_dephasing_rate(
    df01_dphi: float,
    S_phi: float,
) -> float:
    """
    Approximation phénoménologique :

        gamma_phi_GHz = 0.5 * (df01/dPhi)^2 * S_phi

    Résultat converti en s^-1.

    Cette expression n'est pas une formule universelle :
    sa validité dépend de la convention spectrale et du régime de bruit.
    """
    df01_dphi = _finite("df01_dphi", df01_dphi)
    S_phi = _finite("S_phi", S_phi)

    if S_phi < 0.0:
        raise ValueError("S_phi ne peut pas être négatif.")

    gamma_GHz = 0.5 * df01_dphi**2 * S_phi
    return gamma_GHz * 1e9


def dephasing_rate_to_T_phi(
    gamma_phi_s: float,
) -> float:
    gamma_phi_s = _finite("gamma_phi_s", gamma_phi_s)

    if gamma_phi_s < 0.0:
        raise ValueError("Le taux ne peut pas être négatif.")

    if gamma_phi_s == 0.0:
        return math.inf

    return 1.0 / gamma_phi_s


def apply_readout_error(
    pauli_probs: Dict[str, float],
    p_readout: float,
) -> Dict[str, float]:
    p_readout = _finite("p_readout", p_readout)

    if not 0.0 <= p_readout <= 1.0:
        raise ValueError("p_readout doit appartenir à [0,1].")

    if p_readout > 0.0:
        LOGGER.info(
            "Erreur de lecture classique séparée : %.6g",
            p_readout,
        )

    return _validate_probabilities(pauli_probs)


@dataclass(frozen=True)
class ChannelConfig:
    T1: float
    T_phi: float
    tau: float
    df01_dphi: float = 0.0
    S_phi: float = 0.0
    p_readout: float = 0.0
    use_flux_noise: bool = False
    combine_flux_noise: bool = False

    def __post_init__(self):
        _positive_time("T1", self.T1)
        _positive_time(
            "T_phi", self.T_phi, allow_infinity=True
        )
        _positive_time("tau", self.tau)

        _finite("df01_dphi", self.df01_dphi)
        _finite("S_phi", self.S_phi)
        _finite("p_readout", self.p_readout)

        if self.S_phi < 0.0:
            raise ValueError("S_phi ne peut pas être négatif.")

        if not 0.0 <= self.p_readout <= 1.0:
            raise ValueError("p_readout doit appartenir à [0,1].")

        if self.combine_flux_noise and not self.use_flux_noise:
            raise ValueError(
                "combine_flux_noise exige use_flux_noise=True."
            )

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        if math.isinf(self.T_phi):
            result["T_phi"] = "Infinity"
        return result


def build_pauli_channel(
    config: ChannelConfig,
) -> Dict[str, float]:
    T_phi = config.T_phi

    if config.use_flux_noise:
        if config.S_phi <= 0.0:
            raise ValueError(
                "use_flux_noise=True exige S_phi > 0."
            )

        gamma_flux = flux_noise_to_dephasing_rate(
            config.df01_dphi,
            config.S_phi,
        )

        if config.combine_flux_noise:
            gamma_base = (
                0.0 if math.isinf(T_phi) else 1.0 / T_phi
            )
            T_phi = dephasing_rate_to_T_phi(
                gamma_base + gamma_flux
            )
        else:
            T_phi = dephasing_rate_to_T_phi(gamma_flux)

    channel = pauli_twirl_channel(
        T1=config.T1,
        T_phi=T_phi,
        tau=config.tau,
    )

    return apply_readout_error(channel, config.p_readout)


def effective_error_rate(
    pauli_probs: Dict[str, float],
) -> float:
    p = _validate_probabilities(pauli_probs)
    return 1.0 - p["I"]


def run_all_tests() -> None:
    for gamma in (0.0, 1e-9, 0.1, 0.5, 1.0):
        p = amplitude_damping_twirl(gamma)
        assert abs(sum(p.values()) - 1.0) < 1e-12

    gamma = 1e-5
    p = amplitude_damping_twirl(gamma)
    assert abs(p["X"] - gamma / 4.0) < 1e-14
    assert abs(p["Y"] - gamma / 4.0) < 1e-14
    assert p["Z"] < 1e-9

    p = dephasing_twirl(0.05)
    assert abs(sum(p.values()) - 1.0) < 1e-12

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

    original = {"I": 0.7, "X": 0.1, "Y": 0.1, "Z": 0.1}
    reconstructed = probs_from_ptm_diag(
        *pauli_transfer_matrix_diag(original)
    )

    for key in PAULI_KEYS:
        assert abs(original[key] - reconstructed[key]) < 1e-12

    channel = pauli_twirl_channel(
        T1=1e-3,
        T_phi=math.inf,
        tau=1e-15,
    )
    assert channel["I"] > 0.999999

    config = ChannelConfig(
        T1=7e-6,
        T_phi=math.inf,
        tau=20e-9,
    )
    assert abs(sum(build_pauli_channel(config).values()) - 1.0) < 1e-10

    print("Tous les tests noise_channel.py sont passés.")


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
    parser.add_argument("--combine_flux_noise", action="store_true")
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
        combine_flux_noise=args.combine_flux_noise,
    )

    channel = build_pauli_channel(config)
    result = {
        **channel,
        "effective_error_rate": effective_error_rate(channel),
        "configuration": config.to_dict(),
    }

    print(json.dumps(result, indent=2, allow_nan=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        LOGGER.info("Résultat sauvegardé : %s", output_path)


if __name__ == "__main__":
    main()
