#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUXONIUM CORE — VERSION RECONSTRUITE ET AUDITABLE
=================================================

Fichier autonome : fluxonium_core.py

Fonctions :
    1. Hamiltonien du fluxonium dans une base harmonique tronquée.
    2. Diagonalisation et éléments de matrice.
    3. Dérivées spectrales analytiques par perturbation.
    4. Métrique de Fubini–Study et contrôle par différences finies.
    5. Relaxation diélectrique capacitive à Q constant.
    6. Relaxation quasiparticulaire froide de la petite jonction.
    7. Bruit quasi statique 1/f et enveloppe de Ramsey quadratique.
    8. Réduction à un qubit effectif.
    9. Générateur de Lindblad pour quatre qubits.
   10. Mesures projectives des stabilisateurs [[4,2,2]].
   11. Post-sélection répétée du secteur (++), sans correction d'erreur.
   12. Diagnostics, tests, exports JSON/CSV et interface CLI.

CONVENTIONS
-----------
EC, EL, EJ, f01 :
    valeurs en GHz, c'est-à-dire énergie/fréquence divisée par h.

phi_ext :
    phase externe en radians, phi_ext = 2*pi*Phi_ext/Phi0.

Dérivées spectrales :
    GHz/rad et GHz/rad².

Taux :
    s^-1. Hamiltonien de Lindblad :
    H_rad_s = 2*pi*H_Hz.

Temps :
    secondes dans le noyau, microsecondes dans certains rapports.

Vecteurs propres :
    colonnes orthonormées ; produits scalaires via np.vdot.

Vectorisation de rho :
    ordre Fortran (par colonnes), vec(A rho B) = (B.T kron A) vec(rho).

LIMITES PHYSIQUES
-----------------
- Le modèle diélectrique suppose un facteur Q constant.
- Le modèle QP suppose des quasiparticules froides, une petite jonction
  seulement, et néglige le taux montant QP.
- Le bruit 1/f est traité comme quasi statique et n'est pas ajouté au
  générateur de Lindblad.
- Les quatre qubits sont identiques et le couplage XX est phénoménologique.
- La mesure des stabilisateurs est idéale et projective.
- La post-sélection (++) n'est pas une correction d'erreur.
- Les taux calculés ne sont pas une calibration expérimentale.

Dépendances :
    numpy
    scipy
"""

from __future__ import annotations

import os
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply


# =============================================================================
# 0. CONSTANTES ET CONVENTIONS D'UNITÉS
# =============================================================================

TWO_PI = 2.0 * math.pi
GHZ_TO_HZ = 1.0e9
KB_OVER_H_GHZ_K = 20.836619123
DEFAULT_PHI_EXT = math.pi
JSON_SCHEMA_VERSION = "1.1"
MODEL_NAME = "FLUXONIUM CORE"


def ghz_to_hz(value: float) -> float:
    """Convertit une fréquence GHz en Hz."""
    return float(value) * GHZ_TO_HZ


def hz_to_ghz(value: float) -> float:
    """Convertit une fréquence Hz en GHz."""
    return float(value) / GHZ_TO_HZ


def hz_to_angular(value: float) -> float:
    """Convertit Hz en rad/s."""
    return TWO_PI * float(value)


def ghz_to_angular(value: float) -> float:
    """Convertit GHz en rad/s."""
    return TWO_PI * GHZ_TO_HZ * float(value)


def seconds_to_microseconds(value: float) -> float:
    """Convertit secondes en microsecondes."""
    return float(value) * 1.0e6


def microseconds_to_seconds(value: float) -> float:
    """Convertit microsecondes en secondes."""
    return float(value) * 1.0e-6


# =============================================================================
# 1. VALIDATION ET SÉRIALISATION
# =============================================================================

def finite_number(name: str, value: float) -> float:
    """Convertit en float et rejette NaN/inf."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} ne doit pas être booléen.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} doit être numérique.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} doit être fini.")
    return result


def positive_number(name: str, value: float) -> float:
    """Validation d'un réel strictement positif."""
    result = finite_number(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} doit être strictement positif.")
    return result


def nonnegative_number(name: str, value: float) -> float:
    """Validation d'un réel positif ou nul."""
    result = finite_number(name, value)
    if result < 0.0:
        raise ValueError(f"{name} doit être >= 0.")
    return result


def integer_at_least(name: str, value: int, minimum: int) -> int:
    """Validation stricte d'un entier."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum):
        raise ValueError(f"{name} doit être un entier >= {minimum}.")
    return int(value)


def inverse_rate(rate: float) -> float:
    """Temps caractéristique associé à un taux."""
    rate = nonnegative_number("rate", rate)
    return math.inf if rate == 0.0 else 1.0 / rate


def check(condition: bool, message: str) -> None:
    """Assertion nommée utilisée par les tests internes."""
    if not condition:
        raise AssertionError(message)


def json_compatible(value: Any) -> Any:
    """
    Convertit récursivement les objets numériques en objets JSON stricts.
    Les infinis et NaN sont représentés par null.
    """
    if isinstance(value, dict):
        return {str(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            return None
        return {"real": value.real, "imag": value.imag}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def write_json(path: Path, payload: Any, indent: int = 2) -> None:
    """Écrit un JSON strict, UTF-8, après conversion des scalaires."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_compatible(payload), indent=indent,
                      ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def relative_error(a: float, b: float, floor: float = 1.0e-14) -> float:
    """Erreur relative avec plancher numérique."""
    return abs(a - b) / max(abs(a), abs(b), floor)


# =============================================================================
# 2. PARAMÈTRES DU FLUXONIUM
# =============================================================================

@dataclass(frozen=True)
class FluxoniumParams:
    """Paramètres physiques et numériques du solveur."""

    EC: float = 1.0
    EL: float = 0.5
    EJ: float = 4.0

    N_basis: int = 45
    N_levels: int = 4
    operator_padding: int = 35

    Q_diel: float = 6.0e5
    T: float = 0.020

    x_qp: float = 1.0e-8
    gap_GHz: float = 45.0

    A_phi: float = 2.0e-6
    noise_f_low_Hz: float = 1.0
    noise_f_high_Hz: float = 1.0e3

    def __post_init__(self) -> None:
        positive_number("EC", self.EC)
        positive_number("EL", self.EL)
        nonnegative_number("EJ", self.EJ)
        integer_at_least("N_basis", self.N_basis, 10)
        integer_at_least("N_levels", self.N_levels, 3)
        integer_at_least("operator_padding", self.operator_padding, 0)
        if self.N_levels > self.N_basis:
            raise ValueError("N_levels ne peut pas dépasser N_basis.")
        positive_number("Q_diel", self.Q_diel)
        nonnegative_number("T", self.T)
        nonnegative_number("x_qp", self.x_qp)
        positive_number("gap_GHz", self.gap_GHz)
        nonnegative_number("A_phi", self.A_phi)
        positive_number("noise_f_low_Hz", self.noise_f_low_Hz)
        positive_number("noise_f_high_Hz", self.noise_f_high_Hz)
        if self.noise_f_high_Hz <= self.noise_f_low_Hz:
            raise ValueError("La coupure haute doit dépasser la basse.")

    def as_dict(self) -> dict[str, Any]:
        """Retourne les paramètres sous forme de dictionnaire."""
        return asdict(self)

    def with_updates(self, **kwargs: Any) -> "FluxoniumParams":
        """Construit une copie immuable avec paramètres modifiés."""
        return replace(self, **kwargs)


# =============================================================================
# 3. OPÉRATEURS HARMONIQUES
# =============================================================================

def annihilation_operator(dimension: int) -> np.ndarray:
    """Opérateur d'annihilation dans une base de Fock tronquée."""
    dimension = integer_at_least("dimension", dimension, 2)
    return np.diag(np.sqrt(np.arange(1, dimension, dtype=float)), 1)


def creation_operator(dimension: int) -> np.ndarray:
    """Opérateur de création dans une base de Fock tronquée."""
    return annihilation_operator(dimension).T


def number_operator(dimension: int) -> np.ndarray:
    """Opérateur a†a."""
    dimension = integer_at_least("dimension", dimension, 2)
    return np.diag(np.arange(dimension, dtype=float)).astype(complex)


def hermiticity_error(matrix: np.ndarray) -> float:
    """Norme de l'écart à l'hermiticité."""
    return float(np.linalg.norm(matrix - matrix.conj().T))


# =============================================================================
# 4. SOLVEUR FLUXONIUM
# =============================================================================

class FluxoniumSolver:
    """
    Diagonalisation du Hamiltonien :
        H/h = 4 EC n² + EL phi²/2 - EJ cos(phi - phi_ext).

    La base harmonique est celle de l'oscillateur LC défini par EC et EL.
    Les opérateurs sin/cos sont projetés depuis une base auxiliaire élargie.
    """

    def __init__(self, params: FluxoniumParams):
        self.p = params
        n = params.N_basis
        m = n + params.operator_padding

        self.phi_zpf = (2.0 * params.EC / params.EL) ** 0.25
        self.f_lc_GHz = math.sqrt(8.0 * params.EC * params.EL)

        a = annihilation_operator(n)
        self.phi_op = self.phi_zpf * (a + a.T)
        self.n_op = -1j * (a - a.T) / (2.0 * self.phi_zpf)
        self.H_lc = np.diag(
            self.f_lc_GHz * (np.arange(n, dtype=float) + 0.5)
        ).astype(complex)

        a_aux = annihilation_operator(m)
        phi_aux = self.phi_zpf * (a_aux + a_aux.T)
        values, vectors = np.linalg.eigh(phi_aux)
        projected = vectors[:n, :]

        self.cos_phi_op = (projected * np.cos(values)) @ projected.T
        self.sin_phi_op = (projected * np.sin(values)) @ projected.T
        self.cos_half_phi_op = (
            projected * np.cos(0.5 * values)
        ) @ projected.T
        self.sin_half_phi_op = (
            projected * np.sin(0.5 * values)
        ) @ projected.T

        self._validate_internal_operators()

    def _validate_internal_operators(self) -> None:
        """Contrôles de cohérence des opérateurs pré-calculés."""
        for name in ("phi_op", "n_op", "H_lc", "cos_phi_op",
                     "sin_phi_op", "cos_half_phi_op",
                     "sin_half_phi_op"):
            matrix = getattr(self, name)
            if not np.all(np.isfinite(matrix)):
                raise ArithmeticError(f"{name} contient une valeur non finie.")
            if hermiticity_error(matrix) > 1.0e-9:
                raise ArithmeticError(f"{name} n'est pas hermitien.")

    def energie_josephson(self, EJ_override: float | None = None) -> float:
        """Renvoie EJ, éventuellement remplacé pour un balayage."""
        value = self.p.EJ if EJ_override is None else EJ_override
        return nonnegative_number("EJ", value)

    def hamiltonien(self, phi_ext: float,
                    EJ_override: float | None = None) -> np.ndarray:
        """Hamiltonien H/h en GHz."""
        phase = finite_number("phi_ext", phi_ext)
        ej = self.energie_josephson(EJ_override)
        cos_shift = (math.cos(phase) * self.cos_phi_op
                     + math.sin(phase) * self.sin_phi_op)
        return self.H_lc - ej * cos_shift

    def operateur_qp(self, phi_ext: float) -> np.ndarray:
        """Opérateur sin((phi-phi_ext)/2) de la petite jonction."""
        phase = finite_number("phi_ext", phi_ext)
        return (math.cos(phase / 2.0) * self.sin_half_phi_op
                - math.sin(phase / 2.0) * self.cos_half_phi_op)

    def derivees_hamiltonien(self, phi_ext: float,
                             EJ_override: float | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        """Retourne dH/dphi_ext et d²H/dphi_ext², en GHz."""
        phase = finite_number("phi_ext", phi_ext)
        ej = self.energie_josephson(EJ_override)
        sin_shift = (math.cos(phase) * self.sin_phi_op
                     - math.sin(phase) * self.cos_phi_op)
        cos_shift = (math.cos(phase) * self.cos_phi_op
                     + math.sin(phase) * self.sin_phi_op)
        return -ej * sin_shift, ej * cos_shift

    def diagonaliser(self, phi_ext: float,
                     EJ_override: float | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalise le Hamiltonien complet tronqué."""
        return np.linalg.eigh(self.hamiltonien(phi_ext, EJ_override))

    def resoudre_spectre(self, phi_ext: float,
                         EJ_override: float | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Retourne les N_levels énergies relatives et vecteurs propres."""
        energies, vectors = self.diagonaliser(phi_ext, EJ_override)
        k = self.p.N_levels
        return energies[:k] - energies[0], vectors[:, :k]

    def geometrie_spectrale(self, phi_ext: float,
                            EJ_override: float | None = None) -> dict:
        """
        Dérivées de Hellmann–Feynman et courbures du gap 0→1.
        Calcule aussi la métrique de Fubini–Study du fondamental.
        """
        energies, vectors = self.diagonaliser(phi_ext, EJ_override)
        dH, d2H = self.derivees_hamiltonien(phi_ext, EJ_override)
        dH_e = vectors.conj().T @ dH @ vectors
        d2H_e = vectors.conj().T @ d2H @ vectors

        derivatives: list[float] = []
        curvatures: list[float] = []
        for state in (0, 1):
            mask = np.arange(len(energies)) != state
            gaps = energies[state] - energies[mask]
            if np.min(np.abs(gaps)) < 1.0e-10:
                raise ArithmeticError(
                    "Quasi-dégénérescence : formule non dégénérée invalide."
                )
            derivatives.append(float(dH_e[state, state].real))
            curvature = (d2H_e[state, state].real
                         + 2.0 * np.sum(
                             np.abs(dH_e[mask, state]) ** 2 / gaps))
            curvatures.append(float(curvature))

        gaps0 = energies[1:] - energies[0]
        metric = float(np.sum(np.abs(dH_e[1:, 0]) ** 2 / gaps0**2))
        return {
            "energies": energies,
            "vectors": vectors,
            "df01_dphi_GHz_per_rad": derivatives[1] - derivatives[0],
            "d2f01_dphi2_GHz_per_rad2": curvatures[1] - curvatures[0],
            "g_FS_per_rad2": metric,
        }

    def metrique_difference_finie(self, phi_ext: float,
                                  eps: float = 1.0e-4,
                                  EJ_override: float | None = None) -> float:
        """Estimation de la métrique du fondamental par recouvrement."""
        eps = positive_number("eps", eps)
        _, vm = self.diagonaliser(phi_ext - eps, EJ_override)
        _, vp = self.diagonaliser(phi_ext + eps, EJ_override)
        u = vm[:, 0].astype(complex)
        v = vp[:, 0].astype(complex)
        overlap = np.vdot(u, v)
        if abs(overlap) < 1.0e-12:
            raise ArithmeticError("Recouvrement trop faible pour cette valeur eps.")
        v *= np.exp(-1j * np.angle(overlap))
        derivative = (v - u) / (2.0 * eps)
        tangent = derivative - u * np.vdot(u, derivative)
        return float(np.vdot(tangent, tangent).real)

    def elements_matrice(self, phi_ext: float,
                         EJ_override: float | None = None) -> dict:
        """Éléments de matrice entre |0> et |1>."""
        energies, vectors = self.diagonaliser(phi_ext, EJ_override)
        g = vectors[:, 0]
        e = vectors[:, 1]
        return {
            "n01": float(abs(np.vdot(g, self.n_op @ e))),
            "phi01": float(abs(np.vdot(g, self.phi_op @ e))),
            "qp01": float(abs(np.vdot(
                g, self.operateur_qp(phi_ext) @ e))),
            "f01_GHz": float(energies[1] - energies[0]),
            "f12_GHz": float(energies[2] - energies[1]),
        }

    def diagnostic_base(self, phi_ext: float) -> dict:
        """Diagnostic simple de troncature de l'état fondamental."""
        _, vectors = self.diagonaliser(phi_ext)
        ground = vectors[:, 0]
        tail = float(np.sum(np.abs(ground[-5:]) ** 2))
        return {
            "N_basis": self.p.N_basis,
            "operator_padding": self.p.operator_padding,
            "ground_tail_last5": tail,
            "phi_zpf": self.phi_zpf,
            "f_lc_GHz": self.f_lc_GHz,
        }


# =============================================================================
# 5. TAUX DIÉLECTRIQUES ET QUASIPARTICULAIRES
# =============================================================================

def occupation_bose(frequency_GHz: float, temperature_K: float) -> float:
    """Occupation thermique de Bose à la fréquence spécifiée."""
    f = positive_number("frequency_GHz", frequency_GHz)
    temperature = nonnegative_number("temperature_K", temperature_K)
    if temperature == 0.0:
        return 0.0
    x = f / (KB_OVER_H_GHZ_K * temperature)
    if x > 700.0:
        return 0.0
    return 1.0 / math.expm1(x)


def taux_dielectriques(params: FluxoniumParams,
                       f01_GHz: float, n01: float) -> dict[str, float]:
    """
    Modèle capacitif à Q constant :
    Gamma0 = 2*pi*1e9*(16 EC/Q)*|n01|².
    """
    frequency = positive_number("f01_GHz", f01_GHz)
    element = nonnegative_number("n01", n01)
    gamma0 = (TWO_PI * GHZ_TO_HZ * 16.0 * params.EC
              / params.Q_diel * element**2)
    occupation = occupation_bose(frequency, params.T)
    down = gamma0 * (occupation + 1.0)
    up = gamma0 * occupation
    return {
        "gamma0_s_inv": gamma0,
        "n_therm": occupation,
        "down_s_inv": down,
        "up_s_inv": up,
        "total_s_inv": down + up,
    }


def taux_quasiparticules(params: FluxoniumParams,
                         f01_GHz: float, qp01: float,
                         EJ_override: float | None = None) -> float:
    """
    Taux QP descendant :
    (8/pi)*(EJ/hbar)*x_qp*sqrt(2 Delta/(h f01))*|A01|².
    Toutes les énergies de la racine sont exprimées en GHz.
    """
    frequency = positive_number("f01_GHz", f01_GHz)
    element = nonnegative_number("qp01", qp01)
    ej = params.EJ if EJ_override is None else EJ_override
    ej = nonnegative_number("EJ", ej)
    if params.x_qp == 0.0 or ej == 0.0 or element == 0.0:
        return 0.0
    ej_over_hbar = TWO_PI * GHZ_TO_HZ * ej
    return ((8.0 / math.pi) * ej_over_hbar * params.x_qp
            * math.sqrt(2.0 * params.gap_GHz / frequency) * element**2)


def decomposition_taux(params: FluxoniumParams, solver: FluxoniumSolver,
                       phi_ext: float) -> dict:
    """Retourne séparément les canaux de pertes du qubit."""
    elements = solver.elements_matrice(phi_ext)
    diel = taux_dielectriques(params, elements["f01_GHz"], elements["n01"])
    qp = taux_quasiparticules(params, elements["f01_GHz"],
                              elements["qp01"])
    return {
        "dielectric": diel,
        "quasiparticle": {"down_s_inv": qp, "up_s_inv": 0.0},
        "total": {
            "down_s_inv": diel["down_s_inv"] + qp,
            "up_s_inv": diel["up_s_inv"],
        },
    }


# =============================================================================
# 6. BRUIT QUASI STATIQUE ET RAMSEY
# =============================================================================

def sigma_flux_phase(params: FluxoniumParams) -> float:
    """Écart-type de phase associé à la PSD unilatérale A_phi²/f."""
    variance_reduced_flux = (params.A_phi**2
                             * math.log(params.noise_f_high_Hz
                                        / params.noise_f_low_Hz))
    return TWO_PI * math.sqrt(variance_reduced_flux)


def log_enveloppe_ramsey(t_s: float, sigma_rad: float,
                         df_GHz_rad: float,
                         d2f_GHz_rad2: float) -> float:
    """Log du module de l'enveloppe quasi statique au second ordre."""
    t = nonnegative_number("t_s", t_s)
    sigma = nonnegative_number("sigma_rad", sigma_rad)
    df = finite_number("df_GHz_rad", df_GHz_rad)
    d2f = finite_number("d2f_GHz_rad2", d2f_GHz_rad2)
    a = TWO_PI * GHZ_TO_HZ * df
    b = TWO_PI * GHZ_TO_HZ * d2f
    z = b * sigma**2 * t
    return (-0.25 * math.log1p(z*z)
            -0.5 * (a*sigma*t)**2 / (1.0 + z*z))


def temps_ramsey_1e(sigma_rad: float, df_GHz_rad: float,
                    d2f_GHz_rad2: float,
                    gamma1_s: float = 0.0) -> float:
    """Temps à 1/e de exp(-Gamma1*t/2)*|W(t)|."""
    sigma = nonnegative_number("sigma_rad", sigma_rad)
    df = finite_number("df_GHz_rad", df_GHz_rad)
    d2f = finite_number("d2f_GHz_rad2", d2f_GHz_rad2)
    gamma1 = nonnegative_number("gamma1_s", gamma1_s)
    if gamma1 == 0.0 and (sigma == 0.0 or (df == 0.0 and d2f == 0.0)):
        return math.inf

    def residual(t: float) -> float:
        return 0.5 * gamma1 * t - log_enveloppe_ramsey(
            t, sigma, df, d2f) - 1.0

    lo, hi = 0.0, 1.0e-9
    for _ in range(256):
        if residual(hi) >= 0.0:
            break
        hi *= 2.0
        if not math.isfinite(hi):
            raise ArithmeticError("Encadrement Ramsey non fini.")
    else:
        raise ArithmeticError("Impossible d'encadrer le temps Ramsey.")
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if residual(mid) >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def enveloppe_ramsey(t_s: float, sigma_rad: float,
                     df_GHz_rad: float,
                     d2f_GHz_rad2: float,
                     gamma1_s: float = 0.0) -> float:
    """Module de cohérence incluant la relaxation T1."""
    gamma = nonnegative_number("gamma1_s", gamma1_s)
    exponent = (-0.5 * gamma * nonnegative_number("t_s", t_s)
                + log_enveloppe_ramsey(t_s, sigma_rad,
                                       df_GHz_rad, d2f_GHz_rad2))
    return math.exp(exponent) if exponent > -745.0 else 0.0


# =============================================================================
# 7. OBSERVABLES ET QUBIT EFFECTIF
# =============================================================================

def analyser_point_flux(solver: FluxoniumSolver,
                        phi_ext: float = DEFAULT_PHI_EXT,
                        EJ_override: float | None = None,
                        d_phi: float = 1.0e-4) -> dict[str, float]:
    """Calcule spectre, éléments de matrice, taux et temps caractéristiques."""
    positive_number("d_phi", d_phi)
    p = solver.p
    geo = solver.geometrie_spectrale(phi_ext, EJ_override)
    energies = geo["energies"]
    vectors = geo["vectors"]
    f01 = positive_number("f01_GHz", energies[1] - energies[0])
    f12 = float(energies[2] - energies[1])
    g, e = vectors[:, 0], vectors[:, 1]

    n01 = float(abs(np.vdot(g, solver.n_op @ e)))
    phi01 = float(abs(np.vdot(g, solver.phi_op @ e)))
    qp01 = float(abs(np.vdot(g, solver.operateur_qp(phi_ext) @ e)))

    diel = taux_dielectriques(p, f01, n01)
    qp_down = taux_quasiparticules(p, f01, qp01, EJ_override)
    gamma_down = diel["down_s_inv"] + qp_down
    gamma_up = diel["up_s_inv"]
    gamma1 = gamma_down + gamma_up

    df = geo["df01_dphi_GHz_per_rad"]
    d2f = geo["d2f01_dphi2_GHz_per_rad2"]
    sigma = sigma_flux_phase(p)
    tphi = temps_ramsey_1e(sigma, df, d2f)
    t2star = temps_ramsey_1e(sigma, df, d2f, gamma1)

    metric_fd = solver.metrique_difference_finie(
        phi_ext, d_phi, EJ_override)
    metric = geo["g_FS_per_rad2"]

    return {
        "EJ_GHz": solver.energie_josephson(EJ_override),
        "phi_ext_rad": float(phi_ext),
        "phi_ext_over_pi": float(phi_ext / math.pi),
        "f01_GHz": f01,
        "f12_GHz": f12,
        "anharmonicite_GHz": f12 - f01,
        "anharmonicite_rel": (f12 - f01) / f01,
        "n01": n01,
        "phi01": phi01,
        "qp01": qp01,
        "Gamma_diel_down_s_inv": diel["down_s_inv"],
        "Gamma_diel_up_s_inv": diel["up_s_inv"],
        "Gamma1_diel_s_inv": diel["total_s_inv"],
        "Gamma_qp_down_s_inv": qp_down,
        "Gamma_down_s_inv": gamma_down,
        "Gamma_up_s_inv": gamma_up,
        "Gamma1_total_s_inv": gamma1,
        "T1_diel_us": seconds_to_microseconds(
            inverse_rate(diel["total_s_inv"])),
        "T1_qp_us": seconds_to_microseconds(inverse_rate(qp_down)),
        "T1_total_us": seconds_to_microseconds(inverse_rate(gamma1)),
        "T_phi_us": seconds_to_microseconds(tphi),
        "T2_star_us": seconds_to_microseconds(t2star),
        "df01_dphi_GHz_per_rad": df,
        "d2f01_dphi2_GHz_per_rad2": d2f,
        "sigma_flux_phase_rad": sigma,
        "metrique_fubini_study": metric,
        "metrique_fubini_study_fd": metric_fd,
        "metric_relative_discrepancy": relative_error(metric, metric_fd),
        "ground_tail_last5": float(np.sum(np.abs(g[-5:])**2)),
    }


def exporter_qubit_effectif(solver: FluxoniumSolver,
                            phi_ext: float = DEFAULT_PHI_EXT,
                            EJ_override: float | None = None,
                            d_phi: float = 1.0e-4) -> dict:
    """Description sérialisable d'un qubit réduit à deux niveaux."""
    obs = analyser_point_flux(solver, phi_ext, EJ_override, d_phi)
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "units": {
            "frequency": "Hz",
            "angular_frequency": "rad/s",
            "rates": "s^-1",
            "external_phase": "rad",
        },
        "hamiltonian": {
            "f01_Hz": obs["f01_GHz"] * GHZ_TO_HZ,
            "omega01_rad_s": ghz_to_angular(obs["f01_GHz"]),
            "anharmonicity_Hz": obs["anharmonicite_GHz"] * GHZ_TO_HZ,
            "external_phase_rad": float(phi_ext),
        },
        "matrix_elements_abs": {
            "n01": obs["n01"],
            "phi01": obs["phi01"],
            "sin_junction_half_phase_01": obs["qp01"],
        },
        "lindblad_local_rates": {
            "gamma_down_s_inv": obs["Gamma_down_s_inv"],
            "gamma_up_s_inv": obs["Gamma_up_s_inv"],
            "dielectric_down_s_inv": obs["Gamma_diel_down_s_inv"],
            "dielectric_up_s_inv": obs["Gamma_diel_up_s_inv"],
            "cold_qp_down_s_inv": obs["Gamma_qp_down_s_inv"],
        },
        "quasistatic_flux_noise": {
            "sigma_external_phase_rad": obs["sigma_flux_phase_rad"],
            "df01_Hz_per_rad": obs["df01_dphi_GHz_per_rad"] * GHZ_TO_HZ,
            "d2f01_Hz_per_rad2": (
                obs["d2f01_dphi2_GHz_per_rad2"] * GHZ_TO_HZ),
            "psd_convention": "one-sided reduced-flux PSD A_phi^2/f",
            "f_low_Hz": solver.p.noise_f_low_Hz,
            "f_high_Hz": solver.p.noise_f_high_Hz,
        },
        "geometry": {
            "g_FS_per_rad2": obs["metrique_fubini_study"],
            "metric_relative_discrepancy": obs["metric_relative_discrepancy"],
        },
        "truncation": {
            "N_basis": solver.p.N_basis,
            "operator_padding": solver.p.operator_padding,
            "ground_tail_last5": obs["ground_tail_last5"],
        },
        "assumptions": {
            "two_level_reduction": True,
            "local_dielectric_bath": True,
            "cold_qp_small_junction_only": True,
            "qp_upward_rate_neglected": True,
            "interqubit_coupling_derived": False,
            "quasistatic_noise_is_lindblad": False,
        },
    }


# =============================================================================
# 8. ALGÈBRE À QUATRE QUBITS
# =============================================================================

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)
SP = SM.conj().T
N_OP = np.diag([0, 1]).astype(complex)
IDENTITY_16 = np.eye(16, dtype=complex)


def kron_all(operators: Iterable[np.ndarray]) -> np.ndarray:
    """Produit de Kronecker ordonné de gauche à droite."""
    result = np.array([[1.0]], dtype=complex)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def local(operator: np.ndarray, index: int) -> np.ndarray:
    """Opérateur local sur le qubit indexé de 0 à 3."""
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index doit être un entier.")
    if not 0 <= int(index) < 4:
        raise ValueError("index doit appartenir à {0,1,2,3}.")
    factors = [I2, I2, I2, I2]
    factors[int(index)] = operator
    return kron_all(factors)


def stabilizers() -> tuple[np.ndarray, np.ndarray]:
    """Générateurs S_Z=ZZZZ et S_X=XXXX."""
    return kron_all([Z] * 4), kron_all([X] * 4)


def projectors() -> dict[tuple[int, int], np.ndarray]:
    """Projecteurs P(z,x)=(I+z S_Z)(I+x S_X)/4."""
    sz, sx = stabilizers()
    return {
        (z, x): ((IDENTITY_16 + z * sz) @ (IDENTITY_16 + x * sx)) / 4.0
        for z in (1, -1) for x in (1, -1)
    }


def ghz_ket() -> np.ndarray:
    """Vecteur GHZ à quatre qubits."""
    ket = np.zeros(16, dtype=complex)
    ket[0] = 1.0 / math.sqrt(2.0)
    ket[-1] = 1.0 / math.sqrt(2.0)
    return ket


def ghz_density() -> np.ndarray:
    """Matrice densité GHZ."""
    ket = ghz_ket()
    return np.outer(ket, ket.conj())


def measure(rho: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Branches non normalisées d'une mesure projective idéale."""
    return {key: p @ rho @ p for key, p in projectors().items()}


def syndrome_probabilities(rho: np.ndarray) -> dict[str, float]:
    """Probabilités des quatre secteurs de syndrome."""
    branches = measure(rho)
    return {
        f"Z{z:+d}_X{x:+d}": float(np.trace(branch).real)
        for (z, x), branch in branches.items()
    }


def expectation(operator: np.ndarray, rho: np.ndarray) -> float:
    """Valeur moyenne réelle d'un observable hermitien."""
    return float(np.trace(operator @ rho).real)


# =============================================================================
# 9. CONTRÔLES D'ÉTAT DENSITÉ
# =============================================================================

def density_diagnostics(rho: np.ndarray) -> dict[str, float]:
    """Mesures numériques sans lever d'exception."""
    herm = (rho + rho.conj().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(herm)
    tr = np.trace(rho)
    return {
        "trace_real": float(tr.real),
        "trace_imag": float(tr.imag),
        "hermiticity_error": hermiticity_error(rho),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "purity": float(np.trace(rho @ rho).real),
    }


def density_check(rho: np.ndarray, tolerance: float,
                  normalized: bool = True) -> None:
    """Vérifie finitude, hermiticité, positivité et poids de trace."""
    tol = positive_number("tolerance", tolerance)
    if rho.shape != (16, 16):
        raise AssertionError("Une matrice densité 4-qubits doit être 16x16.")
    if not np.all(np.isfinite(rho)):
        raise AssertionError("État contenant NaN ou inf.")
    if hermiticity_error(rho) > tol:
        raise AssertionError("Hermiticité perdue.")
    tr = np.trace(rho)
    if abs(tr.imag) > tol:
        raise AssertionError("Trace complexe.")
    eig_min = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)[0]
    if eig_min < -tol:
        raise AssertionError(f"Positivité perdue : lambda_min={eig_min}.")
    if normalized:
        if abs(tr.real - 1.0) > tol:
            raise AssertionError(f"Trace non conservée : {tr.real}.")
    elif tr.real < -tol or tr.real > 1.0 + tol:
        raise AssertionError(f"Poids de branche invalide : {tr.real}.")


# =============================================================================
# 10. GÉNÉRATEUR DE LINDBLAD À QUATRE QUBITS
# =============================================================================

def dissipator_superoperator(collapse: sp.csr_matrix,
                             identity: sp.csr_matrix) -> sp.csr_matrix:
    """
    Super-opérateur D[L] pour vectorisation colonne :
      vec(L rho L†) = (L* kron L) vec(rho).
    """
    product = collapse.conj().T @ collapse
    return (
        sp.kron(collapse.conj(), collapse, format="csr")
        - 0.5 * sp.kron(identity, product, format="csr")
        - 0.5 * sp.kron(product.T, identity, format="csr")
    )


def liouvillian_4q(f01_GHz: float, J_Hz: float,
                   gamma_down: float,
                   gamma_up: float) -> sp.csr_matrix:
    """
    Générateur de Lindblad 4-qubits, dimension 256x256.

    Chaque qubit possède son propre canal descendant et montant.
    J_Hz est signé. Le Hamiltonien est exprimé en rad/s.
    """
    frequency = positive_number("f01_GHz", f01_GHz)
    coupling = finite_number("J_Hz", J_Hz)
    down = nonnegative_number("gamma_down", gamma_down)
    up = nonnegative_number("gamma_up", gamma_up)

    hamiltonian = np.zeros((16, 16), dtype=complex)
    omega01 = ghz_to_angular(frequency)
    for qubit in range(4):
        hamiltonian += omega01 * local(N_OP, qubit)
    for qubit in range(3):
        hamiltonian += (
            TWO_PI * coupling
            * (local(X, qubit) @ local(X, qubit + 1))
        )

    h = sp.csr_matrix(hamiltonian)
    identity = sp.eye(16, format="csr", dtype=complex)
    generator = -1j * (
        sp.kron(identity, h, format="csr")
        - sp.kron(h.T, identity, format="csr")
    )

    for qubit in range(4):
        for rate, local_operator in ((down, SM), (up, SP)):
            if rate == 0.0:
                continue
            collapse = sp.csr_matrix(
                math.sqrt(rate) * local(local_operator, qubit)
            )
            generator += dissipator_superoperator(collapse, identity)

    return generator.tocsr()


def liouvillian_trace(generator: sp.spmatrix) -> complex:
    """Trace matricielle du super-opérateur."""
    return complex(generator.diagonal().sum())


def liouvillian_trace_preservation_error(
        generator: sp.csr_matrix) -> float:
    """
    Teste la condition <I|L=0 en vectorisation colonne.
    Le vecteur ligne associé à la trace est vec(I)^T.
    """
    trace_row = np.eye(16, dtype=complex).reshape(-1, order="F")
    residual = trace_row @ generator
    return float(np.linalg.norm(residual))


def liouvillian_hermiticity_preservation_probe(
        generator: sp.csr_matrix, seed: int = 7) -> float:
    """Sonde aléatoire : L(rho) doit être hermitien pour rho hermitien."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    rho = a + a.conj().T
    evolved = generator @ rho.reshape(-1, order="F")
    out = evolved.reshape((16, 16), order="F")
    return hermiticity_error(out)


# =============================================================================
# 11. MESURE RÉPÉTÉE ET DYNAMIQUE À QUATRE QUBITS
# =============================================================================

def propagate_density(generator_dt: sp.csr_matrix,
                      rho: np.ndarray,
                      kept: np.ndarray,
                      trace_a: complex | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Propage simultanément la densité inconditionnelle et la branche gardée."""
    columns = np.column_stack((
        rho.reshape(-1, order="F"),
        kept.reshape(-1, order="F"),
    ))
    if trace_a is None:
        trace_a = liouvillian_trace(generator_dt)
    evolved = expm_multiply(generator_dt, columns, traceA=trace_a)
    rho_out = evolved[:, 0].reshape((16, 16), order="F")
    kept_out = evolved[:, 1].reshape((16, 16), order="F")
    return rho_out, kept_out


def apply_stabilizer_measurement(rho: np.ndarray,
                                 kept: np.ndarray,
                                 selected: tuple[int, int] = (1, 1)
                                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mesure non sélective pour rho et sélective pour kept."""
    branches = measure(rho)
    probabilities = {
        f"Z{z:+d}_X{x:+d}": float(np.trace(branch).real)
        for (z, x), branch in branches.items()
    }
    rho_after = sum(branches.values(), np.zeros((16, 16), complex))
    p_selected = projectors()[selected]
    kept_after = p_selected @ kept @ p_selected
    return rho_after, kept_after, probabilities


def simuler_quatre_qubits(params: FluxoniumParams,
                          phi_ext: float,
                          J_Hz: float,
                          steps: int,
                          dt_s: float,
                          tolerance: float = 1.0e-9) -> dict:
    """
    Simule l'évolution Lindblad et une mesure stabilisatrice après chaque pas.

    rho :
        état inconditionnel, mesure non sélective.

    kept :
        branche non normalisée ayant obtenu (++) à chaque mesure.

    La fidélité rapportée est conditionnelle à la survie de cette branche.
    Aucune opération de récupération n'est appliquée.
    """
    dt = positive_number("dt_s", dt_s)
    nsteps = integer_at_least("steps", steps, 1)
    finite_number("J_Hz", J_Hz)
    tol = positive_number("tolerance", tolerance)

    solver = FluxoniumSolver(params)
    obs = analyser_point_flux(solver, phi_ext)
    effective = exporter_qubit_effectif(solver, phi_ext)
    gamma_down = obs["Gamma_down_s_inv"]
    gamma_up = obs["Gamma_up_s_inv"]

    generator_dt = liouvillian_4q(
        obs["f01_GHz"], J_Hz, gamma_down, gamma_up
    ) * dt
    trace_a = liouvillian_trace(generator_dt)

    rho = ghz_density()
    kept = rho.copy()
    initial = rho.copy()
    history = []
    sz, sx = stabilizers()

    for step in range(1, nsteps + 1):
        pre_measure, kept_pre = propagate_density(
            generator_dt, rho, kept, trace_a
        )
        density_check(pre_measure, tol, normalized=True)
        rho, kept, probabilities = apply_stabilizer_measurement(
            pre_measure, kept_pre, (1, 1)
        )
        density_check(rho, tol, normalized=True)
        density_check(kept, tol, normalized=False)

        survival = max(0.0, float(np.trace(kept).real))
        fidelity = None
        if survival > 1.0e-15:
            fidelity = float(np.trace(initial @ kept).real / survival)

        history.append({
            "step": step,
            "time_s": step * dt,
            "pre_measure_ZZZZ": expectation(sz, pre_measure),
            "pre_measure_XXXX": expectation(sx, pre_measure),
            "syndrome_probabilities": probabilities,
            "survival_all_plus": survival,
            "GHZ_fidelity_conditioned_all_plus": fidelity,
            "unconditional_purity": float(np.trace(rho @ rho).real),
            "kept_trace_imag": float(np.trace(kept).imag),
        })

    return {
        "params": asdict(params),
        "point": obs,
        "effective_qubit": effective,
        "four_qubit_model": {
            "J_Hz": float(J_Hz),
            "steps": nsteps,
            "dt_s": dt,
            "gamma_down_s_inv": gamma_down,
            "gamma_up_s_inv": gamma_up,
            "code": "[[4,2,2]] detector code",
            "measurement": "ideal projective stabilizer measurement",
            "postselection": "(++) at every step",
            "error_recovery": False,
            "local_channels_per_qubit": True,
        },
        "diagnostics": {
            "liouvillian_dimension": 256,
            "liouvillian_trace_preservation_error":
                liouvillian_trace_preservation_error(
                    liouvillian_4q(obs["f01_GHz"], J_Hz,
                                   gamma_down, gamma_up)),
        },
        "history": history,
    }


# =============================================================================
# 12. DIAGNOSTICS DE CONVERGENCE
# =============================================================================

def _spectral_record(solver: FluxoniumSolver,
                     phi_ext: float) -> dict[str, float]:
    """Sous-routine de convergence spectrale."""
    geo = solver.geometrie_spectrale(phi_ext)
    energies, vectors = geo["energies"], geo["vectors"]
    qp = abs(np.vdot(vectors[:, 0],
                     solver.operateur_qp(phi_ext) @ vectors[:, 1]))
    return {
        "f01_GHz": float(energies[1] - energies[0]),
        "f12_GHz": float(energies[2] - energies[1]),
        "g_FS_per_rad2": float(geo["g_FS_per_rad2"]),
        "qp01": float(qp),
    }


def diagnostics_convergence(params: FluxoniumParams,
                            phi_ext: float) -> dict:
    """Étudie la sensibilité à N_basis, padding et pas de différence finie."""
    basis_records = []
    for n in (params.N_basis, params.N_basis + 10,
              params.N_basis + 20):
        p = replace(params, N_basis=n)
        solver = FluxoniumSolver(p)
        basis_records.append({
            "N_basis": n,
            **_spectral_record(solver, phi_ext),
        })

    padding_records = []
    for padding in (params.operator_padding,
                    params.operator_padding + 20,
                    params.operator_padding + 40):
        p = replace(params, operator_padding=padding)
        solver = FluxoniumSolver(p)
        padding_records.append({
            "operator_padding": padding,
            **_spectral_record(solver, phi_ext),
        })

    solver = FluxoniumSolver(params)
    finite_difference_records = []
    for eps in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5):
        finite_difference_records.append({
            "eps_rad": eps,
            "g_FS_fd_per_rad2":
                solver.metrique_difference_finie(phi_ext, eps),
        })

    return {
        "basis": basis_records,
        "operator_padding": padding_records,
        "finite_difference": finite_difference_records,
    }


def convergence_summary(records: list[dict],
                        key: str) -> dict[str, float | bool]:
    """Compare première et dernière valeur d'une série."""
    if len(records) < 2:
        raise ValueError("Au moins deux points sont requis.")
    first = float(records[0][key])
    last = float(records[-1][key])
    return {
        "first": first,
        "last": last,
        "relative_change": relative_error(first, last),
        "monotonicity_not_assumed": True,
    }


# =============================================================================
# 13. TESTS STRUCTURELS DU FLUXONIUM
# =============================================================================

def test_hamiltonian_hermiticity(params: FluxoniumParams) -> None:
    """Le Hamiltonien doit être hermitien."""
    solver = FluxoniumSolver(params)
    h = solver.hamiltonien(math.pi)
    np.testing.assert_allclose(h, h.conj().T, atol=1.0e-12, rtol=0.0)


def test_harmonic_limit(params: FluxoniumParams) -> None:
    """Pour EJ=0, le spectre est celui de l'oscillateur LC."""
    harmonic = FluxoniumSolver(replace(params, EJ=0.0))
    energies, _ = harmonic.resoudre_spectre(0.37)
    expected = harmonic.f_lc_GHz * np.arange(params.N_levels)
    np.testing.assert_allclose(energies, expected, atol=1.0e-10, rtol=0.0)


def test_flux_symmetry(params: FluxoniumParams) -> None:
    """Le spectre est symétrique autour de phi_ext=pi."""
    solver = FluxoniumSolver(params)
    ep, _ = solver.resoudre_spectre(math.pi + 0.03)
    em, _ = solver.resoudre_spectre(math.pi - 0.03)
    np.testing.assert_allclose(ep, em, atol=1.0e-9, rtol=1.0e-9)


def test_spectral_derivative(params: FluxoniumParams) -> None:
    """Compare la dérivée analytique du gap à une différence centrée."""
    solver = FluxoniumSolver(params)
    phase, eps = math.pi + 0.02, 1.0e-4
    geo = solver.geometrie_spectrale(phase)
    ep, _ = solver.resoudre_spectre(phase + eps)
    em, _ = solver.resoudre_spectre(phase - eps)
    derivative_fd = (ep[1] - em[1]) / (2.0 * eps)
    np.testing.assert_allclose(
        geo["df01_dphi_GHz_per_rad"], derivative_fd,
        rtol=3.0e-4, atol=2.0e-7)


def test_metric_consistency(params: FluxoniumParams) -> None:
    """Compare métrique perturbative et métrique de recouvrement."""
    solver = FluxoniumSolver(params)
    phase = math.pi + 0.02
    analytic = solver.geometrie_spectrale(phase)["g_FS_per_rad2"]
    numeric = solver.metrique_difference_finie(phase, 1.0e-4)
    np.testing.assert_allclose(analytic, numeric,
                               rtol=5.0e-4, atol=2.0e-7)


def test_sweet_spot(params: FluxoniumParams) -> None:
    """La dérivée du gap doit s'annuler à pi par symétrie."""
    solver = FluxoniumSolver(params)
    derivative = solver.geometrie_spectrale(math.pi)[
        "df01_dphi_GHz_per_rad"]
    if abs(derivative) > 2.0e-8:
        raise AssertionError(f"Point doux non vérifié : df/dphi={derivative}.")


def test_qp_operator(params: FluxoniumParams) -> None:
    """L'opérateur QP est hermitien et antiperiodique sous 2pi."""
    solver = FluxoniumSolver(params)
    phase = math.pi + 0.02
    op = solver.operateur_qp(phase)
    np.testing.assert_allclose(op, op.conj().T, atol=1.0e-12)
    np.testing.assert_allclose(
        solver.operateur_qp(phase + TWO_PI), -op,
        atol=1.0e-12, rtol=0.0)


# =============================================================================
# 14. TESTS DES TAUX ET DU BRUIT
# =============================================================================

def test_qp_scaling(params: FluxoniumParams) -> None:
    """Le taux QP est nul à x_qp=0 et linéaire en x_qp."""
    if taux_quasiparticules(replace(params, x_qp=0.0),
                            0.3, 0.2) != 0.0:
        raise AssertionError("Le taux QP doit être nul à x_qp=0.")
    r1 = taux_quasiparticules(replace(params, x_qp=1.0e-8),
                              0.3, 0.2)
    r2 = taux_quasiparticules(replace(params, x_qp=2.0e-8),
                              0.3, 0.2)
    np.testing.assert_allclose(r2, 2.0 * r1, rtol=1.0e-12)


def test_thermal_detailed_balance(params: FluxoniumParams) -> None:
    """Vérifie Gamma_up/Gamma_down = exp(-hf/kT)."""
    rates = taux_dielectriques(params, 0.3, 0.1)
    expected = math.exp(-0.3 / (KB_OVER_H_GHZ_K * params.T))
    np.testing.assert_allclose(
        rates["up_s_inv"] / rates["down_s_inv"],
        expected, rtol=1.0e-12)


def test_zero_temperature(params: FluxoniumParams) -> None:
    """À T=0, le taux thermique montant est nul."""
    rates = taux_dielectriques(replace(params, T=0.0), 0.3, 0.1)
    if rates["up_s_inv"] != 0.0:
        raise AssertionError("Gamma_up doit être nul à T=0.")


def test_ramsey_limits() -> None:
    """Limites analytiques de Ramsey : absence de bruit et T1 seul."""
    if temps_ramsey_1e(0.0, 1.0, 2.0) != math.inf:
        raise AssertionError("Tphi doit être infini sans bruit.")
    gamma = 1.0e5
    np.testing.assert_allclose(
        temps_ramsey_1e(0.0, 1.0, 2.0, gamma),
        2.0 / gamma, rtol=1.0e-12)


# =============================================================================
# 15. TESTS DU CODE [[4,2,2]]
# =============================================================================

def test_stabilizer_algebra() -> None:
    """Vérifie hermiticité, involution et commutation des stabilisateurs."""
    sz, sx = stabilizers()
    np.testing.assert_allclose(sz @ sz, IDENTITY_16, atol=1.0e-12)
    np.testing.assert_allclose(sx @ sx, IDENTITY_16, atol=1.0e-12)
    np.testing.assert_allclose(sz @ sx, sx @ sz, atol=1.0e-12)
    np.testing.assert_allclose(sz, sz.conj().T, atol=1.0e-12)
    np.testing.assert_allclose(sx, sx.conj().T, atol=1.0e-12)


def test_projectors() -> None:
    """Les quatre projecteurs sont orthogonaux et complets."""
    projs = projectors()
    np.testing.assert_allclose(
        sum(projs.values()), IDENTITY_16, atol=1.0e-12)
    for key_a, pa in projs.items():
        np.testing.assert_allclose(pa @ pa, pa, atol=1.0e-12)
        for key_b, pb in projs.items():
            if key_a != key_b:
                np.testing.assert_allclose(pa @ pb, 0.0, atol=1.0e-12)


def test_ghz_sector() -> None:
    """GHZ appartient au secteur de stabilisateur (++)."""
    rho = ghz_density()
    np.testing.assert_allclose(
        np.trace(projectors()[(1, 1)] @ rho), 1.0, atol=1.0e-12)
    for key in ((1, -1), (-1, 1), (-1, -1)):
        np.testing.assert_allclose(
            np.trace(projectors()[key] @ rho), 0.0, atol=1.0e-12)


def test_pauli_syndromes() -> None:
    """Les erreurs X, Z et Y simples produisent les syndromes attendus."""
    rho = ghz_density()
    expected = {X.__repr__(): (-1, 1),
                Z.__repr__(): (1, -1),
                Y.__repr__(): (-1, -1)}
    for operator, syndrome in ((X, (-1, 1)),
                               (Z, (1, -1)),
                               (Y, (-1, -1))):
        for qubit in range(4):
            error = local(operator, qubit)
            corrupted = error @ rho @ error.conj().T
            branches = measure(corrupted)
            np.testing.assert_allclose(
                np.trace(branches[syndrome]), 1.0, atol=1.0e-12)
            np.testing.assert_allclose(
                np.trace(branches[(1, 1)]), 0.0, atol=1.0e-12)
            recovered_by_same_pauli = error @ corrupted @ error.conj().T
            np.testing.assert_allclose(
                recovered_by_same_pauli, rho, atol=1.0e-12)


def test_amplitude_damping_kraus() -> None:
    """Les opérateurs de Kraus locaux satisfont la complétude."""
    probability = 0.2
    for qubit in range(4):
        k0 = local(np.diag([1.0, math.sqrt(1.0-probability)]), qubit)
        k1 = local(math.sqrt(probability) * SM, qubit)
        np.testing.assert_allclose(
            k0.conj().T @ k0 + k1.conj().T @ k1,
            IDENTITY_16, atol=1.0e-12)


# =============================================================================
# 16. TESTS DU GÉNÉRATEUR DE LINDBLAD
# =============================================================================

def test_liouvillian_shape_and_trace() -> None:
    """Dimension 256x256 et conservation infinitésimale de la trace."""
    generator = liouvillian_4q(0.3, 1.2e6, 2.0e3, 1.0e2)
    if generator.shape != (256, 256):
        raise AssertionError("Dimension Liouvillienne incorrecte.")
    error = liouvillian_trace_preservation_error(generator)
    if error > 1.0e-8:
        raise AssertionError(f"Trace non préservée : erreur {error}.")


def test_liouvillian_local_channels() -> None:
    """
    Chaque qubit doit porter son propre canal.
    Sonde les populations excitées initialisées séparément.
    """
    gamma = 2.5
    generator = liouvillian_4q(0.25, 0.0, gamma, 0.0)
    for qubit in range(4):
        ket = np.zeros(16, complex)
        index = 1 << (3 - qubit)
        ket[index] = 1.0
        rho = np.outer(ket, ket.conj())
        derivative = (generator @ rho.reshape(-1, order="F")).reshape(
            (16, 16), order="F")
        expected = -gamma
        observed = float(np.trace(
            local(N_OP, qubit) @ derivative).real)
        np.testing.assert_allclose(observed, expected,
                                   rtol=1.0e-10, atol=1.0e-10)


def test_signed_coupling() -> None:
    """Le couplage XX peut être positif ou négatif."""
    plus = liouvillian_4q(0.3, 1.0e6, 0.0, 0.0)
    minus = liouvillian_4q(0.3, -1.0e6, 0.0, 0.0)
    if plus.shape != minus.shape:
        raise AssertionError("Dimensions différentes pour J signé.")
    if np.linalg.norm((plus - minus).toarray()) == 0.0:
        raise AssertionError("Le signe de J n'a aucun effet sur le générateur.")


# =============================================================================
# 17. TESTS DYNAMIQUES ET EXPORTS
# =============================================================================

def test_noiseless_dynamics(params: FluxoniumParams) -> None:
    """
    Le cas sans bruit utilise Q très grand et x_qp=0.
    Le test ne confond donc plus pertes diélectriques et pertes QP.
    """
    quiet = replace(params, Q_diel=1.0e30, x_qp=0.0, A_phi=0.0)
    solver = FluxoniumSolver(quiet)
    obs = analyser_point_flux(solver, math.pi)
    dt = 1.0 / (obs["f01_GHz"] * GHZ_TO_HZ)
    result = simuler_quatre_qubits(
        quiet, math.pi, J_Hz=0.0, steps=2, dt_s=dt)
    survival = result["history"][-1]["survival_all_plus"]
    np.testing.assert_allclose(survival, 1.0, atol=2.0e-8)


def test_qp_vs_dielectric(params: FluxoniumParams) -> None:
    """
    Compare les mêmes paramètres diélectriques avec x_qp=0 et x_qp>0.
    Le test vérifie une contribution QP additionnelle, pas simplement
    l'existence de pertes totales.
    """
    base = replace(params, x_qp=0.0)
    qp = replace(params, x_qp=1.0e-4)
    solver_base = FluxoniumSolver(base)
    solver_qp = FluxoniumSolver(qp)
    obs_base = analyser_point_flux(solver_base, math.pi)
    obs_qp = analyser_point_flux(solver_qp, math.pi)
    if obs_qp["Gamma_qp_down_s_inv"] <= 0.0:
        raise AssertionError("Le canal QP de test est nul.")
    if obs_qp["Gamma_down_s_inv"] <= obs_base["Gamma_down_s_inv"]:
        raise AssertionError("Les QP n'ajoutent aucune relaxation.")


def test_json_export(params: FluxoniumParams) -> None:
    """Tous les exports doivent être sérialisables en JSON strict."""
    solver = FluxoniumSolver(params)
    payload = exporter_qubit_effectif(solver, math.pi)
    json.dumps(json_compatible(payload), allow_nan=False)


def run_all_tests() -> None:
    """Lance l'ensemble des tests structurels et numériques."""
    params = FluxoniumParams(N_basis=35, operator_padding=35)
    tests = [
        ("Hamiltonien hermitien", lambda: test_hamiltonian_hermiticity(params)),
        ("limite harmonique", lambda: test_harmonic_limit(params)),
        ("symétrie de flux", lambda: test_flux_symmetry(params)),
        ("dérivée spectrale", lambda: test_spectral_derivative(params)),
        ("métrique Fubini–Study", lambda: test_metric_consistency(params)),
        ("point doux", lambda: test_sweet_spot(params)),
        ("opérateur QP", lambda: test_qp_operator(params)),
        ("échelle QP", lambda: test_qp_scaling(params)),
        ("balance thermique", lambda: test_thermal_detailed_balance(params)),
        ("température nulle", lambda: test_zero_temperature(params)),
        ("limites Ramsey", test_ramsey_limits),
        ("algèbre stabilisateurs", test_stabilizer_algebra),
        ("projecteurs", test_projectors),
        ("secteur GHZ", test_ghz_sector),
        ("syndromes Pauli", test_pauli_syndromes),
        ("complétude Kraus", test_amplitude_damping_kraus),
        ("trace Liouvillienne", test_liouvillian_shape_and_trace),
        ("canaux locaux quatre qubits", test_liouvillian_local_channels),
        ("couplage signé", test_signed_coupling),
        ("dynamique sans bruit", lambda: test_noiseless_dynamics(params)),
        ("QP contre diélectrique", lambda: test_qp_vs_dielectric(params)),
        ("export JSON strict", lambda: test_json_export(params)),
    ]
    failures = []
    for name, function in tests:
        try:
            function()
            print(f"[OK] {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"[ÉCHEC] {name}: {exc}")
    if failures:
        details = "; ".join(f"{name}: {exc}" for name, exc in failures)
        raise RuntimeError(f"{len(failures)} test(s) en échec : {details}")
    print(f"{len(tests)} tests réussis.")


# =============================================================================
# 18. BALAYAGES ET RAPPORTS
# =============================================================================

def scan_ej(params: FluxoniumParams, phi_ext: float,
            values: Iterable[float]) -> list[dict]:
    """Balayage de EJ en réutilisant le solveur pré-calculé."""
    solver = FluxoniumSolver(params)
    rows = []
    for value in values:
        ej = nonnegative_number("EJ_scan", value)
        rows.append(analyser_point_flux(solver, phi_ext, ej))
    return rows


def scan_flux(params: FluxoniumParams,
              phases: Iterable[float]) -> list[dict]:
    """Balayage de phase externe."""
    solver = FluxoniumSolver(params)
    return [analyser_point_flux(solver, finite_number("phi", phase))
            for phase in phases]


def make_report(result: dict) -> str:
    """Construit un rapport texte compact à partir des observables."""
    point = result["point"]
    lines = [
        "FLUXONIUM CORE — RAPPORT",
        "=" * 72,
        f"f01 : {point['f01_GHz']:.9g} GHz",
        f"f12 : {point['f12_GHz']:.9g} GHz",
        f"Anharmonicité : {point['anharmonicite_GHz']:.9g} GHz",
        f"Gamma_down : {point['Gamma_down_s_inv']:.9g} s^-1",
        f"Gamma_up : {point['Gamma_up_s_inv']:.9g} s^-1",
        f"T1 total : {point['T1_total_us']:.9g} µs",
        f"Tphi : {point['T_phi_us']:.9g} µs",
        f"T2* : {point['T2_star_us']:.9g} µs",
        f"Métrique FS : {point['metrique_fubini_study']:.9g} rad^-2",
        "",
        "Modèle quatre qubits :",
        f"J : {result['four_qubit_model']['J_Hz']:.9g} Hz",
        f"Pas : {result['four_qubit_model']['steps']}",
        f"dt : {result['four_qubit_model']['dt_s']:.9g} s",
        "Post-sélection : (++), sans correction d'erreur.",
    ]
    return "\n".join(lines)


def export_results(result: dict, output: str | Path) -> Path:
    """Exporte résumé JSON, qubit effectif, CSV et rapport texte."""
    out = Path(output).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "fluxonium_summary.json", result)
    write_json(out / "qubit_effectif.json", result["effective_qubit"])

    rows = result.get("scan_EJ", [result["point"]])
    if rows:
        csv_path = out / "fluxonium_points.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows([json_compatible(row) for row in rows])

    (out / "fluxonium_report.txt").write_text(
        make_report(result) + "\n", encoding="utf-8")
    return out


# =============================================================================
# 19. ARGUMENTS DE LIGNE DE COMMANDE
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Construit l'interface CLI."""
    parser = argparse.ArgumentParser(
        description=("Fluxonium Core : spectre, pertes, quatre qubits "
                     "effectifs et détection [[4,2,2]].")
    )
    parser.add_argument("--EC", type=float, default=1.0)
    parser.add_argument("--EL", type=float, default=0.5)
    parser.add_argument("--EJ", type=float, default=4.0)
    parser.add_argument("--basis", type=int, default=45)
    parser.add_argument("--padding", type=int, default=35)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--phi-ext", type=float, default=math.pi)
    parser.add_argument("--temperature", type=float, default=0.020)
    parser.add_argument("--Q-diel", type=float, default=6.0e5)
    parser.add_argument("--xqp", type=float, default=1.0e-8)
    parser.add_argument("--gap-GHz", type=float, default=45.0)
    parser.add_argument("--A-phi", type=float, default=2.0e-6)
    parser.add_argument("--noise-low-Hz", type=float, default=1.0)
    parser.add_argument("--noise-high-Hz", type=float, default=1.0e3)
    parser.add_argument("--J-Hz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt-ns", type=float, default=100.0)
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def params_from_args(args: argparse.Namespace) -> FluxoniumParams:
    """Construit le dataclass à partir des arguments CLI."""
    return FluxoniumParams(
        EC=args.EC,
        EL=args.EL,
        EJ=args.EJ,
        N_basis=args.basis,
        N_levels=args.levels,
        operator_padding=args.padding,
        Q_diel=args.Q_diel,
        T=args.temperature,
        x_qp=args.xqp,
        gap_GHz=args.gap_GHz,
        A_phi=args.A_phi,
        noise_f_low_Hz=args.noise_low_Hz,
        noise_f_high_Hz=args.noise_high_Hz,
    )


def print_point(point: dict[str, float]) -> None:
    """Affiche les observables calculées."""
    print("OBSERVABLES DU QUBIT")
    print("-" * 72)
    for name, value in point.items():
        if isinstance(value, (float, int)):
            if math.isfinite(float(value)):
                print(f"{name:<42} : {float(value):.10g}")
            else:
                print(f"{name:<42} : inf")
        else:
            print(f"{name:<42} : {value}")


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.test:
        run_all_tests()
        return 0

    params = params_from_args(args)
    result = simuler_quatre_qubits(
        params=params,
        phi_ext=args.phi_ext,
        J_Hz=args.J_Hz,
        steps=args.steps,
        dt_s=args.dt_ns * 1.0e-9,
    )

    if args.scan:
        result["scan_EJ"] = scan_ej(
            params, args.phi_ext, np.linspace(2.0, 10.0, 9))
    if args.convergence:
        result["convergence"] = diagnostics_convergence(
            params, args.phi_ext)

    if not args.quiet:
        print(f"{MODEL_NAME} — spectre GHz, taux s^-1, temps µs")
        print_point(result["point"])
        print()
        print("Tphi et T2* sont des temps à 1/e du modèle quasi statique.")
        print("Ils ne sont pas des taux de Lindblad.")
        print()
        if result["history"]:
            last = result["history"][-1]
            print(f"Survie (++): {last['survival_all_plus']:.9g}")
            print(f"Fidélité conditionnelle : "
                  f"{last['GHZ_fidelity_conditioned_all_plus']}")

    if args.output:
        path = export_results(result, args.output)
        if not args.quiet:
            print(f"Exports écrits dans : {path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interruption utilisateur.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        if os.environ.get("FLUXONIUM_TRACEBACK", "0") == "1":
            traceback.print_exc()
        raise SystemExit(1)


# =============================================================================
# 20. ANNEXE DE RÉFÉRENCE : CONVENTIONS, INTERPRÉTATION ET CRITÈRES D'ÉCHEC
# =============================================================================
#
# Cette annexe est volontairement conservée dans le fichier source afin que
# les conventions numériques restent accessibles lors d'un transfert vers
# un autre environnement ou un dépôt de recherche.
#
# A. HAMILTONIEN
# --------------
# Le Hamiltonien stocké par FluxoniumSolver est H/h en GHz. Il ne contient
# donc pas de facteur 2*pi. Ce facteur est introduit uniquement lorsqu'une
# évolution temporelle est construite :
#
#       U(t) = exp[-i (2*pi*H/h) t].
#
# Le terme LC est diagonal dans la base harmonique choisie. La phase réduite
# phi et la charge réduite n sont sans dimension dans les conventions usuelles
# du circuit. Le commutateur canonique est approximé dans la base tronquée.
#
# B. PROJECTION DES OPÉRATEURS TRIGONOMÉTRIQUES
# ----------------------------------------------
# Les opérateurs cos(phi), sin(phi), cos(phi/2) et sin(phi/2) sont calculés
# dans une base auxiliaire de dimension N_basis + operator_padding. Le bloc
# supérieur gauche des vecteurs propres de phi_aux est utilisé pour projeter
# l'opérateur dans la base physique. Une augmentation de operator_padding
# doit laisser les observables pertinentes invariantes à la précision visée.
#
# C. DÉRIVÉES SPECTRALES
# ----------------------
# Pour un niveau non dégénéré |n>, la dérivée première est
#
#       dE_n/dlambda = <n|dH/dlambda|n>.
#
# La dérivée seconde est
#
#       d²E_n/dlambda² =
#           <n|d²H/dlambda²|n>
#           + 2 sum_{m != n} |<m|dH/dlambda|n>|²/(E_n-E_m).
#
# Le code applique ces formules aux niveaux 0 et 1 puis soustrait les
# dérivées afin d'obtenir celles du gap f01. Près d'une dégénérescence, la
# formule non dégénérée devient mal conditionnée ; le solveur lève alors une
# exception si un écart énergétique devient inférieur au seuil interne.
#
# D. MÉTRIQUE DE FUBINI–STUDY
# ---------------------------
# Pour l'état fondamental, la métrique locale est
#
#       g = sum_{m>0} |<m|dH/dphi_ext|0>|²/(E_m-E_0)².
#
# La comparaison numérique utilise le recouvrement entre états propres aux
# phases phi_ext-eps et phi_ext+eps. La phase globale est alignée avant la
# dérivation. La valeur numérique dépend de eps et de la troncature ; la
# convergence doit être examinée, pas présumée.
#
# E. RELAXATION DIÉLECTRIQUE
# --------------------------
# Le modèle capacitif employé ici est
#
#       Gamma0 = 2*pi*1e9*(16 EC/Q_diel)*|n01|².
#
# L'occupation thermique vaut
#
#       n_th = 1/(exp(h f01/kB T)-1).
#
# Les taux sont Gamma_down=Gamma0(n_th+1) et Gamma_up=Gamma0 n_th.
# Ce modèle ne décrit pas explicitement une dépendance fréquentielle de Q,
# les pertes inductives, les pertes de rayonnement ou les mécanismes
# non markoviens.
#
# F. RELAXATION QUASIPARTICULAIRE
# -------------------------------
# Le taux QP est implémenté pour la petite jonction, avec l'opérateur
# sin[(phi-phi_ext)/2]. Le modèle suppose des quasiparticules froides et une
# hiérarchie énergétique compatible avec l'approximation indiquée dans
# l'en-tête. Aucun taux montant QP n'est ajouté. L'utilisateur doit calibrer
# x_qp et vérifier que l'approximation est adaptée à son dispositif.
#
# G. BRUIT QUASI STATIQUE
# -----------------------
# La variance intégrée de flux réduit est calculée à partir de la PSD
# unilatérale A_phi²/f entre les deux coupures spécifiées. La conversion en
# phase utilise un facteur 2*pi. Le développement de fréquence est tronqué
# au second ordre. Le résultat dépend donc des coupures fréquentielles et
# de la validité du développement local.
#
# H. TEMPS DE COHÉRENCE
# ---------------------
# T_phi est défini par l'enveloppe quasi statique seule à 1/e. T2* inclut
# en plus exp(-Gamma1*t/2). Ces temps ne sont pas assimilés à un taux
# Markovien de déphasage. En particulier, aucun gamma_phi artificiel n'est
# exporté vers le générateur de Lindblad.
#
# I. MODÈLE À QUATRE QUBITS
# -------------------------
# Le modèle réduit utilise quatre copies identiques du qubit effectif. Le
# terme local est proportionnel à |1><1|. Les termes XX relient les voisins
# 0-1, 1-2 et 2-3. Le signe du couplage est conservé. Le couplage est donné
# en entrée et n'est pas dérivé automatiquement du circuit fluxonium.
#
# J. VECTORISATION
# ----------------
# Les matrices densité sont vectorisées en ordre Fortran :
#
#       vec(A rho B) = (B.T kron A) vec(rho).
#
# Pour un opérateur de saut L, le dissipateur est
#
#       L* kron L
#       - 1/2 I kron (L†L)
#       - 1/2 (L†L).T kron I.
#
# La conservation de la trace s'exprime par <I|L=0. Le test
# liouvillian_trace_preservation_error vérifie cette identité numériquement.
#
# K. CANAUX LOCAUX
# ----------------
# Chaque qubit reçoit son propre opérateur de saut descendant et son propre
# opérateur montant. Réutiliser un opérateur local construit sur le qubit 0
# dans une boucle sur les quatre indices serait une erreur : elle ferait
# porter tous les canaux sur le premier qubit. La fonction liouvillian_4q
# reconstruit donc explicitement chaque opérateur local dans la boucle.
#
# L. CODE DÉTECTEUR [[4,2,2]]
# ---------------------------
# Les stabilisateurs mesurés sont ZZZZ et XXXX. Ils commutent et définissent
# quatre secteurs projectifs. Le code est utilisé ici comme détecteur :
# aucune table de récupération, aucun décodage logique et aucune correction
# conditionnelle ne sont appliqués à l'état.
#
# M. POST-SÉLECTION
# -----------------
# La branche kept est non normalisée. Après chaque évolution, elle est
# projetée dans le secteur (++). Sa trace représente la probabilité cumulée
# de conserver ce résultat à toutes les mesures. La fidélité conditionnelle
# est calculée après division par cette trace. Elle est indéfinie si la trace
# devient numériquement nulle.
#
# N. INTERPRÉTATION DES SYNDROMES
# -------------------------------
# Une probabilité de syndrome non nulle peut provenir de l'évolution
# cohérente, des pertes, des mesures idéales ou de leur combinaison. Elle ne
# constitue pas, à elle seule, une signature spécifique de quasiparticules.
# L'attribution d'un mécanisme exige des simulations témoins et une
# comparaison contrôlée des canaux.
#
# O. CRITÈRES DE CONVERGENCE
# --------------------------
# Une valeur calculée ne doit pas être qualifiée de convergée sur la seule
# base d'une dimension de matrice élevée. Il faut comparer au minimum :
#   - le gap f01 lorsque N_basis augmente ;
#   - les éléments de matrice n01 et qp01 ;
#   - la métrique géométrique ;
#   - l'effet de operator_padding sur les opérateurs trigonométriques ;
#   - la stabilité des taux dérivés.
#
# P. CRITÈRES D'ÉCHEC
# -------------------
# Les situations suivantes doivent invalider ou suspendre l'interprétation :
#   - NaN ou inf dans les paramètres physiques ;
#   - gap f01 nul ou négatif ;
#   - quasi-dégénérescence pour la formule perturbative ;
#   - perte d'hermiticité ou de positivité au-delà de la tolérance ;
#   - défaut de conservation de la trace ;
#   - non-convergence sous augmentation de la base ;
#   - paramètres QP utilisés hors du régime d'approximation ;
#   - confusion entre bruit quasi statique et déphasage Lindblad ;
#   - interprétation de la post-sélection comme correction d'erreur.
#
# Q. EXPORTS
# ----------
# fluxonium_summary.json contient paramètres, observables, modèle à quatre
# qubits, diagnostics et historique temporel.
# qubit_effectif.json contient le contrat de transfert vers un simulateur
# externe. Les valeurs infinies sont converties en null dans le JSON strict.
# fluxonium_points.csv contient les observables d'un point ou le balayage EJ.
# fluxonium_report.txt contient un résumé lisible.
#
# R. REPRODUCTIBILITÉ
# -------------------
# Les tests utilisent des valeurs déterministes et ne reposent pas sur un
# tirage aléatoire, sauf la sonde de préservation hermitienne qui fixe sa
# graine. Les tests ne remplacent pas une comparaison à des données
# expérimentales ou à un solveur indépendant.
#
# S. COMMANDES
# ------------
# Compilation :
#       python -m py_compile fluxonium_core.py
#
# Tests :
#       python fluxonium_core.py --test
#
# Simulation et export :
#       python fluxonium_core.py --output resultats
#
# Balayage EJ et convergence :
#       python fluxonium_core.py --scan --convergence --output resultats
#
# Couplage signé :
#       python fluxonium_core.py --J-Hz -1000000
#
# Trace détaillée en cas d'erreur :
#       FLUXONIUM_TRACEBACK=1 python fluxonium_core.py --test
#
# T. NOTE SUR LES UNITÉS
# ----------------------
# La valeur EC=1.0 signifie EC/h=1 GHz, pas une énergie de 1 joule.
# La valeur EJ=4.0 signifie EJ/h=4 GHz. Le gap Δ est également sa valeur
# divisée par h en GHz. Les conversions vers les taux s^-1 incluent les
# facteurs 2*pi appropriés.
#
# U. NOTE SUR LA VALIDATION PHYSIQUE
# ----------------------------------
# Le passage des tests indique la cohérence interne des identités et des
# limites testées. Il ne prouve ni l'exactitude de tous les modèles de pertes,
# ni la fidélité d'une réduction à deux niveaux, ni la validité des paramètres
# pour un circuit réel. Ces validations requièrent des références physiques,
# une calibration et des données du dispositif.
#
# V. EXTENSION DU MODÈLE
# ----------------------
# Toute extension (bain non markovien, pertes inductives, couplage dérivé
# microscopiquement, bruit coloré dynamique, contrôle temporel, lecture
# imparfaite ou correction d'erreur) doit être ajoutée comme canal distinct,
# documenté par ses unités et testé indépendamment. Elle ne doit pas être
# fusionnée silencieusement avec les taux déjà présents.
#
# W. GARDE-FOUS D'INTERPRÉTATION
# ------------------------------
# Une stabilisation numérique n'est pas une preuve de cohérence physique.
# Une métrique géométrique calculée sur un espace de paramètres n'est pas,
# sans hypothèses supplémentaires, une observable directement mesurée.
# Une probabilité de syndrome n'identifie pas un mécanisme microscopique.
# Une fidélité conditionnelle post-sélectionnée ne représente pas la fidélité
# inconditionnelle du dispositif.
#
# X. MAINTENANCE
# --------------
# Lors d'une modification du Hamiltonien, mettre à jour conjointement les
# dérivées analytiques et leurs tests par différences finies.
# Lors d'une modification des conventions de vectorisation, mettre à jour
# simultanément le Hamiltonien Liouvillien, les dissipateurs et le test de
# conservation de la trace.
# Lors d'une modification du code détecteur, vérifier la commutation des
# stabilisateurs, la complétude des projecteurs et les syndromes Pauli.
#
# Y. CONTRAT DE SORTIE
# --------------------
# Les noms de champs JSON sont conservés autant que possible pour faciliter
# l'intégration avec les scripts d'analyse existants. Toute modification
# incompatible doit faire évoluer schema_version et être explicitée.
#
# Z. FIN DE L'ANNEXE
# -----------------
