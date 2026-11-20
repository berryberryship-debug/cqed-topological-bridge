#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUXONIUM CORE — VERSION OPTIMALE
=================================

Un seul fichier autonome pour :

1. Résoudre le Fluxonium.
2. Calculer les taux diélectriques et quasiparticules.
3. Construire quatre qubits effectifs.
4. Simuler la dynamique Lindblad.
5. Mesurer les stabilisateurs du code détecteur [[4,2,2]].
6. Exporter les paramètres pour un pont lourd.

HAMILTONIEN
-----------
H/h =
    4 EC n²
    + EL phi²/2
    - EJ cos(phi - phi_ext).

CONVENTIONS
-----------
EC, EL, EJ, gap_GHz :
    énergies divisées par h, en GHz.

phi_ext :
    phase externe en radians :
        phi_ext = 2*pi*Phi_ext/Phi0.

Dérivées :
    GHz/rad et GHz/rad².

Taux :
    s^-1.

Temps internes :
    secondes.

Base harmonique :
    vecteurs normalisés par vdot, sans facteur de grille.

RELAXATION
-----------
1. Diélectrique capacitif à Q constant :
       Gamma0 = 2*pi*1e9*(16 EC/Q)*|n01|²
       Gamma_down = Gamma0*(n_therm + 1)
       Gamma_up   = Gamma0*n_therm

2. Quasiparticules froides de la petite jonction :
       theta = phi - phi_ext
       A_qp = sin(theta/2)

       Gamma_qp_down =
           (8/pi)*(EJ/hbar)*x_qp
           * sqrt(2*Delta/(h*f01))
           * |<0|A_qp|1>|²

Hypothèses QP :
- quasiparticules froides ;
- énergie typique QP << h*f01 << Delta ;
- petite jonction uniquement ;
- aucun taux montant QP ;
- aucune perte QP de la superinductance.

BRUIT QUASI STATIQUE
---------------------
PSD unilatérale du flux réduit :
    S_flux(f) = A_phi²/f
pour :
    noise_f_low_Hz <= f <= noise_f_high_Hz.

Variance intégrée :
    sigma_flux² = A_phi²*log(f_high/f_low).

La fréquence est développée à l'ordre 2 en phase externe.

T_phi et T2* sont des temps à 1/e du modèle quasi statique.
Ils ne sont pas des constantes de Lindblad.

DYNAMIQUE À QUATRE QUBITS
--------------------------
- Quatre qubits effectifs identiques.
- Couplage XX effectif donné en entrée.
- Générateur de Lindblad constant.
- Propagation par scipy.sparse.linalg.expm_multiply.
- Mesures projectives des stabilisateurs [[4,2,2]].
- Branche post-sélectionnée sur (++).
- Aucune correction d'erreur inconnue n'est annoncée.

Le Hamiltonien local est dans le référentiel du laboratoire.
Une évolution cohérente peut donc modifier XXXX sans dissipation.
Une détection n'est pas automatiquement une signature de QP.

DÉPENDANCES
-----------
numpy
scipy
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply


TWO_PI = 2.0 * math.pi
GHZ_TO_HZ = 1.0e9
KB_OVER_H_GHZ_K = 20.836619123


# ============================================================================
# 1. VALIDATION
# ============================================================================

def finite_number(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} doit être fini.")
    return value


def positive_number(name: str, value: float) -> float:
    value = finite_number(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} doit être strictement positif.")
    return value


def nonnegative_number(name: str, value: float) -> float:
    value = finite_number(name, value)
    if value < 0.0:
        raise ValueError(f"{name} doit être supérieur ou égal à zéro.")
    return value


def integer_at_least(name: str, value: int, minimum: int) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} doit être un entier >= {minimum}.")


def inverse_rate(rate: float) -> float:
    rate = nonnegative_number("rate", rate)
    return math.inf if rate == 0.0 else 1.0 / rate


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ============================================================================
# 2. PARAMÈTRES
# ============================================================================

@dataclass(frozen=True)
class FluxoniumParams:
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
            raise ValueError(
                "La coupure haute doit dépasser la coupure basse."
            )


# ============================================================================
# 3. SOLVEUR FLUXONIUM
# ============================================================================

class FluxoniumSolver:
    """Solveur harmonique sans dépendance vers un autre fichier."""

    def __init__(self, params: FluxoniumParams):
        self.p = params

        n = params.N_basis
        m = n + params.operator_padding

        self.phi_zpf = (2.0 * params.EC / params.EL) ** 0.25
        self.f_lc_GHz = math.sqrt(8.0 * params.EC * params.EL)

        a = np.diag(np.sqrt(np.arange(1, n, dtype=float)), 1)

        self.phi_op = self.phi_zpf * (a + a.T)
        self.n_op = -1j * (a - a.T) / (2.0 * self.phi_zpf)

        self.H_lc = np.diag(
            self.f_lc_GHz * (np.arange(n, dtype=float) + 0.5)
        )

        a_aux = np.diag(np.sqrt(np.arange(1, m, dtype=float)), 1)
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

    def energie_josephson(
        self, EJ_override: float | None = None
    ) -> float:
        value = self.p.EJ if EJ_override is None else EJ_override
        return nonnegative_number("EJ", value)

    def _ej(self, EJ_override: float | None = None) -> float:
        return self.energie_josephson(EJ_override)

    def hamiltonien(
        self,
        phi_ext: float,
        EJ_override: float | None = None,
    ) -> np.ndarray:
        phase = finite_number("phi_ext", phi_ext)
        ej = self.energie_josephson(EJ_override)

        cos_shift = (
            math.cos(phase) * self.cos_phi_op
            + math.sin(phase) * self.sin_phi_op
        )

        return self.H_lc - ej * cos_shift

    def operateur_qp(self, phi_ext: float) -> np.ndarray:
        """sin((phi - phi_ext)/2), petite jonction."""
        phase = finite_number("phi_ext", phi_ext)

        return (
            math.cos(phase / 2.0) * self.sin_half_phi_op
            - math.sin(phase / 2.0) * self.cos_half_phi_op
        )

    def derivees_hamiltonien(
        self,
        phi_ext: float,
        EJ_override: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        phase = finite_number("phi_ext", phi_ext)
        ej = self.energie_josephson(EJ_override)

        sin_shift = (
            math.cos(phase) * self.sin_phi_op
            - math.sin(phase) * self.cos_phi_op
        )

        cos_shift = (
            math.cos(phase) * self.cos_phi_op
            + math.sin(phase) * self.sin_phi_op
        )

        return -ej * sin_shift, ej * cos_shift

    def diagonaliser(
        self,
        phi_ext: float,
        EJ_override: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(
            self.hamiltonien(phi_ext, EJ_override)
        )

    def resoudre_spectre(
        self,
        phi_ext: float,
        EJ_override: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        energies, vectors = self.diagonaliser(phi_ext, EJ_override)
        k = self.p.N_levels

        return energies[:k] - energies[0], vectors[:, :k]

    def geometrie_spectrale(
        self,
        phi_ext: float,
        EJ_override: float | None = None,
    ) -> dict:
        energies, vectors = self.diagonaliser(phi_ext, EJ_override)

        dH, d2H = self.derivees_hamiltonien(phi_ext, EJ_override)

        dH_e = vectors.conj().T @ dH @ vectors
        d2H_e = vectors.conj().T @ d2H @ vectors

        derivatives = []
        curvatures = []

        for state in (0, 1):
            mask = np.arange(len(energies)) != state
            gaps = energies[state] - energies[mask]

            if np.min(np.abs(gaps)) < 1.0e-10:
                raise ArithmeticError(
                    "Quasi-dégénérescence : perturbation non dégénérée "
                    "insuffisante."
                )

            derivatives.append(float(dH_e[state, state].real))

            curvatures.append(
                float(
                    d2H_e[state, state].real
                    + 2.0 * np.sum(
                        np.abs(dH_e[mask, state]) ** 2 / gaps
                    )
                )
            )

        gaps0 = energies[1:] - energies[0]

        metric = float(
            np.sum(np.abs(dH_e[1:, 0]) ** 2 / gaps0**2)
        )

        return {
            "energies": energies,
            "vectors": vectors,
            "df01_dphi_GHz_per_rad":
                derivatives[1] - derivatives[0],
            "d2f01_dphi2_GHz_per_rad2":
                curvatures[1] - curvatures[0],
            "g_FS_per_rad2": metric,
        }

    def metrique_difference_finie(
        self,
        phi_ext: float,
        eps: float = 1.0e-4,
        EJ_override: float | None = None,
    ) -> float:
        positive_number("eps", eps)

        _, vm = self.diagonaliser(phi_ext - eps, EJ_override)
        _, vp = self.diagonaliser(phi_ext + eps, EJ_override)

        u = vm[:, 0].astype(complex)
        v = vp[:, 0].astype(complex)

        overlap = np.vdot(u, v)

        if abs(overlap) < 1.0e-12:
            raise ArithmeticError(
                "eps trop grand pour une métrique locale."
            )

        v *= np.exp(-1j * np.angle(overlap))

        derivative = (v - u) / (2.0 * eps)
        tangent = derivative - u * np.vdot(u, derivative)

        return float(np.vdot(tangent, tangent).real)


# ============================================================================
# 4. TAUX DE RELAXATION
# ============================================================================

def taux_dielectriques(
    params: FluxoniumParams,
    f01_GHz: float,
    n01: float,
) -> dict[str, float]:
    """Taux diélectriques descendant et montant, en s^-1."""
    frequency = positive_number("f01_GHz", f01_GHz)
    element = nonnegative_number("n01", n01)

    gamma0 = (
        TWO_PI
        * GHZ_TO_HZ
        * 16.0
        * params.EC
        / params.Q_diel
        * element**2
    )

    if params.T == 0.0:
        occupation = 0.0
    else:
        x = frequency / (KB_OVER_H_GHZ_K * params.T)

        occupation = (
            0.0 if x > 700.0 else 1.0 / math.expm1(x)
        )

    down = gamma0 * (occupation + 1.0)
    up = gamma0 * occupation

    return {
        "down_s_inv": down,
        "up_s_inv": up,
        "total_s_inv": down + up,
    }


def taux_quasiparticules(
    params: FluxoniumParams,
    f01_GHz: float,
    qp01: float,
    EJ_override: float | None = None,
) -> float:
    """Taux QP descendant, petite jonction, en s^-1."""
    frequency = positive_number("f01_GHz", f01_GHz)
    element = nonnegative_number("qp01", qp01)

    ej = params.EJ if EJ_override is None else EJ_override
    ej = nonnegative_number("EJ", ej)

    if params.x_qp == 0.0 or ej == 0.0 or element == 0.0:
        return 0.0

    EJ_over_hbar = TWO_PI * GHZ_TO_HZ * ej

    return (
        (8.0 / math.pi)
        * EJ_over_hbar
        * params.x_qp
        * math.sqrt(2.0 * params.gap_GHz / frequency)
        * element**2
    )


# ============================================================================
# 5. BRUIT QUASI STATIQUE ET RAMSEY
# ============================================================================

def sigma_flux_phase(params: FluxoniumParams) -> float:
    """Écart-type de phase externe en radians."""
    variance_flux = params.A_phi**2 * math.log(
        params.noise_f_high_Hz / params.noise_f_low_Hz
    )

    return TWO_PI * math.sqrt(variance_flux)


def log_enveloppe_ramsey(
    t_s: float,
    sigma_rad: float,
    df_GHz_rad: float,
    d2f_GHz_rad2: float,
) -> float:
    """Log du module de l'enveloppe quasi statique quadratique."""
    t = nonnegative_number("t_s", t_s)
    sigma = nonnegative_number("sigma_rad", sigma_rad)
    df = finite_number("df_GHz_rad", df_GHz_rad)
    d2f = finite_number("d2f_GHz_rad2", d2f_GHz_rad2)

    a = TWO_PI * GHZ_TO_HZ * df
    b = TWO_PI * GHZ_TO_HZ * d2f

    z = b * sigma**2 * t

    return (
        -0.25 * math.log1p(z * z)
        -0.5 * (a * sigma * t) ** 2 / (1.0 + z * z)
    )


def temps_ramsey_1e(
    sigma_rad: float,
    df_GHz_rad: float,
    d2f_GHz_rad2: float,
    gamma1_s: float = 0.0,
) -> float:
    """Temps où exp(-Gamma1*t/2)*|W(t)| atteint exp(-1)."""
    sigma = nonnegative_number("sigma_rad", sigma_rad)
    df = finite_number("df_GHz_rad", df_GHz_rad)
    d2f = finite_number("d2f_GHz_rad2", d2f_GHz_rad2)
    gamma1 = nonnegative_number("gamma1_s", gamma1_s)

    if gamma1 == 0.0 and (
        sigma == 0.0 or (df == 0.0 and d2f == 0.0)
    ):
        return math.inf

    def residual(t):
        return (
            0.5 * gamma1 * t
            - log_enveloppe_ramsey(t, sigma, df, d2f)
            - 1.0
        )

    lo = 0.0
    hi = 1.0e-9

    for _ in range(200):
        if residual(hi) >= 0.0:
            break
        hi *= 2.0
    else:
        raise ArithmeticError(
            "Impossible d'encadrer le temps de Ramsey."
        )

    for _ in range(100):
        mid = 0.5 * (lo + hi)

        if residual(mid) >= 0.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


# ============================================================================
# 6. OBSERVABLES D'UN QUBIT
# ============================================================================

def analyser_point_flux(
    solver: FluxoniumSolver,
    phi_ext: float = math.pi,
    EJ_override: float | None = None,
    d_phi: float = 1.0e-4,
) -> dict[str, float]:
    positive_number("d_phi", d_phi)

    p = solver.p

    geo = solver.geometrie_spectrale(phi_ext, EJ_override)

    energies = geo["energies"]
    vectors = geo["vectors"]

    f01 = float(energies[1] - energies[0])
    f12 = float(energies[2] - energies[1])

    positive_number("f01_GHz", f01)

    u = vectors[:, 0]
    v = vectors[:, 1]

    n01 = float(abs(np.vdot(u, solver.n_op @ v)))
    phi01 = float(abs(np.vdot(u, solver.phi_op @ v)))

    qp01 = float(
        abs(
            np.vdot(
                u,
                solver.operateur_qp(phi_ext) @ v,
            )
        )
    )

    dielectric = taux_dielectriques(p, f01, n01)

    qp_down = taux_quasiparticules(
        p,
        f01,
        qp01,
        EJ_override,
    )

    gamma_down = dielectric["down_s_inv"] + qp_down
    gamma_up = dielectric["up_s_inv"]
    gamma1 = gamma_down + gamma_up

    df = geo["df01_dphi_GHz_per_rad"]
    d2f = geo["d2f01_dphi2_GHz_per_rad2"]

    sigma = sigma_flux_phase(p)

    tphi = temps_ramsey_1e(sigma, df, d2f)
    t2 = temps_ramsey_1e(sigma, df, d2f, gamma1)

    metric_fd = solver.metrique_difference_finie(
        phi_ext,
        d_phi,
        EJ_override,
    )

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

        "Gamma_diel_down_s_inv": dielectric["down_s_inv"],
        "Gamma_diel_up_s_inv": dielectric["up_s_inv"],
        "Gamma1_diel_s_inv": dielectric["total_s_inv"],

        "Gamma_qp_down_s_inv": qp_down,

        "Gamma_down_s_inv": gamma_down,
        "Gamma_up_s_inv": gamma_up,
        "Gamma1_total_s_inv": gamma1,

        "T1_diel_us":
            inverse_rate(dielectric["total_s_inv"]) * 1.0e6,

        "T1_qp_us": inverse_rate(qp_down) * 1.0e6,
        "T1_total_us": inverse_rate(gamma1) * 1.0e6,

        "T_phi_us": tphi * 1.0e6,
        "T2_star_us": t2 * 1.0e6,

        "df01_dphi_GHz_per_rad": df,
        "d2f01_dphi2_GHz_per_rad2": d2f,
        "sigma_flux_phase_rad": sigma,

        "metrique_fubini_study": metric,
        "metrique_fubini_study_fd": metric_fd,

        "metric_relative_discrepancy":
            abs(metric - metric_fd)
            / max(abs(metric), 1.0e-14),

        "ground_tail_last5":
            float(np.sum(np.abs(u[-5:]) ** 2)),
    }


# ============================================================================
# 7. EXPORT DU QUBIT EFFECTIF
# ============================================================================

def exporter_qubit_effectif(
    solver: FluxoniumSolver,
    phi_ext: float = math.pi,
    EJ_override: float | None = None,
    d_phi: float = 1.0e-4,
) -> dict:
    """
    Renvoie une description sérialisable d'un qubit effectif.

    Les taux sont prêts pour un modèle local de Lindblad :
        sqrt(gamma_down) * sigma_minus
        sqrt(gamma_up) * sigma_plus

    Le bruit quasi statique est transmis séparément.
    """
    obs = analyser_point_flux(
        solver,
        phi_ext,
        EJ_override,
        d_phi,
    )

    return {
        "schema_version": "1.0",

        "units": {
            "frequency": "Hz",
            "angular_frequency": "rad/s",
            "rates": "s^-1",
            "external_phase": "rad",
        },

        "hamiltonian": {
            "f01_Hz": obs["f01_GHz"] * GHZ_TO_HZ,
            "omega01_rad_s":
                TWO_PI * GHZ_TO_HZ * obs["f01_GHz"],
            "anharmonicity_Hz":
                obs["anharmonicite_GHz"] * GHZ_TO_HZ,
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
            "dielectric_down_s_inv":
                obs["Gamma_diel_down_s_inv"],
            "dielectric_up_s_inv":
                obs["Gamma_diel_up_s_inv"],
            "cold_qp_down_s_inv":
                obs["Gamma_qp_down_s_inv"],
        },

        "quasistatic_flux_noise": {
            "sigma_external_phase_rad":
                obs["sigma_flux_phase_rad"],
            "df01_Hz_per_rad":
                obs["df01_dphi_GHz_per_rad"] * GHZ_TO_HZ,
            "d2f01_Hz_per_rad2":
                obs["d2f01_dphi2_GHz_per_rad2"] * GHZ_TO_HZ,
            "psd_convention":
                "one-sided reduced-flux PSD A_phi^2/f",
            "f_low_Hz": solver.p.noise_f_low_Hz,
            "f_high_Hz": solver.p.noise_f_high_Hz,
        },

        "geometry": {
            "g_FS_per_rad2":
                obs["metrique_fubini_study"],
            "metric_relative_discrepancy":
                obs["metric_relative_discrepancy"],
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


# ============================================================================
# 8. OPÉRATEURS À QUATRE QUBITS
# ============================================================================

I2 = np.eye(2, dtype=complex)

X = np.array(
    [[0.0, 1.0], [1.0, 0.0]],
    dtype=complex,
)

Y = np.array(
    [[0.0, -1j], [1j, 0.0]],
    dtype=complex,
)

Z = np.diag([1.0, -1.0]).astype(complex)

SM = np.array(
    [[0.0, 1.0], [0.0, 0.0]],
    dtype=complex,
)

SP = SM.T

N_OP = np.diag([0.0, 1.0]).astype(complex)


def kron_all(operators):
    """Produit tensoriel d'une séquence d'opérateurs."""
    result = np.array([[1.0]], dtype=complex)

    for operator in operators:
        result = np.kron(result, operator)

    return result


def local(operator, index):
    """Place un opérateur sur le qubit index."""
    operators = [I2] * 4
    operators[index] = operator

    return kron_all(operators)


def stabilizers():
    """Stabilisateurs du code détecteur [[4,2,2]]."""
    return kron_all([Z] * 4), kron_all([X] * 4)


def projectors():
    """
    Projecteurs conjoints sur les quatre secteurs de syndrome.

    P(z,x) = (I + z ZZZZ)(I + x XXXX)/4.
    """
    S_Z, S_X = stabilizers()
    identity = np.eye(16, dtype=complex)

    return {
        (z, x): (
            (identity + z * S_Z)
            @ (identity + x * S_X)
        ) / 4.0
        for z in (1, -1)
        for x in (1, -1)
    }


def ghz_density():
    """Matrice densité de (|0000> + |1111>)/sqrt(2)."""
    ket = np.zeros(16, dtype=complex)

    ket[0] = 1.0 / math.sqrt(2.0)
    ket[-1] = 1.0 / math.sqrt(2.0)

    return np.outer(ket, ket.conj())


def measure(rho):
    """Branches non normalisées d'une mesure projective."""
    return {
        key: projector @ rho @ projector
        for key, projector in projectors().items()
    }


# ============================================================================
# 9. CONTRÔLES DE MATRICE DENSITÉ
# ============================================================================

def density_check(rho, tolerance, normalized=True):
    """Vérifie finitude, hermiticité, positivité et trace."""
    if not np.all(np.isfinite(rho)):
        raise AssertionError("État non fini.")

    check(
        np.linalg.norm(rho - rho.conj().T) <= tolerance,
        "Hermiticité perdue.",
    )

    trace = np.trace(rho)

    check(
        abs(trace.imag) <= tolerance,
        "Trace complexe.",
    )

    hermitian_part = (rho + rho.conj().T) / 2.0

    check(
        np.linalg.eigvalsh(hermitian_part)[0] >= -tolerance,
        "Positivité perdue.",
    )

    if normalized:
        check(
            abs(trace.real - 1.0) <= tolerance,
            "Trace non conservée.",
        )
    else:
        check(
            -tolerance <= trace.real <= 1.0 + tolerance,
            "Poids de branche invalide.",
        )


# ============================================================================
# 10. GÉNÉRATEUR DE LINDBLAD
# ============================================================================

def liouvillian_4q(
    f01_GHz: float,
    J_Hz: float,
    gamma_down: float,
    gamma_up: float,
) -> sp.csr_matrix:
    """
    Générateur de Lindblad à quatre qubits.

    Vectorisation par colonnes :
        vec(A rho B) = (B.T kron A) vec(rho).

    Hamiltonien exprimé en rad/s.
    """
    H = np.zeros((16, 16), dtype=complex)

    omega01 = TWO_PI * GHZ_TO_HZ * f01_GHz

    for qubit in range(4):
        H += omega01 * local(N_OP, qubit)

    for qubit in range(3):
        H += (
            TWO_PI
            * J_Hz
            * (
                local(X, qubit)
                @ local(X, qubit + 1)
            )
        )

    H = sp.csr_matrix(H)

    identity = sp.eye(16, format="csr", dtype=complex)

    generator = -1j * (
        sp.kron(identity, H)
        - sp.kron(H.T, identity)
    )

    collapse_down = sp.csr_matrix(
        math.sqrt(gamma_down) * local(SM, 0)
    )

    collapse_up = sp.csr_matrix(
        math.sqrt(gamma_up) * local(SP, 0)
    )

    for qubit in range(4):
        for collapse in (collapse_down, collapse_up):
            if collapse.nnz == 0:
                continue

            product = collapse.conj().T @ collapse

            generator += (
                sp.kron(collapse.conj(), collapse)
                - 0.5 * sp.kron(identity, product)
                - 0.5 * sp.kron(product.T, identity)
            )

    return generator.tocsr()


# ============================================================================
# 11. SIMULATION À QUATRE QUBITS
# ============================================================================

def simuler_quatre_qubits(
    params: FluxoniumParams,
    phi_ext: float,
    J_Hz: float,
    steps: int,
    dt_s: float,
    tolerance: float = 1.0e-9,
) -> dict:
    """
    Simule deux branches.

    rho :
        état inconditionnel après mesure non sélective.

    kept :
        état non normalisé des trajectoires ayant obtenu (++),
        à toutes les mesures précédentes.
    """
    positive_number("dt_s", dt_s)
    integer_at_least("steps", steps, 1)
    nonnegative_number("J_Hz", J_Hz)
    positive_number("tolerance", tolerance)

    solver = FluxoniumSolver(params)

    obs = analyser_point_flux(solver, phi_ext)

    effective = exporter_qubit_effectif(
        solver,
        phi_ext,
    )

    gamma_down = obs["Gamma_down_s_inv"]
    gamma_up = obs["Gamma_up_s_inv"]

    generator = liouvillian_4q(
        obs["f01_GHz"],
        J_Hz,
        gamma_down,
        gamma_up,
    ) * dt_s

    generator_trace = generator.diagonal().sum()

    rho = ghz_density()
    kept = rho.copy()
    initial = rho.copy()

    history = []

    S_Z, S_X = stabilizers()

    for step in range(1, steps + 1):
        columns = np.column_stack(
            [
                rho.reshape(-1, order="F"),
                kept.reshape(-1, order="F"),
            ]
        )

        evolved = expm_multiply(
            generator,
            columns,
            traceA=generator_trace,
        )

        pre_measure = evolved[:, 0].reshape(
            (16, 16),
            order="F",
        )

        kept_pre_measure = evolved[:, 1].reshape(
            (16, 16),
            order="F",
        )

        density_check(pre_measure, tolerance)

        branches = measure(pre_measure)

        probabilities = {
            f"Z{z:+d}_X{x:+d}":
                float(np.trace(branch).real)
            for (z, x), branch in branches.items()
        }

        rho = sum(
            branches.values(),
            np.zeros((16, 16), dtype=complex),
        )

        P_plus = projectors()[(1, 1)]

        kept = (
            P_plus
            @ kept_pre_measure
            @ P_plus
        )

        density_check(rho, tolerance)
        density_check(kept, tolerance, normalized=False)

        survival = float(np.trace(kept).real)

        if survival > 1.0e-15:
            fidelity = float(
                np.trace(initial @ kept).real / survival
            )
        else:
            fidelity = None

        history.append(
            {
                "step": step,
                "time_s": step * dt_s,

                "pre_measure_ZZZZ":
                    float(np.trace(S_Z @ pre_measure).real),

                "pre_measure_XXXX":
                    float(np.trace(S_X @ pre_measure).real),

                "syndrome_probabilities": probabilities,

                "survival_all_plus": survival,

                "GHZ_fidelity_conditioned_all_plus": fidelity,

                "unconditional_purity":
                    float(np.trace(rho @ rho).real),
            }
        )

    return {
        "params": asdict(params),

        "point": obs,

        "effective_qubit": effective,

        "four_qubit_model": {
            "J_Hz": J_Hz,
            "steps": steps,
            "dt_s": dt_s,
            "gamma_down_s_inv": gamma_down,
            "gamma_up_s_inv": gamma_up,
            "code": "[[4,2,2]] detector code",
            "measurement": "ideal projective stabilizer measurement",
            "postselection": "(++)",
            "error_recovery": False,
        },

        "history": history,
    }


# ============================================================================
# 12. DIAGNOSTICS DE CONVERGENCE
# ============================================================================

def diagnostics_convergence(
    params: FluxoniumParams,
    phi_ext: float,
) -> dict:
    def spectral_record(solver):
        geo = solver.geometrie_spectrale(phi_ext)
        energies = geo["energies"]
        vectors = geo["vectors"]

        qp01 = abs(
            np.vdot(
                vectors[:, 0],
                solver.operateur_qp(phi_ext) @ vectors[:, 1],
            )
        )

        return {
            "f01_GHz": float(energies[1] - energies[0]),
            "g_FS_per_rad2": geo["g_FS_per_rad2"],
            "qp01": float(qp01),
        }

    basis_results = []

    for n in (
        params.N_basis,
        params.N_basis + 10,
        params.N_basis + 20,
    ):
        solver = FluxoniumSolver(
            replace(params, N_basis=n)
        )

        basis_results.append(
            {
                "N_basis": n,
                **spectral_record(solver),
            }
        )

    padding_results = []

    for padding in (
        params.operator_padding,
        params.operator_padding + 20,
        params.operator_padding + 40,
    ):
        solver = FluxoniumSolver(
            replace(params, operator_padding=padding)
        )

        padding_results.append(
            {
                "operator_padding": padding,
                **spectral_record(solver),
            }
        )

    solver = FluxoniumSolver(params)

    eps_results = [
        {
            "eps_rad": eps,
            "g_FS_fd_per_rad2":
                solver.metrique_difference_finie(phi_ext, eps),
        }
        for eps in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)
    ]

    return {
        "basis": basis_results,
        "operator_padding": padding_results,
        "finite_difference": eps_results,
    }


# ============================================================================
# 13. TESTS INTÉGRÉS
# ============================================================================

def run_all_tests() -> None:
    """Tests structurels et numériques."""
    params = FluxoniumParams(
        N_basis=35,
        operator_padding=35,
    )

    solver = FluxoniumSolver(params)

    H = solver.hamiltonien(math.pi)

    np.testing.assert_allclose(
        H,
        H.conj().T,
        atol=1.0e-12,
        rtol=0.0,
    )

    harmonic = FluxoniumSolver(replace(params, EJ=0.0))
    energies, _ = harmonic.resoudre_spectre(0.37)

    np.testing.assert_allclose(
        energies,
        harmonic.f_lc_GHz * np.arange(params.N_levels),
        atol=1.0e-11,
    )

    ep, _ = solver.resoudre_spectre(math.pi + 0.03)
    em, _ = solver.resoudre_spectre(math.pi - 0.03)

    np.testing.assert_allclose(
        ep,
        em,
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    phase = math.pi + 0.02
    eps = 1.0e-4

    geo = solver.geometrie_spectrale(phase)

    ep, _ = solver.resoudre_spectre(phase + eps)
    em, _ = solver.resoudre_spectre(phase - eps)

    df_fd = (ep[1] - em[1]) / (2.0 * eps)

    np.testing.assert_allclose(
        geo["df01_dphi_GHz_per_rad"],
        df_fd,
        rtol=2.0e-4,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        geo["g_FS_per_rad2"],
        solver.metrique_difference_finie(phase, eps),
        rtol=2.0e-4,
        atol=1.0e-7,
    )

    sweet = solver.geometrie_spectrale(math.pi)

    if abs(sweet["df01_dphi_GHz_per_rad"]) > 1.0e-8:
        raise AssertionError(
            "Dérivée spectrale non nulle au point doux."
        )

    qp_operator = solver.operateur_qp(phase)

    np.testing.assert_allclose(
        qp_operator,
        qp_operator.conj().T,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        solver.operateur_qp(phase + TWO_PI),
        -qp_operator,
        atol=1.0e-12,
    )

    if taux_quasiparticules(
        replace(params, x_qp=0.0),
        0.3,
        0.2,
    ) != 0.0:
        raise AssertionError(
            "Le taux QP doit être nul à densité nulle."
        )

    rate1 = taux_quasiparticules(
        replace(params, x_qp=1.0e-8),
        0.3,
        0.2,
    )

    rate2 = taux_quasiparticules(
        replace(params, x_qp=2.0e-8),
        0.3,
        0.2,
    )

    np.testing.assert_allclose(
        rate2,
        2.0 * rate1,
        rtol=1.0e-12,
    )

    dielectric = taux_dielectriques(params, 0.3, 0.1)

    np.testing.assert_allclose(
        dielectric["up_s_inv"] / dielectric["down_s_inv"],
        math.exp(
            -0.3 / (KB_OVER_H_GHZ_K * params.T)
        ),
        rtol=1.0e-12,
    )

    zero_temperature = taux_dielectriques(
        replace(params, T=0.0),
        0.3,
        0.1,
    )

    if zero_temperature["up_s_inv"] != 0.0:
        raise AssertionError(
            "Taux montant diélectrique non nul à T=0."
        )

    if temps_ramsey_1e(0.0, 1.0, 2.0) != math.inf:
        raise AssertionError(
            "Déphasage non nul sans bruit."
        )

    gamma1 = 1.0e5

    np.testing.assert_allclose(
        temps_ramsey_1e(0.0, 1.0, 2.0, gamma1),
        2.0 / gamma1,
        rtol=1.0e-12,
    )

    syndrome_projectors = projectors()

    np.testing.assert_allclose(
        sum(syndrome_projectors.values()),
        np.eye(16),
        atol=1.0e-12,
    )

    for projector in syndrome_projectors.values():
        np.testing.assert_allclose(
            projector @ projector,
            projector,
            atol=1.0e-12,
        )

    rho = ghz_density()

    error_cases = (
        (X, (-1, 1)),
        (Z, (1, -1)),
        (Y, (-1, -1)),
    )

    for operator, target_syndrome in error_cases:
        for qubit in range(4):
            error = local(operator, qubit)

            corrupted = error @ rho @ error.conj().T

            branches = measure(corrupted)

            np.testing.assert_allclose(
                np.trace(branches[target_syndrome]),
                1.0,
                atol=1.0e-12,
            )

            np.testing.assert_allclose(
                np.trace(branches[(1, 1)]),
                0.0,
                atol=1.0e-12,
            )

            restored = error @ corrupted @ error.conj().T

            np.testing.assert_allclose(
                restored,
                rho,
                atol=1.0e-12,
            )

    for qubit in range(4):
        probability = 0.2

        K0 = local(
            np.diag(
                [
                    1.0,
                    math.sqrt(1.0 - probability),
                ]
            ),
            qubit,
        )

        K1 = local(
            math.sqrt(probability) * SM,
            qubit,
        )

        np.testing.assert_allclose(
            K0.conj().T @ K0 + K1.conj().T @ K1,
            np.eye(16),
            atol=1.0e-12,
        )

    obs = analyser_point_flux(solver, math.pi)

    dt_period = 1.0 / (obs["f01_GHz"] * GHZ_TO_HZ)

    noiseless = simuler_quatre_qubits(
        replace(params, x_qp=0.0),
        math.pi,
        J_Hz=0.0,
        steps=2,
        dt_s=dt_period,
    )

    np.testing.assert_allclose(
        noiseless["history"][-1]["survival_all_plus"],
        1.0,
        atol=1.0e-9,
    )

    noisy = simuler_quatre_qubits(
        replace(params, x_qp=1.0e-4),
        math.pi,
        J_Hz=0.0,
        steps=2,
        dt_s=dt_period,
    )

    check(
        noisy["history"][-1]["survival_all_plus"] < 1.0,
        "Détection QP absente.",
    )

    exported = exporter_qubit_effectif(
        solver,
        phi_ext=phase,
    )

    json.dumps(exported, allow_nan=False)

    if "gamma_phi_s_inv" in exported["lindblad_local_rates"]:
        raise AssertionError(
            "Le bruit quasi statique ne doit pas devenir "
            "implicitement un taux de Lindblad."
        )

    print(
        "Tests réussis : spectre, convergence locale, QP, "
        "diélectrique, projecteurs, 12 erreurs Pauli, Kraus "
        "et dynamique à quatre qubits."
    )


# ============================================================================
# 14. EXPORT JSON
# ============================================================================

def json_compatible(value):
    """JSON strict : les temps infinis sont encodés par null."""
    if isinstance(value, dict):
        return {
            key: json_compatible(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_compatible(val) for val in value]

    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())

    if isinstance(value, (float, np.floating)):
        return (
            float(value)
            if math.isfinite(float(value))
            else None
        )

    if isinstance(value, np.integer):
        return int(value)

    return value


# ============================================================================
# 15. CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fluxonium Core optimal : spectre, pertes, "
            "quatre qubits effectifs et détection [[4,2,2]]."
        )
    )

    parser.add_argument("--EC", type=float, default=1.0)
    parser.add_argument("--EL", type=float, default=0.5)
    parser.add_argument("--EJ", type=float, default=4.0)

    parser.add_argument("--basis", type=int, default=45)
    parser.add_argument("--padding", type=int, default=35)

    parser.add_argument(
        "--phi-ext",
        type=float,
        default=math.pi,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.020,
    )

    parser.add_argument(
        "--Q-diel",
        type=float,
        default=6.0e5,
    )

    parser.add_argument(
        "--xqp",
        type=float,
        default=1.0e-8,
    )

    parser.add_argument(
        "--gap-GHz",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--A-phi",
        type=float,
        default=2.0e-6,
    )

    parser.add_argument(
        "--noise-low-Hz",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--noise-high-Hz",
        type=float,
        default=1.0e3,
    )

    parser.add_argument(
        "--J-Hz",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--dt-ns",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--convergence",
        action="store_true",
    )

    parser.add_argument(
        "--scan",
        action="store_true",
    )

    parser.add_argument(
        "--test",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.test:
        run_all_tests()
        return

    params = FluxoniumParams(
        EC=args.EC,
        EL=args.EL,
        EJ=args.EJ,
        N_basis=args.basis,
        operator_padding=args.padding,
        Q_diel=args.Q_diel,
        T=args.temperature,
        x_qp=args.xqp,
        gap_GHz=args.gap_GHz,
        A_phi=args.A_phi,
        noise_f_low_Hz=args.noise_low_Hz,
        noise_f_high_Hz=args.noise_high_Hz,
    )

    result = simuler_quatre_qubits(
        params,
        args.phi_ext,
        J_Hz=args.J_Hz,
        steps=args.steps,
        dt_s=args.dt_ns * 1.0e-9,
    )

    print(
        "FLUXONIUM CORE OPTIMAL — "
        "spectre GHz, taux s^-1, temps µs"
    )

    for name, value in result["point"].items():
        print(f"  {name:<38} : {value:.9g}")

    print(
        "T_phi et T2* : temps à 1/e du modèle quasi statique, "
        "pas constantes de Lindblad."
    )

    if args.scan:
        result["scan_EJ"] = [
            analyser_point_flux(
                FluxoniumSolver(params),
                args.phi_ext,
                float(ej),
            )
            for ej in np.linspace(2.0, 10.0, 9)
        ]

    if args.convergence:
        result["convergence"] = diagnostics_convergence(
            params,
            args.phi_ext,
        )

    if args.output:
        out = Path(args.output).expanduser()
        out.mkdir(parents=True, exist_ok=True)

        (out / "fluxonium_summary.json").write_text(
            json.dumps(
                json_compatible(result),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        (out / "qubit_effectif.json").write_text(
            json.dumps(
                result["effective_qubit"],
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        rows = result.get(
            "scan_EJ",
            [result["point"]],
        )

        with (out / "fluxonium_points.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
            )

            writer.writeheader()
            writer.writerows(rows)

        print(f"Exports : {out}")


if __name__ == "__main__":
    main()
