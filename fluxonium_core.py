#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUXONIUM CORE
==============

Solveur NumPy autonome et fournisseur de paramètres effectifs.

Hamiltonien :
    H/h = 4 EC n² + EL phi²/2 - EJ cos(phi - phi_ext).

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

MODÈLE NUMÉRIQUE
----------------
- Partie LC diagonale analytique.
- Fonctions de phase calculées dans une base auxiliaire.
- Projection dans la base de travail.
- Dérivées énergétiques par perturbation non dégénérée.
- Métrique FS par somme spectrale.
- Vérification indépendante par différences finies.

RELAXATION
-----------
1. Pertes diélectriques capacitives à Q constant.
   Taux descendant et montant séparés.

2. Quasiparticules froides de la petite jonction :
       theta = phi - phi_ext
       A_qp = sin(theta/2)

       Gamma_qp_down =
           (8/pi)*(EJ/hbar)*x_qp
           * sqrt(2*Delta/(h*f01))
           * |<0|A_qp|1>|²

   Hypothèses :
   - quasiparticules froides ;
   - énergie typique QP << h*f01 << Delta ;
   - aucun taux montant QP ;
   - aucune perte QP de la superinductance.

DÉPHASAGE
----------
Approximation gaussienne quasi statique.

PSD unilatérale du flux réduit :
    S_flux(f) = A_phi²/f
pour :
    noise_f_low_Hz <= f <= noise_f_high_Hz.

Variance intégrée :
    sigma_flux² = A_phi²*log(f_high/f_low).

La fréquence est développée à l'ordre 2 en phase externe.

Les temps T_phi et T2* sont définis à une enveloppe de 1/e.
Ils ne sont pas des constantes de Lindblad.

INTERFACE PUBLIQUE
------------------
exporter_qubit_effectif(...)

Cette fonction fournit des nombres et dictionnaires simples :
- fréquence ;
- éléments de matrice ;
- taux descendants et montants ;
- paramètres de bruit quasi statique ;
- indications de troncature.

Aucun import depuis V52, le pont lourd ou src/.

Dépendance :
    numpy.
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


TWO_PI = 2.0 * math.pi
GHZ_TO_HZ = 1.0e9
KB_OVER_H_GHZ_K = 20.836619123


# ============================================================================
# 1. VALIDATION
# ============================================================================

def finite_number(name: str, value: float) -> float:
    """Convertit en float et rejette NaN et inf."""
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
    """Temps en secondes ; inf si le taux est nul."""
    rate = nonnegative_number("rate", rate)
    return math.inf if rate == 0.0 else 1.0 / rate


# ============================================================================
# 2. PARAMÈTRES
# ============================================================================

@dataclass(frozen=True)
class FluxoniumParams:
    EC: float = 1.0
    EL: float = 0.5
    EJ: float = 5.0

    N_basis: int = 45
    N_levels: int = 4
    operator_padding: int = 35

    Q_diel: float = 6.0e5
    T: float = 0.020

    x_qp: float = 0.0
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
        integer_at_least(
            "operator_padding",
            self.operator_padding,
            0,
        )

        if self.N_levels > self.N_basis:
            raise ValueError(
                "N_levels ne peut pas dépasser N_basis."
            )

        positive_number("Q_diel", self.Q_diel)
        nonnegative_number("T", self.T)

        nonnegative_number("x_qp", self.x_qp)
        positive_number("gap_GHz", self.gap_GHz)

        nonnegative_number("A_phi", self.A_phi)

        positive_number(
            "noise_f_low_Hz",
            self.noise_f_low_Hz,
        )
        positive_number(
            "noise_f_high_Hz",
            self.noise_f_high_Hz,
        )

        if self.noise_f_high_Hz <= self.noise_f_low_Hz:
            raise ValueError(
                "La coupure haute doit dépasser la coupure basse."
            )


# ============================================================================
# 3. SOLVEUR SPECTRAL
# ============================================================================

class FluxoniumSolver:
    """Solveur harmonique sans dépendance vers les autres modules."""

    def __init__(self, params: FluxoniumParams):
        self.p = params

        n = params.N_basis
        m = n + params.operator_padding

        self.phi_zpf = (
            2.0 * params.EC / params.EL
        ) ** 0.25

        self.f_lc_GHz = math.sqrt(
            8.0 * params.EC * params.EL
        )

        a = np.diag(
            np.sqrt(np.arange(1, n, dtype=float)),
            1,
        )

        self.phi_op = self.phi_zpf * (a + a.T)

        self.n_op = (
            -1j * (a - a.T)
            / (2.0 * self.phi_zpf)
        )

        self.H_lc = np.diag(
            self.f_lc_GHz
            * (np.arange(n, dtype=float) + 0.5)
        )

        a_aux = np.diag(
            np.sqrt(np.arange(1, m, dtype=float)),
            1,
        )

        phi_aux = self.phi_zpf * (a_aux + a_aux.T)

        values, vectors = np.linalg.eigh(phi_aux)
        projected = vectors[:n, :]

        self.cos_phi_op = (
            projected * np.cos(values)
        ) @ projected.T

        self.sin_phi_op = (
            projected * np.sin(values)
        ) @ projected.T

        self.cos_half_phi_op = (
            projected * np.cos(0.5 * values)
        ) @ projected.T

        self.sin_half_phi_op = (
            projected * np.sin(0.5 * values)
        ) @ projected.T

    def energie_josephson(
        self,
        EJ_override: float | None = None,
    ) -> float:
        """Énergie EJ effective en GHz."""
        value = self.p.EJ if EJ_override is None else EJ_override
        return nonnegative_number("EJ", value)

    def _ej(self, EJ_override: float | None) -> float:
        """Alias conservé pour le code précédent."""
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
        """
        sin((phi - phi_ext)/2), petite jonction.

        La phase n'est pas réduite modulo 2*pi ici.
        """
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
        energies, vectors = self.diagonaliser(
            phi_ext,
            EJ_override,
        )

        k = self.p.N_levels

        return (
            energies[:k] - energies[0],
            vectors[:, :k],
        )

    def geometrie_spectrale(
        self,
        phi_ext: float,
        EJ_override: float | None = None,
    ) -> dict:
        energies, vectors = self.diagonaliser(
            phi_ext,
            EJ_override,
        )

        dH, d2H = self.derivees_hamiltonien(
            phi_ext,
            EJ_override,
        )

        dH_e = vectors.conj().T @ dH @ vectors
        d2H_e = vectors.conj().T @ d2H @ vectors

        derivatives = []
        curvatures = []

        for state in (0, 1):
            mask = np.arange(len(energies)) != state
            gaps = energies[state] - energies[mask]

            if np.min(np.abs(gaps)) < 1.0e-10:
                raise ArithmeticError(
                    "Quasi-dégénérescence : "
                    "perturbation non dégénérée insuffisante."
                )

            derivatives.append(
                float(dH_e[state, state].real)
            )

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
            np.sum(
                np.abs(dH_e[1:, 0]) ** 2 / gaps0**2
            )
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

        _, vm = self.diagonaliser(
            phi_ext - eps,
            EJ_override,
        )

        _, vp = self.diagonaliser(
            phi_ext + eps,
            EJ_override,
        )

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
    """
    Taux capacitif descendant et montant, en s^-1.

    gamma0 :
        taux descendant à température nulle.

    gamma_down = gamma0*(n_therm + 1)
    gamma_up   = gamma0*n_therm

    gamma_down + gamma_up :
        taux de relaxation de la population vers son équilibre.
    """
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
        x = frequency / (
            KB_OVER_H_GHZ_K * params.T
        )

        occupation = (
            0.0
            if x > 700.0
            else 1.0 / math.expm1(x)
        )

    down = gamma0 * (occupation + 1.0)
    up = gamma0 * occupation

    return {
        "down_s_inv": down,
        "up_s_inv": up,
        "total_s_inv": down + up,
    }


def taux_dielectrique(
    params: FluxoniumParams,
    f01_GHz: float,
    n01: float,
) -> float:
    """Interface historique : taux diélectrique total."""
    return taux_dielectriques(
        params,
        f01_GHz,
        n01,
    )["total_s_inv"]


def taux_quasiparticules(
    params: FluxoniumParams,
    f01_GHz: float,
    qp01: float,
    EJ_override: float | None = None,
) -> float:
    """
    Taux QP descendant, petite jonction, en s^-1.

    Approximation de quasiparticules froides.
    Aucun taux montant ni plancher artificiel.
    """
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
        * math.sqrt(
            2.0 * params.gap_GHz / frequency
        )
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
    """
    Log du module de l'enveloppe quasi statique quadratique.

    delta ~ normale(0, sigma_rad²)
    delta_omega = a*delta + b*delta²/2.

    Résultat exact pour cette distribution et ce développement.
    """
    t = nonnegative_number("t_s", t_s)
    sigma = nonnegative_number("sigma_rad", sigma_rad)
    df = finite_number("df_GHz_rad", df_GHz_rad)
    d2f = finite_number("d2f_GHz_rad2", d2f_GHz_rad2)

    a = TWO_PI * GHZ_TO_HZ * df
    b = TWO_PI * GHZ_TO_HZ * d2f

    z = b * sigma**2 * t

    return (
        -0.25 * math.log1p(z * z)
        -0.5 * (a * sigma * t) ** 2
        / (1.0 + z * z)
    )


def temps_ramsey_1e(
    sigma_rad: float,
    df_GHz_rad: float,
    d2f_GHz_rad2: float,
    gamma1_s: float = 0.0,
) -> float:
    """
    Temps où exp(-Gamma1*t/2)*|W(t)| atteint exp(-1).

    Gamma1 = Gamma_down + Gamma_up.
    """
    sigma = nonnegative_number("sigma_rad", sigma_rad)
    df = finite_number("df_GHz_rad", df_GHz_rad)
    d2f = finite_number("d2f_GHz_rad2", d2f_GHz_rad2)
    gamma1 = nonnegative_number("gamma1_s", gamma1_s)

    if gamma1 == 0.0 and (
        sigma == 0.0
        or (df == 0.0 and d2f == 0.0)
    ):
        return math.inf

    def residual(t):
        return (
            0.5 * gamma1 * t
            - log_enveloppe_ramsey(
                t,
                sigma,
                df,
                d2f,
            )
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
# 6. OBSERVABLES
# ============================================================================

def analyser_point_flux(
    solver: FluxoniumSolver,
    phi_ext: float = math.pi,
    EJ_override: float | None = None,
    d_phi: float = 1.0e-4,
) -> dict[str, float]:
    positive_number("d_phi", d_phi)

    p = solver.p

    geo = solver.geometrie_spectrale(
        phi_ext,
        EJ_override,
    )

    energies = geo["energies"]
    vectors = geo["vectors"]

    f01 = float(energies[1] - energies[0])
    f12 = float(energies[2] - energies[1])

    positive_number("f01_GHz", f01)

    u = vectors[:, 0]
    v = vectors[:, 1]

    n01 = float(
        abs(np.vdot(u, solver.n_op @ v))
    )

    phi01 = float(
        abs(np.vdot(u, solver.phi_op @ v))
    )

    qp01 = float(
        abs(
            np.vdot(
                u,
                solver.operateur_qp(phi_ext) @ v,
            )
        )
    )

    dielectric = taux_dielectriques(
        p,
        f01,
        n01,
    )

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

    tphi = temps_ramsey_1e(
        sigma,
        df,
        d2f,
    )

    t2 = temps_ramsey_1e(
        sigma,
        df,
        d2f,
        gamma1,
    )

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

        "Gamma_diel_down_s_inv":
            dielectric["down_s_inv"],

        "Gamma_diel_up_s_inv":
            dielectric["up_s_inv"],

        "Gamma1_diel_s_inv":
            dielectric["total_s_inv"],

        "Gamma_qp_down_s_inv": qp_down,

        "Gamma_down_s_inv": gamma_down,
        "Gamma_up_s_inv": gamma_up,
        "Gamma1_total_s_inv": gamma1,

        "T1_diel_us":
            inverse_rate(dielectric["total_s_inv"]) * 1.0e6,

        "T1_qp_us":
            inverse_rate(qp_down) * 1.0e6,

        "T1_total_us":
            inverse_rate(gamma1) * 1.0e6,

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
# 7. INTERFACE VERS V52 ET LE PONT
# ============================================================================

def exporter_qubit_effectif(
    solver: FluxoniumSolver,
    phi_ext: float = math.pi,
    EJ_override: float | None = None,
    d_phi: float = 1.0e-4,
) -> dict:
    """
    Renvoie une description sérialisable du qubit effectif.

    Taux utilisables dans un modèle local de Lindblad :
        sqrt(gamma_down) * sigma_minus
        sqrt(gamma_up)   * sigma_plus

    Le bruit quasi statique est fourni séparément.
    Aucun gamma_phi markovien n'est déduit de T_phi_us.

    Les couplages inter-qubits ne sont pas calculés ici.
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
# 8. DIAGNOSTICS DE CONVERGENCE
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
            "f01_GHz":
                float(energies[1] - energies[0]),
            "g_FS_per_rad2":
                geo["g_FS_per_rad2"],
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

        basis_results.append({
            "N_basis": n,
            **spectral_record(solver),
        })

    padding_results = []

    for padding in (
        params.operator_padding,
        params.operator_padding + 20,
        params.operator_padding + 40,
    ):
        solver = FluxoniumSolver(
            replace(
                params,
                operator_padding=padding,
            )
        )

        padding_results.append({
            "operator_padding": padding,
            **spectral_record(solver),
        })

    solver = FluxoniumSolver(params)

    eps_results = [
        {
            "eps_rad": eps,
            "g_FS_fd_per_rad2":
                solver.metrique_difference_finie(
                    phi_ext,
                    eps,
                ),
        }
        for eps in (
            1.0e-2,
            1.0e-3,
            1.0e-4,
            1.0e-5,
        )
    ]

    return {
        "basis": basis_results,
        "operator_padding": padding_results,
        "finite_difference": eps_results,
    }


# ============================================================================
# 9. TESTS INTÉGRÉS
# ============================================================================

def run_all_tests() -> None:
    """Tests structurels ; aucune validation expérimentale annoncée."""
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

    harmonic = FluxoniumSolver(
        replace(params, EJ=0.0)
    )

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

    dielectric = taux_dielectriques(
        params,
        0.3,
        0.1,
    )

    np.testing.assert_allclose(
        dielectric["up_s_inv"]
        / dielectric["down_s_inv"],
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
        temps_ramsey_1e(
            0.0,
            1.0,
            2.0,
            gamma1,
        ),
        2.0 / gamma1,
        rtol=1.0e-12,
    )

    exported = exporter_qubit_effectif(
        solver,
        phi_ext=phase,
    )

    json.dumps(
        exported,
        allow_nan=False,
    )

    if "gamma_phi_s_inv" in exported["lindblad_local_rates"]:
        raise AssertionError(
            "Le bruit quasi statique ne doit pas devenir "
            "implicitement un taux de Lindblad."
        )

    print("Tests structurels terminés avec succès.")


# ============================================================================
# 10. EXPORT JSON
# ============================================================================

def json_compatible(value):
    """JSON strict : les temps infinis sont encodés par null."""
    if isinstance(value, dict):
        return {
            key: json_compatible(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_compatible(val)
            for val in value
        ]

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
# 11. CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fluxonium NumPy : spectre, géométrie, pertes "
            "et export de qubit effectif."
        )
    )

    parser.add_argument("--EC", type=float, default=1.0)
    parser.add_argument("--EL", type=float, default=0.5)
    parser.add_argument("--EJ", type=float, default=5.0)

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
        default=0.0,
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

    solver = FluxoniumSolver(params)

    obs = analyser_point_flux(
        solver,
        args.phi_ext,
    )

    effective = exporter_qubit_effectif(
        solver,
        args.phi_ext,
    )

    result = {
        "params": asdict(params),

        "noise_model":
            "Gaussian quasistatic, quadratic frequency expansion",

        "T1_channels": [
            "capacitive_dielectric",
            "cold_qp_small_junction",
        ],

        "quasiparticle_model_implemented": True,

        "point": obs,

        "effective_qubit": effective,
    }

    print(
        "FLUXONIUM CORE — "
        "spectre GHz, taux s^-1, temps µs"
    )

    for name, value in obs.items():
        print(f"  {name:<38} : {value:.9g}")

    print(
        "T_phi et T2* : temps à 1/e du modèle quasi statique, "
        "pas constantes de Lindblad."
    )

    if args.scan:
        result["scan_EJ"] = [
            analyser_point_flux(
                solver,
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
        out.mkdir(
            parents=True,
            exist_ok=True,
        )

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
                effective,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        rows = result.get(
            "scan_EJ",
            [obs],
        )

        with (out / "fluxonium_points.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(obs),
            )

            writer.writeheader()
            writer.writerows(rows)

        print(f"Exports : {out}")


if __name__ == "__main__":
    main()
