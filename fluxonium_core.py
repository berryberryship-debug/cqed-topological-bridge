#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUXONIUM CORE
==============

Solveur autonome NumPy pour :
    H/h = 4 EC n² + EL phi²/2 - EJ cos(phi - phi_ext).

Conventions
-----------
EC, EL, EJ et les valeurs propres : GHz = énergie / h.
phi_ext : phase externe en radians, 2*pi*Phi_ext/Phi0.
Dérivées spectrales : GHz/rad et GHz/rad².
Taux : s^-1. Temps internes : secondes.
Les vecteurs en base de Fock sont normalisés avec vdot, sans facteur de grille.

Modèle numérique
----------------
- Base harmonique LC.
- Partie LC diagonale analytique.
- Fonctions trigonométriques calculées dans une base auxiliaire,
  puis projetées sur la base de travail.
- Dérivées énergétiques par perturbation non dégénérée.
- Métrique FS par somme spectrale dans la base finie.
- Vérifications indépendantes par différences finies.

Bruit
-----
Relaxation : modèle diélectrique capacitif à Q constant.
Aucun modèle de pertes par quasiparticules n'est implémenté ici.

Déphasage : approximation gaussienne quasi statique.
PSD unilatérale déclarée :
    S_f(f) = A_phi² / f, pour f_low <= f <= f_high,
où f désigne ici la fréquence du bruit en Hz et le flux est en Phi0.

La variance intégrée vaut :
    sigma_flux² = A_phi² * log(f_high/f_low).

La fréquence du qubit est développée à l'ordre 2 en phase externe.
La cohérence de Ramsey résultante n'est généralement pas exponentielle.
T_phi et T2* sont définis par une enveloppe atteignant exp(-1).

Limites
-------
Le modèle quasi statique ne remplace pas une intégration complète
par fonctions de filtre. Les coupures de bruit sont des entrées.
La convergence en taille de base et en dimension auxiliaire doit
être vérifiée pour le domaine de paramètres étudié.

Dépendance : numpy.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
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


def integer_at_least(name: str, value: int, minimum: int) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} doit être un entier >= {minimum}.")


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
    A_phi: float = 2.0e-6
    noise_f_low_Hz: float = 1.0
    noise_f_high_Hz: float = 1.0e3

    def __post_init__(self) -> None:
        positive_number("EC", self.EC)
        positive_number("EL", self.EL)
        if finite_number("EJ", self.EJ) < 0.0:
            raise ValueError("EJ doit être >= 0.")
        integer_at_least("N_basis", self.N_basis, 10)
        integer_at_least("N_levels", self.N_levels, 3)
        integer_at_least("operator_padding", self.operator_padding, 0)
        if self.N_levels > self.N_basis:
            raise ValueError("N_levels ne peut pas dépasser N_basis.")
        positive_number("Q_diel", self.Q_diel)
        if finite_number("T", self.T) < 0.0:
            raise ValueError("T doit être >= 0.")
        if finite_number("A_phi", self.A_phi) < 0.0:
            raise ValueError("A_phi doit être >= 0.")
        positive_number("noise_f_low_Hz", self.noise_f_low_Hz)
        positive_number("noise_f_high_Hz", self.noise_f_high_Hz)
        if self.noise_f_high_Hz <= self.noise_f_low_Hz:
            raise ValueError("La coupure haute doit dépasser la coupure basse.")


class FluxoniumSolver:
    """Solveur spectral sans dépendance vers les autres fichiers du projet."""

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
        vals, vecs = np.linalg.eigh(phi_aux)
        projected_vecs = vecs[:n, :]

        self.cos_phi_op = (
            projected_vecs * np.cos(vals)
        ) @ projected_vecs.T
        self.sin_phi_op = (
            projected_vecs * np.sin(vals)
        ) @ projected_vecs.T

    def _ej(self, EJ_override: float | None) -> float:
        ej = self.p.EJ if EJ_override is None else float(EJ_override)
        if finite_number("EJ", ej) < 0.0:
            raise ValueError("EJ doit être >= 0.")
        return ej

    def hamiltonien(
        self, phi_ext: float, EJ_override: float | None = None
    ) -> np.ndarray:
        phase = finite_number("phi_ext", phi_ext)
        ej = self._ej(EJ_override)
        cos_shift = (
            math.cos(phase) * self.cos_phi_op
            + math.sin(phase) * self.sin_phi_op
        )
        return self.H_lc - ej * cos_shift

    def derivees_hamiltonien(
        self, phi_ext: float, EJ_override: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        phase = finite_number("phi_ext", phi_ext)
        ej = self._ej(EJ_override)
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
        self, phi_ext: float, EJ_override: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(self.hamiltonien(phi_ext, EJ_override))

    def resoudre_spectre(
        self, phi_ext: float, EJ_override: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        energies, vectors = self.diagonaliser(phi_ext, EJ_override)
        k = self.p.N_levels
        return energies[:k] - energies[0], vectors[:, :k]

    def geometrie_spectrale(
        self, phi_ext: float, EJ_override: float | None = None
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
                    "insuffisante pour ce point."
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
            "df01_dphi_GHz_per_rad": derivatives[1] - derivatives[0],
            "d2f01_dphi2_GHz_per_rad2": curvatures[1] - curvatures[0],
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
            raise ArithmeticError("eps trop grand pour une métrique locale.")
        v *= np.exp(-1j * np.angle(overlap))
        derivative = (v - u) / (2.0 * eps)
        tangent = derivative - u * np.vdot(u, derivative)
        return float(np.vdot(tangent, tangent).real)


def taux_dielectrique(
    params: FluxoniumParams, f01_GHz: float, n01: float
) -> float:
    """
    Taux total descendant + montant, en s^-1.

    Modèle de perte capacitive à Q constant :
        Gamma1 = 2*pi*1e9 * (16 EC / Q) * |n01|²
                 * coth[h f01/(2 kB T)].
    """
    positive_number("f01_GHz", f01_GHz)
    if params.T == 0.0:
        thermal = 1.0
    else:
        x = f01_GHz / (KB_OVER_H_GHZ_K * params.T)
        thermal = 1.0 / math.tanh(0.5 * x)
    return (
        TWO_PI * GHZ_TO_HZ
        * 16.0 * params.EC / params.Q_diel
        * n01**2 * thermal
    )


def sigma_flux_phase(params: FluxoniumParams) -> float:
    """Écart-type de phase externe en radians, PSD unilatérale A²/f."""
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
    Log du module de <exp[-i (a delta + b delta²/2) t]>.

    delta est gaussien de variance sigma_rad².
    a et b sont les dérivées de pulsation, en unités SI.
    Résultat exact dans ce modèle quasi statique quadratique.
    """
    t = finite_number("t_s", t_s)
    if t < 0.0:
        raise ValueError("t_s doit être >= 0.")
    a = TWO_PI * GHZ_TO_HZ * df_GHz_rad
    b = TWO_PI * GHZ_TO_HZ * d2f_GHz_rad2
    z = b * sigma_rad**2 * t
    return (
        -0.25 * math.log1p(z * z)
        -0.5 * (a * sigma_rad * t) ** 2 / (1.0 + z * z)
    )


def temps_ramsey_1e(
    sigma_rad: float,
    df_GHz_rad: float,
    d2f_GHz_rad2: float,
    gamma1_s: float = 0.0,
) -> float:
    """Temps où exp(-Gamma1*t/2)*|W(t)| atteint exp(-1)."""
    if gamma1_s == 0.0 and (
        sigma_rad == 0.0
        or (df_GHz_rad == 0.0 and d2f_GHz_rad2 == 0.0)
    ):
        return math.inf

    def residual(t: float) -> float:
        return (
            0.5 * gamma1_s * t
            - log_enveloppe_ramsey(
                t, sigma_rad, df_GHz_rad, d2f_GHz_rad2
            )
            - 1.0
        )

    lo, hi = 0.0, 1.0e-9
    for _ in range(200):
        if residual(hi) >= 0.0:
            break
        hi *= 2.0
    else:
        raise ArithmeticError("Impossible d'encadrer le temps de Ramsey.")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if residual(mid) >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def analyser_point_flux(
    solver: FluxoniumSolver,
    phi_ext: float = math.pi,
    EJ_override: float | None = None,
    d_phi: float = 1.0e-4,
) -> dict[str, float]:
    positive_number("d_phi", d_phi)
    p = solver.p
    geo = solver.geometrie_spectrale(phi_ext, EJ_override)
    energies, vectors = geo["energies"], geo["vectors"]
    f01 = float(energies[1] - energies[0])
    f12 = float(energies[2] - energies[1])
    u, v = vectors[:, 0], vectors[:, 1]

    n01 = float(abs(np.vdot(u, solver.n_op @ v)))
    phi01 = float(abs(np.vdot(u, solver.phi_op @ v)))
    gamma1 = taux_dielectrique(p, f01, n01)
    t1 = 1.0 / gamma1 if gamma1 > 0.0 else math.inf

    df = geo["df01_dphi_GHz_per_rad"]
    d2f = geo["d2f01_dphi2_GHz_per_rad2"]
    sigma = sigma_flux_phase(p)
    tphi = temps_ramsey_1e(sigma, df, d2f)
    t2 = temps_ramsey_1e(sigma, df, d2f, gamma1)

    metric_fd = solver.metrique_difference_finie(
        phi_ext, d_phi, EJ_override
    )
    metric = geo["g_FS_per_rad2"]

    return {
        "EJ_GHz": solver._ej(EJ_override),
        "phi_ext_rad": float(phi_ext),
        "phi_ext_over_pi": float(phi_ext / math.pi),
        "f01_GHz": f01,
        "f12_GHz": f12,
        "anharmonicite_GHz": f12 - f01,
        "anharmonicite_rel": (f12 - f01) / f01,
        "n01": n01,
        "phi01": phi01,
        "Gamma1_diel_s_inv": gamma1,
        "T1_diel_us": t1 * 1.0e6,
        "T1_total_us": t1 * 1.0e6,
        "T_phi_us": tphi * 1.0e6,
        "T2_star_us": t2 * 1.0e6,
        "df01_dphi_GHz_per_rad": df,
        "d2f01_dphi2_GHz_per_rad2": d2f,
        "sigma_flux_phase_rad": sigma,
        "metrique_fubini_study": metric,
        "metrique_fubini_study_fd": metric_fd,
        "metric_relative_discrepancy": abs(metric - metric_fd)
        / max(abs(metric), 1.0e-14),
        "ground_tail_last5": float(np.sum(np.abs(u[-5:]) ** 2)),
    }


def diagnostics_convergence(
    params: FluxoniumParams, phi_ext: float
) -> dict:
    sizes = sorted(set([
        params.N_basis,
        params.N_basis + 10,
        params.N_basis + 20,
    ]))
    basis_results = []
    for n in sizes:
        solver = FluxoniumSolver(replace(params, N_basis=n))
        geo = solver.geometrie_spectrale(phi_ext)
        energies = geo["energies"]
        basis_results.append({
            "N_basis": n,
            "f01_GHz": float(energies[1] - energies[0]),
            "g_FS_per_rad2": geo["g_FS_per_rad2"],
        })

    padding_results = []
    for padding in (
        params.operator_padding,
        params.operator_padding + 20,
        params.operator_padding + 40,
    ):
        solver = FluxoniumSolver(
            replace(params, operator_padding=padding)
        )
        geo = solver.geometrie_spectrale(phi_ext)
        energies = geo["energies"]
        padding_results.append({
            "operator_padding": padding,
            "f01_GHz": float(energies[1] - energies[0]),
            "g_FS_per_rad2": geo["g_FS_per_rad2"],
        })

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


def run_all_tests() -> None:
    """Tests structurels ; ne remplacent pas une validation expérimentale."""
    p = FluxoniumParams(N_basis=35, operator_padding=35)
    solver = FluxoniumSolver(p)
    H = solver.hamiltonien(math.pi)
    if not np.allclose(H, H.conj().T, atol=1.0e-12, rtol=0.0):
        raise AssertionError("Hamiltonien non hermitien.")

    harmonic = FluxoniumSolver(replace(p, EJ=0.0))
    energies, _ = harmonic.resoudre_spectre(0.37)
    expected = harmonic.f_lc_GHz * np.arange(p.N_levels)
    np.testing.assert_allclose(energies, expected, atol=1.0e-11)

    ep, _ = solver.resoudre_spectre(math.pi + 0.03)
    em, _ = solver.resoudre_spectre(math.pi - 0.03)
    np.testing.assert_allclose(ep, em, atol=1.0e-10, rtol=1.0e-10)

    phase = math.pi + 0.02
    geo = solver.geometrie_spectrale(phase)
    eps = 1.0e-4
    ep, _ = solver.resoudre_spectre(phase + eps)
    em, _ = solver.resoudre_spectre(phase - eps)
    df_fd = (ep[1] - em[1]) / (2.0 * eps)
    np.testing.assert_allclose(
        geo["df01_dphi_GHz_per_rad"], df_fd,
        rtol=2.0e-4, atol=1.0e-7,
    )
    np.testing.assert_allclose(
        geo["g_FS_per_rad2"],
        solver.metrique_difference_finie(phase, eps),
        rtol=2.0e-4, atol=1.0e-7,
    )

    sweet = solver.geometrie_spectrale(math.pi)
    if abs(sweet["df01_dphi_GHz_per_rad"]) > 1.0e-8:
        raise AssertionError("Dérivée spectrale non nulle au point doux.")

    if temps_ramsey_1e(0.0, 1.0, 2.0) != math.inf:
        raise AssertionError("Déphasage non nul sans bruit.")
    gamma1 = 1.0e5
    np.testing.assert_allclose(
        temps_ramsey_1e(0.0, 1.0, 2.0, gamma1),
        2.0 / gamma1,
        rtol=1.0e-12,
    )

    # Aucun test n'impose un minimum de g_FS au point doux :
    # une fréquence stationnaire n'implique pas un état stationnaire.
    print("Tests structurels terminés avec succès.")


def json_compatible(value):
    """JSON strict : les temps infinis sont encodés par null."""
    if isinstance(value, dict):
        return {key: json_compatible(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(val) for val in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fluxonium autonome NumPy : spectre et diagnostics."
    )
    parser.add_argument("--EC", type=float, default=1.0)
    parser.add_argument("--EL", type=float, default=0.5)
    parser.add_argument("--EJ", type=float, default=5.0)
    parser.add_argument("--basis", type=int, default=45)
    parser.add_argument("--padding", type=int, default=35)
    parser.add_argument("--phi-ext", type=float, default=math.pi)
    parser.add_argument("--temperature", type=float, default=0.020)
    parser.add_argument("--Q-diel", type=float, default=6.0e5)
    parser.add_argument("--A-phi", type=float, default=2.0e-6)
    parser.add_argument("--noise-low-Hz", type=float, default=1.0)
    parser.add_argument("--noise-high-Hz", type=float, default=1.0e3)
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", type=str, default=None)
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
        A_phi=args.A_phi,
        noise_f_low_Hz=args.noise_low_Hz,
        noise_f_high_Hz=args.noise_high_Hz,
    )
    solver = FluxoniumSolver(params)
    obs = analyser_point_flux(solver, args.phi_ext)
    result = {
        "params": asdict(params),
        "noise_model": "Gaussian quasistatic, quadratic frequency expansion",
        "T1_channels": ["capacitive_dielectric"],
        "quasiparticle_model_implemented": False,
        "point": obs,
    }

    print("FLUXONIUM CORE — unités spectrales GHz, temps µs")
    for name, value in obs.items():
        print(f"  {name:<36} : {value:.9g}")
    print(
        "Attention : T_phi et T2* sont des temps à 1/e du modèle "
        "quasi statique, pas des constantes de Lindblad."
    )

    if args.scan:
        result["scan_EJ"] = [
            analyser_point_flux(solver, args.phi_ext, float(ej))
            for ej in np.linspace(2.0, 10.0, 9)
        ]
    if args.convergence:
        result["convergence"] = diagnostics_convergence(
            params, args.phi_ext
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
        rows = result.get("scan_EJ", [obs])
        with (out / "fluxonium_points.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(obs))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exports : {out}")


if __name__ == "__main__":
    main()
