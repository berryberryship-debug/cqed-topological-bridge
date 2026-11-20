#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUXONIUM CORE — Solveur Hamiltonien, Cohérence cQED & Géométrie Quantique
===========================================================================
Modèle physique :
    H = 4 E_C n^2 + (1/2) E_L phi^2 - E_J cos(phi - Phi_ext)

Observables calculés :
  - Spectre d'énergie (niveaux 0..N_levels-1) et fréquences de transition f01, f12
  - Anharmonicité absolue (GHz) et relative
  - Temps de relaxation T1 (pertes diélectriques, quasiparticules jonction & chaîne)
  - Temps de déphasage pur T_phi (bruit de flux 1/f) et cohérence T2*
  - Métrique locale de Fubini-Study g_{Phi,Phi} et phase géométrique de Berry

Dépendance unique : numpy
"""
from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np


# ============================================================================
# 1. PARAMÈTRES MATÉRIELS DU CIRCUIT FLUXONIUM
# ============================================================================

@dataclass(frozen=True)
class FluxoniumParams:
    EC: float = 1.0          # Énergie de charge (GHz)
    EL: float = 0.5          # Énergie inductive de la superinductance (GHz)
    EJ: float = 5.0          # Énergie Josephson (GHz)
    N_basis: int = 60        # Dimension de troncature de l'oscillateur LC
    N_levels: int = 5        # Nombre de niveaux propres extraits

    # Paramètres environnementaux et de dissipation
    Q_diel: float = 6.0e5    # Facteur de qualité diélectrique
    x_qp: float = 1.0e-8     # Densité normalisée de quasiparticules hors équilibre
    Delta: float = 50.0      # Gap supraconducteur de l'aluminium (GHz)
    T: float = 0.020         # Température cryogénique (K)
    A_phi: float = 2.0e-6    # Amplitude du bruit de flux 1/f (en unités de Phi_0)
    kB_h: float = 20.8366    # Constante de Boltzmann k_B / h (GHz / K)


# ============================================================================
# 2. SOLVEUR SPECTRAL & OPÉRATEURS DE BASE
# ============================================================================

class FluxoniumSolver:
    """Résolution exacte de l'Hamiltonien du fluxonium dans la base de Fock LC."""

    def __init__(self, params: FluxoniumParams):
        if params.EC <= 0 or params.EL <= 0 or params.EJ < 0:
            raise ValueError("Paramètres d'énergie invalides (EC > 0, EL > 0, EJ >= 0).")
        if params.N_basis < 10:
            raise ValueError("N_basis doit être >= 10.")

        self.p = params
        self.phi_zpf = (2.0 * params.EC / params.EL) ** 0.25

        # Opérateurs d'annihilation / création dans la base harmonique LC
        a = np.zeros((params.N_basis, params.N_basis), dtype=complex)
        for k in range(1, params.N_basis):
            a[k - 1, k] = math.sqrt(k)
        adag = a.conj().T

        self.phi_op = self.phi_zpf * (a + adag)
        self.n_op = -1j / (2.0 * self.phi_zpf) * (a - adag)

        # Diagonalisation préalable de phi_op pour évaluer cos(phi - Phi_ext) exactement
        self.phi_vals, self.phi_vecs = np.linalg.eigh(self.phi_op)
        self.H_lc = (
            4.0 * params.EC * (self.n_op @ self.n_op)
            + 0.5 * params.EL * (self.phi_op @ self.phi_op)
        )

    def _op_fonction_phi(self, fn) -> np.ndarray:
        vals = fn(self.phi_vals)
        return (self.phi_vecs * vals) @ self.phi_vecs.conj().T

    def resoudre_spectre(self, phi_ext: float, EJ_override: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalise H(Phi_ext) et renvoie (énergies relatives en GHz, vecteurs propres)."""
        EJ = self.p.EJ if EJ_override is None else float(EJ_override)
        cos_op = self._op_fonction_phi(lambda x: np.cos(x - phi_ext))
        H = self.H_lc - EJ * cos_op
        H = 0.5 * (H + H.conj().T)

        evals, evecs = np.linalg.eigh(H)
        k = self.p.N_levels
        return evals[:k] - evals[0], evecs[:, :k]

    def element_matrice(self, op: np.ndarray, evecs: np.ndarray, i: int, j: int) -> complex:
        return complex(np.vdot(evecs[:, i], op @ evecs[:, j]))


# ============================================================================
# 3. OBSERVABLES DE COHÉRENCE & GÉOMÉTRIE QUANTIQUE
# ============================================================================

def analyser_point_flux(
    solver: FluxoniumSolver,
    phi_ext: float = math.pi,
    EJ_override: float | None = None,
    d_phi: float = 1e-3,
) -> dict[str, float]:
    """Calcule l'ensemble des grandeurs spectrales, de cohérence et géométriques en Phi_ext."""
    p = solver.p
    EJ = p.EJ if EJ_override is None else float(EJ_override)

    E0, V0 = solver.resoudre_spectre(phi_ext, EJ)
    Ep, Vp = solver.resoudre_spectre(phi_ext + d_phi, EJ)
    Em, _ = solver.resoudre_spectre(phi_ext - d_phi, EJ)

    f01 = float(E0[1])
    f12 = float(E0[2] - E0[1])
    alpha = f12 - f01
    alpha_rel = alpha / f01 if abs(f01) > 1e-12 else 0.0

    # Sensibilité de fréquence au premier ordre (d omega_01 / d Phi_ext)
    d_omega_d_phi = float((Ep[1] - Em[1]) / (2.0 * d_phi))

    # Éléments de matrice de transition |0> <-> |1>
    n01 = abs(solver.element_matrice(solver.n_op, V0, 0, 1))
    phi01 = abs(solver.element_matrice(solver.phi_op, V0, 0, 1))
    sin_half_op = solver._op_fonction_phi(lambda x: np.sin(0.5 * x))
    sin01 = abs(solver.element_matrice(sin_half_op, V0, 0, 1))

    # 1. Relaxation diélectrique (T1_diel en µs)
    if f01 > 1e-12:
        x = f01 / (p.kB_h * p.T)
        coth = 1.0 / np.tanh(0.5 * x) if x > 1e-8 else 2.0 / x
        gamma_diel = 16.0 * p.EC * (n01 ** 2) / p.Q_diel * coth
        T1_diel = (1.0 / gamma_diel) * 1e-3 if gamma_diel > 0 else 1e9

        # 2. Relaxation par quasiparticules (jonction Josephson + superinductance)
        facteur_qp = math.sqrt(max(0.0, 2.0 * p.Delta / f01))
        boltz = math.exp(-min(700.0, x))
        gamma_qp_jj = (8.0 * EJ / math.pi) * facteur_qp * p.x_qp * (sin01 ** 2) * (1.0 + boltz)
        gamma_qp_el = (8.0 * p.EL / math.pi) * facteur_qp * p.x_qp * ((0.5 * phi01) ** 2) * (1.0 + boltz)

        T1_qp_jj = (1.0 / gamma_qp_jj) * 1e-3 if gamma_qp_jj > 0 else 1e9
        T1_qp_el = (1.0 / gamma_qp_el) * 1e-3 if gamma_qp_el > 0 else 1e9

        gamma_1 = gamma_diel + gamma_qp_jj + gamma_qp_el
        T1_total = (1.0 / gamma_1) * 1e-3 if gamma_1 > 0 else 1e9
    else:
        T1_diel = T1_qp_jj = T1_qp_el = T1_total = 1e9

    # 3. Déphasage pur 1/f (T_phi en µs) et cohérence T2*
    A_rad = 2.0 * math.pi * p.A_phi
    gamma_phi = abs(d_omega_d_phi) * math.sqrt(2.0 * math.log(2.0)) * A_rad
    T_phi = (1.0 / gamma_phi) * 1e-3 if gamma_phi > 1e-12 else 1e6
    T2_star = 1.0 / (0.5 / T1_total + 1.0 / T_phi)

    # 4. Géométrie quantique locale : fidélité et métrique de Fubini-Study g_{Phi,Phi}
    F_loc = float(np.clip(abs(np.vdot(V0[:, 0], Vp[:, 0])) ** 2, 0.0, 1.0))
    g_fubini = max(0.0, (1.0 - F_loc) / (d_phi ** 2))

    return {
        "EJ_GHz": EJ,
        "phi_ext_over_pi": phi_ext / math.pi,
        "f01_GHz": f01,
        "f12_GHz": f12,
        "anharmonicite_GHz": alpha,
        "anharmonicite_rel": alpha_rel,
        "n01": n01,
        "phi01": phi01,
        "sin_half_01": sin01,
        "T1_diel_us": T1_diel,
        "T1_qp_JJ_us": T1_qp_jj,
        "T1_qp_EL_us": T1_qp_el,
        "T1_total_us": T1_total,
        "T_phi_us": T_phi,
        "T2_star_us": T2_star,
        "domega_dphi_GHz_rad": d_omega_d_phi,
        "fidelite_locale": F_loc,
        "metrique_fubini_study": g_fubini,
    }


def phase_berry_flux(
    solver: FluxoniumSolver,
    phi_centre: float = math.pi,
    demi_largeur: float = 0.25 * math.pi,
    n_points: int = 31,
) -> float:
    """Calcule l'holonomie discrète de Pancharatnam-Berry sur une boucle fermée en flux."""
    chemin = np.linspace(phi_centre - demi_largeur, phi_centre + demi_largeur, n_points)
    etats = [solver.resoudre_spectre(float(p))[1][:, 0] for p in chemin]

    phase = 0.0
    for u, v in zip(etats[:-1], etats[1:]):
        ov = np.vdot(u, v)
        if abs(ov) < 1e-14:
            return float("nan")
        phase += np.angle(ov)

    ov_close = np.vdot(etats[-1], etats[0])
    if abs(ov_close) < 1e-14:
        return float("nan")
    phase += np.angle(ov_close)
    return float(np.angle(np.exp(1j * phase)))


# ============================================================================
# 4. BALAYAGE PARAMÉTRIQUE & RAPPORT
# ============================================================================

def balayage_EJ(
    params: FluxoniumParams,
    EJ_min: float = 2.0,
    EJ_max: float = 10.0,
    n_points: int = 9,
    phi_ext: float = math.pi,
) -> list[dict[str, float]]:
    solver = FluxoniumSolver(params)
    lignes = []
    for ej in np.linspace(EJ_min, EJ_max, n_points):
        obs = analyser_point_flux(solver, phi_ext=phi_ext, EJ_override=float(ej))
        lignes.append(obs)
    return lignes


def main():
    parser = argparse.ArgumentParser(description="Fluxonium Core — Solveur cQED & Géométrie Quantique.")
    parser.add_argument("--EC", type=float, default=1.0, help="Énergie de charge EC (GHz).")
    parser.add_argument("--EL", type=float, default=0.5, help="Énergie inductive EL (GHz).")
    parser.add_argument("--EJ", type=float, default=5.0, help="Énergie Josephson nominale EJ (GHz).")
    parser.add_argument("--basis", type=int, default=60, help="Dimension de la base de Fock LC.")
    parser.add_argument("--output", type=str, default=None, help="Dossier optionnel d'export CSV/JSON.")
    args = parser.parse_args()

    params = FluxoniumParams(EC=args.EC, EL=args.EL, EJ=args.EJ, N_basis=args.basis)
    solver = FluxoniumSolver(params)

    obs_sweet = analyser_point_flux(solver, phi_ext=math.pi)
    berry = phase_berry_flux(solver, phi_centre=math.pi)

    print("=" * 84)
    print(f"  FLUXONIUM CORE (EC={params.EC} GHz | EL={params.EL} GHz | EJ={params.EJ} GHz | Basis={params.N_basis})")
    print("=" * 84)
    print("POINT DOUX (Phi_ext = pi) :")
    print(f"  f01 = {obs_sweet['f01_GHz']:.5f} GHz | f12 = {obs_sweet['f12_GHz']:.5f} GHz | alpha = {obs_sweet['anharmonicite_GHz']:+.5f} GHz")
    print(f"  T1_diel = {obs_sweet['T1_diel_us']:.1f} µs | T1_qp_JJ = {obs_sweet['T1_qp_JJ_us']:.1f} µs | T1_total = {obs_sweet['T1_total_us']:.1f} µs")
    print(f"  T_phi = {obs_sweet['T_phi_us']:.1f} µs | T2* = {obs_sweet['T2_star_us']:.1f} µs")
    print(f"  Métrique Fubini-Study g = {obs_sweet['metrique_fubini_study']:.4f} | Phase de Berry = {berry:.5f} rad")
    print("-" * 84)
    print("BALAYAGE EN ÉNERGIE JOSEPHSON EJ (Phi_ext = pi) :")
    print(f"{'EJ (GHz)':<9} | {'f01 (GHz)':<10} | {'alpha (GHz)':<12} | {'T1 (µs)':<9} | {'T2* (µs)':<9} | {'g_FS'}")
    print("-" * 84)

    rows = balayage_EJ(params, EJ_min=2.0, EJ_max=10.0, n_points=9, phi_ext=math.pi)
    for r in rows:
        print(
            f"{r['EJ_GHz']:7.2f}   | {r['f01_GHz']:8.5f}   | {r['anharmonicite_GHz']:+10.5f}   | "
            f"{r['T1_total_us']:7.1f}   | {r['T2_star_us']:7.1f}   | {r['metrique_fubini_study']:8.3f}"
        )
    print("=" * 84)

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fluxonium_summary.json").write_text(
            json.dumps({"params": asdict(params), "sweet_spot": obs_sweet, "berry_phase_rad": berry, "scan_EJ": rows}, indent=2),
            encoding="utf-8",
        )
        with (out / "fluxonium_scan_EJ.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
