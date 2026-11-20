#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUXONIUM CORE (Optimisé Termux / NumPy)
========================================
Modèle Hamiltonien cQED :
    H = 4 E_C n^2 + (1/2) E_L phi^2 - E_J cos(phi - Phi_ext)

Observables :
  - Spectre (f01, f12, anharmonicite alpha)
  - Relaxation T1 (pertes dielectriques + quasiparticules JJ et inductance)
  - Dephasage pur T_phi (sensibilite ordre 1 et ordre 2 au point doux)
  - Coherence totale T2*
  - Metrique quantique locale de Fubini-Study g_{Phi,Phi}
"""
from __future__ import annotations
import os

# Verrouillage des threads pour processeurs ARM (Termux)
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
# 1. PARAMÈTRES PHYSIQUES DU CIRCUIT
# ============================================================================

@dataclass(frozen=True)
class FluxoniumParams:
    EC: float = 1.0          # Énergie de charge (GHz)
    EL: float = 0.5          # Énergie inductive (GHz)
    EJ: float = 5.0          # Énergie Josephson (GHz)
    N_basis: int = 45        # Troncation harmonique LC (40-50 optimal sur Termux)
    N_levels: int = 4        # Niveaux extraits (0, 1, 2, 3)

    # Paramètres de pertes et d'environnement
    Q_diel: float = 6.0e5    # Facteur de qualité diélectrique
    x_qp: float = 1.0e-8     # Fraction normalisée de quasiparticules hors équilibre
    Delta: float = 50.0      # Gap supraconducteur Al (GHz)
    T: float = 0.020         # Température cryogénique (K)
    A_phi: float = 2.0e-6    # Amplitude du bruit de flux 1/f (Phi_0)
    kB_h: float = 20.8366    # k_B / h (GHz / K)


# ============================================================================
# 2. SOLVEUR HAMILTONIEN PRÉ-VECTORISÉ
# ============================================================================

class FluxoniumSolver:
    """Solveur spectral haute performance avec factorisation trigonométrique."""

    def __init__(self, params: FluxoniumParams):
        if params.EC <= 0 or params.EL <= 0 or params.EJ < 0:
            raise ValueError("Paramètres énergétiques invalides.")
        if params.N_basis < 10:
            raise ValueError("N_basis doit être >= 10.")

        self.p = params
        self.phi_zpf = (2.0 * params.EC / params.EL) ** 0.25

        # Opérateurs échelle dans la base de Fock
        diag_a = np.sqrt(np.arange(1, params.N_basis, dtype=float))
        a = np.diag(diag_a, k=1)
        adag = a.T

        self.phi_op = self.phi_zpf * (a + adag)
        self.n_op = -1j / (2.0 * self.phi_zpf) * (a - adag)

        # Base propre de phi pour construire cos(phi) et sin(phi)
        phi_vals, phi_vecs = np.linalg.eigh(self.phi_op)
        
        # Pré-calcul des opérateurs invariants
        self.cos_phi_op = (phi_vecs * np.cos(phi_vals)) @ phi_vecs.conj().T
        self.sin_phi_op = (phi_vecs * np.sin(phi_vals)) @ phi_vecs.conj().T
        self.sin_half_phi_op = (phi_vecs * np.sin(0.5 * phi_vals)) @ phi_vecs.conj().T

        # Hamiltonien LC statique
        self.H_lc = (
            4.0 * params.EC * (self.n_op @ self.n_op)
            + 0.5 * params.EL * (self.phi_op @ self.phi_op)
        )

    def resoudre_spectre(
        self, phi_ext: float, EJ_override: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Diagonalise H(phi_ext) par recomposition trigonométrique instantanée :
        cos(phi - phi_ext) = cos(phi)*cos(phi_ext) + sin(phi)*sin(phi_ext)
        """
        EJ = self.p.EJ if EJ_override is None else float(EJ_override)
        
        c_ext = math.cos(phi_ext)
        s_ext = math.sin(phi_ext)
        cos_shift_op = c_ext * self.cos_phi_op + s_ext * self.sin_phi_op

        H = self.H_lc - EJ * cos_shift_op
        H = 0.5 * (H + H.conj().T)  # Hermiticité numérique stricte

        evals, evecs = np.linalg.eigh(H)
        k = self.p.N_levels
        evals_rel = evals[:k] - evals[0]
        return evals_rel, evecs[:, :k]


# ============================================================================
# 3. EXTRACTION DES OBSERVABLES PHYSIQUES & GÉOMÉTRIE
# ============================================================================

def analyser_point_flux(
    solver: FluxoniumSolver,
    phi_ext: float = math.pi,
    EJ_override: float | None = None,
    d_phi: float = 1e-3,
) -> dict[str, float]:
    p = solver.p
    EJ = p.EJ if EJ_override is None else float(EJ_override)

    # Trois points pour dérivées centrales d'ordre 1 et 2
    E0, V0 = solver.resoudre_spectre(phi_ext, EJ)
    Ep, Vp = solver.resoudre_spectre(phi_ext + d_phi, EJ)
    Em, _ = solver.resoudre_spectre(phi_ext - d_phi, EJ)

    f01 = float(E0[1])
    f12 = float(E0[2] - E0[1]) if len(E0) > 2 else 0.0
    alpha = f12 - f01
    alpha_rel = alpha / f01 if abs(f01) > 1e-9 else 0.0

    # Dérivées de fréquence par rapport au flux externe
    d_omega_d_phi = float((Ep[1] - Em[1]) / (2.0 * d_phi))
    d2_omega_d_phi2 = float((Ep[1] - 2.0 * E0[1] + Em[1]) / (d_phi ** 2))

    # Éléments de matrice de transition
    n01 = abs(complex(np.vdot(V0[:, 0], solver.n_op @ V0[:, 1])))
    phi01 = abs(complex(np.vdot(V0[:, 0], solver.phi_op @ V0[:, 1])))
    sin01 = abs(complex(np.vdot(V0[:, 0], solver.sin_half_phi_op @ V0[:, 1])))

    # --- 1. RELAXATION T1 ---
    if f01 > 1e-6:
        x = f01 / (p.kB_h * p.T)
        coth = 1.0 / np.tanh(0.5 * x) if x < 50.0 else 1.0
        
        # Pertes diélectriques capacitives
        gamma_diel = 16.0 * p.EC * (n01 ** 2) / p.Q_diel * coth

        # Quasiparticules non appariées
        facteur_qp = math.sqrt(max(0.0, 2.0 * p.Delta / f01))
        boltz = math.exp(-min(500.0, x))
        gamma_qp_jj = (8.0 * EJ / math.pi) * facteur_qp * p.x_qp * (sin01 ** 2) * (1.0 + boltz)
        gamma_qp_el = (8.0 * p.EL / math.pi) * facteur_qp * p.x_qp * ((0.5 * phi01) ** 2) * (1.0 + boltz)

        gamma_1 = gamma_diel + gamma_qp_jj + gamma_qp_el
        T1_diel = (1.0 / gamma_diel) * 1e-3 if gamma_diel > 0 else 1e6
        T1_total = (1.0 / gamma_1) * 1e-3 if gamma_1 > 0 else 1e6
    else:
        T1_diel = T1_total = 1e6

    # --- 2. DÉPHASAGE PUR T_phi (Ordre 1 + Ordre 2 au point doux) ---
    A_rad = 2.0 * math.pi * p.A_phi
    
    # Contribution d'ordre 1
    gamma_phi_1 = abs(d_omega_d_phi) * math.sqrt(2.0 * math.log(2.0)) * A_rad
    
    # Contribution résiduelle d'ordre 2 au point doux (sweet spot)
    # avec coupures ultraviolette/infrarouge log(w_ir/w_uv) ~ 10
    gamma_phi_2 = 0.5 * abs(d2_omega_d_phi2) * (A_rad ** 2) * math.sqrt(2.0 * math.log(2.0) * 10.0)
    
    gamma_phi = gamma_phi_1 + gamma_phi_2
    T_phi = (1.0 / gamma_phi) * 1e-3 if gamma_phi > 1e-12 else 1e6
    
    # Cohérence de Ramsey T2*
    taux_2_star = (0.5 / T1_total) + (1.0 / T_phi)
    T2_star = 1.0 / taux_2_star if taux_2_star > 0 else 1e6

    # --- 3. MÉTRIQUE DE FUBINI-STUDY g_{Phi,Phi} ---
    recouvrement = abs(np.vdot(V0[:, 0], Vp[:, 0])) ** 2
    F_loc = float(np.clip(recouvrement, 0.0, 1.0))
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
        "T1_total_us": T1_total,
        "T_phi_us": T_phi,
        "T2_star_us": T2_star,
        "domega_dphi": d_omega_d_phi,
        "d2omega_dphi2": d2_omega_d_phi2,
        "metrique_fubini_study": g_fubini,
    }


# ============================================================================
# 4. EXÉCUTION & TESTS
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fluxonium Core v2 — Accéléré Termux.")
    parser.add_argument("--EC", type=float, default=1.0, help="Énergie de charge EC (GHz).")
    parser.add_argument("--EL", type=float, default=0.5, help="Énergie inductive EL (GHz).")
    parser.add_argument("--EJ", type=float, default=5.0, help="Énergie Josephson EJ (GHz).")
    parser.add_argument("--basis", type=int, default=45, help="Dimension de la base (40-50 recommandé).")
    parser.add_argument("--output", type=str, default=None, help="Répertoire de sortie JSON/CSV.")
    args = parser.parse_args()

    params = FluxoniumParams(EC=args.EC, EL=args.EL, EJ=args.EJ, N_basis=args.basis)
    solver = FluxoniumSolver(params)

    # Caractérisation au point doux (Sweet Spot)
    obs_sweet = analyser_point_flux(solver, phi_ext=math.pi)

    print("=" * 86)
    print(f"  FLUXONIUM CORE [Termux Edition] (EC={params.EC} GHz | EL={params.EL} GHz | EJ={params.EJ} GHz)")
    print("=" * 86)
    print("CARACTÉRISTIQUES AU POINT DOUX (Phi_ext = pi) :")
    print(f"  f01             : {obs_sweet['f01_GHz']:.6f} GHz ({obs_sweet['f01_GHz']*1e3:.2f} MHz)")
    print(f"  f12             : {obs_sweet['f12_GHz']:.6f} GHz")
    print(f"  Anharmonicité   : {obs_sweet['anharmonicite_GHz']:+.5f} GHz (relative : {obs_sweet['anharmonicite_rel']*100:.1f} %)")
    print(f"  Élément charge  : |<0|n|1>| = {obs_sweet['n01']:.3e} (protection de parité)")
    print(f"  T1 (Total)      : {obs_sweet['T1_total_us']:.1f} µs (Diélectrique : {obs_sweet['T1_diel_us']:.1f} µs)")
    print(f"  T_phi (1/f)     : {obs_sweet['T_phi_us']:.1f} µs | T2* = {obs_sweet['T2_star_us']:.1f} µs")
    print(f"  Métrique F-S    : g_PhiPhi = {obs_sweet['metrique_fubini_study']:.4f}")
    print("-" * 86)
    print("BALAYAGE DE L'ÉNERGIE JOSEPHSON EJ (Phi_ext = pi) :")
    print(f"{'EJ (GHz)':<9} | {'f01 (GHz)':<10} | {'alpha (GHz)':<12} | {'T1 (µs)':<9} | {'T2* (µs)':<9} | {'g_FS'}")
    print("-" * 86)

    scan_EJ = []
    for ej in np.linspace(2.0, 10.0, 9):
        obs = analyser_point_flux(solver, phi_ext=math.pi, EJ_override=float(ej))
        scan_EJ.append(obs)
        print(
            f"{obs['EJ_GHz']:7.2f}   | {obs['f01_GHz']:8.5f}   | {obs['anharmonicite_GHz']:+10.5f}   | "
            f"{obs['T1_total_us']:7.1f}   | {obs['T2_star_us']:7.1f}   | {obs['metrique_fubini_study']:8.3f}"
        )
    print("=" * 86)

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fluxonium_summary.json").write_text(
            json.dumps({"params": asdict(params), "sweet_spot": obs_sweet, "scan_EJ": scan_EJ}, indent=2),
            encoding="utf-8",
        )
        with (out / "fluxonium_scan_EJ.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(scan_EJ[0].keys()))
            writer.writeheader()
            writer.writerows(scan_EJ)
        print(f"Données enregistrées dans : {out}")


if __name__ == "__main__":
    main()
