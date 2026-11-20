#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOYAU QUANTIQUE V51 - Qubit ouvert, canal CPTP, calibration par dichotomie
==========================================================================
Termux : pkg install python python-numpy   (puis : python noyau_quantique_v51.py --rapide)

Modèle : un qubit (|g>, |e>) décrit par une matrice densité rho (2x2).
Chaque pas de temps applique, dans l'ordre :
  1. Extraction (dose)   : canal d'amortissement d'amplitude, Kraus K0, K1
                           dose_t = p_t * rho_ee  (probabilité d'émission détectée)
  2. Couplage cohérent   : U(alpha) = exp(-i H), H = (desaccord*alpha*sz + rabi*sx)/2
  3. Rétroaction         : rotation autour de sy, angle = gain*tanh(2*max(0, seuil - P_e))
  4. Bruit multiplicatif : rotations aléatoires autour de sz et sx, amplitude ~ sqrt(P_e(1-P_e))

Ici la conservation est un vrai test : Tr(rho)=1, rho hermitienne, rho >= 0,
complétude des Kraus et unitarité de U sont vérifiées à chaque pas.
(Le bilan de population P_e reste une identité comptable, comme en V50.)

Limite : modèle jouet. Je ne prétends pas qu'il décrit un système physique précis.
Dépendance unique : numpy.
"""
from __future__ import annotations
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class Config:
    n_graines: int = 5
    n_stress: int = 100
    n_recup: int = 20
    seed: int = 42
    dose_reelle_cible: float = 0.50
    tolerance_dose: float = 1e-6
    alphas: tuple[float, ...] = (0.0, 0.20)
    gains: tuple[float, ...] = (0.0, 0.80)
    seuil: float = 0.50
    decay: float = 0.10
    rabi: float = 0.30
    echelle_desaccord: float = 1.0
    bruit_interne: float = 0.01
    tolerance_bilan: float = 1e-10
    calibration_grille_points: int = 40
    calibration_max_amplitude: float = 1000.0
    calibration_iterations: int = 80

    def __post_init__(self) -> None:
        if self.n_graines < 1 or self.n_stress < 10 or self.n_recup < 0:
            raise ValueError("Dimensions temporelles ou nombre de graines invalides.")
        if self.tolerance_dose <= 0 or self.calibration_grille_points < 2 or self.calibration_max_amplitude <= 0:
            raise ValueError("Paramètres de calibration invalides.")
        if not 0.0 < self.dose_reelle_cible < 1.0:
            raise ValueError("La dose cible doit être dans ]0, 1[ (une seule excitation initiale).")
        if any(not -0.2 <= float(a) <= 0.2 for a in self.alphas):
            raise ValueError("alpha doit appartenir à [-0.2, 0.2].")


# ============================================================================
# 2. OPÉRATEURS ET PROFILS
# ============================================================================

def rot(sigma: np.ndarray, theta: float) -> np.ndarray:
    """exp(-i theta sigma / 2), unitaire exacte pour sigma de carré identité."""
    return np.cos(theta / 2.0) * I2 - 1j * np.sin(theta / 2.0) * sigma


def build_U(cfg: Config, alpha: float) -> np.ndarray:
    """Propagateur cohérent sur un pas (dt = 1)."""
    delta = alpha * cfg.echelle_desaccord
    omega = float(np.hypot(delta, cfg.rabi))
    if omega == 0.0:
        return I2.copy()
    n_sigma = (delta * SZ + cfg.rabi * SX) / omega
    U = np.cos(omega / 2.0) * I2 - 1j * np.sin(omega / 2.0) * n_sigma
    if np.linalg.norm(U.conj().T @ U - I2) > 1e-12:
        raise ValueError(f"U non unitaire pour alpha={alpha}.")
    return U


def construire_profil(mode: str, n_stress: int, n_recup: int) -> np.ndarray:
    p = np.zeros(n_stress + n_recup, dtype=float)
    if mode == "chronique":
        p[:n_stress] = 1.0 / n_stress
    elif mode == "aigu":
        p[np.linspace(0, n_stress - 1, 10, dtype=int)] = 0.1
    else:
        raise ValueError(f"Mode inconnu : {mode}")
    return p


def distance_trace(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.linalg.eigvalsh(a - b)).sum())


# ============================================================================
# 3. SIMULATION AVEC CONTRÔLES CPTP
# ============================================================================

def simuler(cfg: Config, U: np.ndarray, gain: float, profil: np.ndarray,
            amplitude: float, bruit: np.ndarray) -> dict:
    n = len(profil)
    if bruit.shape != (n, 2):
        raise ValueError("Forme du tableau de bruit invalide.")

    psi0 = np.array([0.0, 1.0], dtype=complex)          # |e>
    ref = np.outer(psi0, psi0.conj())
    rho = ref.copy()

    dose_tot = 0.0
    Fmin = 1.0
    dstress = None
    max_viol = 0.0

    for t in range(n):
        pe0 = float(rho[1, 1].real)

        # 1. Extraction : amortissement d'amplitude (CPTP par construction)
        p = float(np.clip(np.tanh(profil[t] * amplitude * cfg.decay), 0.0, 1.0))
        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]], dtype=complex)
        K1 = np.array([[0.0, np.sqrt(p)], [0.0, 0.0]], dtype=complex)
        complet = np.linalg.norm(K0.conj().T @ K0 + K1.conj().T @ K1 - I2)
        r1 = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T
        dose = p * pe0
        dose_tot += dose

        # 2. Couplage cohérent
        r2 = U @ r1 @ U.conj().T
        pe2 = float(r2[1, 1].real)

        # 3. Rétroaction unitaire sous seuil
        theta = gain * np.tanh(2.0 * max(0.0, cfg.seuil - pe2))
        Uf = rot(SY, theta)
        r3 = Uf @ r2 @ Uf.conj().T
        pe3 = float(r3[1, 1].real)

        # 4. Bruit multiplicatif (rotations aléatoires, trajectoire stochastique)
        env = 2.0 * np.sqrt(max(pe3 * (1.0 - pe3), 0.0))
        Un = rot(SX, bruit[t, 1] * env) @ rot(SZ, bruit[t, 0] * env)
        rho = Un @ r3 @ Un.conj().T

        # 5. Vrais tests de conservation et de validité physique
        err_trace = abs(float(np.trace(rho).real) - 1.0)
        err_herm = float(np.linalg.norm(rho - rho.conj().T))
        viol_pos = max(0.0, -float(np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)[0]))
        pas_viol = max(err_trace, err_herm, viol_pos, float(complet))
        max_viol = max(max_viol, pas_viol)
        if pas_viol > cfg.tolerance_bilan:
            raise AssertionError(f"Validité CPTP rompue à t={t} : {pas_viol:.3e}")

        F = float(np.real(psi0.conj() @ rho @ psi0))
        Fmin = min(Fmin, F)
        if t == cfg.n_stress - 1:
            dstress = distance_trace(rho, ref)

    df = distance_trace(rho, ref)
    dstress = df if dstress is None else dstress
    return {
        "dose_reelle": dose_tot,
        "F_min": Fmin,
        "F_final": float(np.real(psi0.conj() @ rho @ psi0)),
        "purete_finale": float(np.trace(rho @ rho).real),
        "coherence_finale": float(abs(rho[0, 1])),
        "distance_stress": dstress,
        "distance_finale": df,
        "amelioration_recuperation": dstress - df,
        "max_violation": max_viol,
    }


# ============================================================================
# 4. CALIBRATION (grille + dichotomie)
# ============================================================================

def calibrer_dose(cfg: Config, U, gain, profil, bruit) -> tuple[Optional[float], dict]:
    cible, tol = cfg.dose_reelle_cible, cfg.tolerance_dose
    f = lambda a: simuler(cfg, U, gain, profil, float(a), bruit)
    ag = np.linspace(0.0, cfg.calibration_max_amplitude, cfg.calibration_grille_points)
    dg = np.array([f(a)["dose_reelle"] for a in ag])
    eg = dg - cible
    nonmono = bool(np.any(np.diff(dg) < -tol))   # effet Zénon quantique possible

    brackets = []
    for i in range(len(ag)):
        if abs(eg[i]) <= tol:
            brackets.append((ag[i], ag[i], 0))
        if i < len(ag) - 1 and eg[i] * eg[i + 1] < 0.0:
            brackets.append((ag[i], ag[i + 1], 1))
    if not brackets:
        return None, {"statut": "non_encadree", "dose_min": float(dg.min()),
                      "dose_max": float(dg.max()), "non_monotone": nonmono}

    sols = []
    for lo, hi, kind in brackets:
        if kind == 0:
            sols.append((float(lo), f(lo)))
            continue
        flo = f(lo)["dose_reelle"] - cible
        best = None
        for _ in range(cfg.calibration_iterations):
            mid = 0.5 * (lo + hi)
            r = f(mid)
            fm = r["dose_reelle"] - cible
            best = (float(mid), r)
            if abs(fm) <= tol:
                break
            if flo * fm <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fm
        if best and abs(best[1]["dose_reelle"] - cible) <= tol:
            sols.append(best)

    if not sols:
        return None, {"statut": "non_convergee", "non_monotone": nonmono}
    sols.sort(key=lambda x: x[0])
    a, r = sols[0]
    r["calibration"] = {"statut": "convergee", "non_monotone": nonmono, "solutions": len(sols)}
    return a, r


# ============================================================================
# 5. PLAN, AFFICHAGE, EXPORT
# ============================================================================

CLES = ("dose_reelle", "F_min", "F_final", "purete_finale", "coherence_finale",
        "distance_stress", "distance_finale", "amelioration_recuperation", "max_violation")


def run_plan(cfg: Config) -> dict:
    seeds = np.random.default_rng(cfg.seed).integers(0, 2**31 - 1, size=cfg.n_graines)
    out: dict = {}
    for mode in ("chronique", "aigu"):
        profil = construire_profil(mode, cfg.n_stress, cfg.n_recup)
        out[mode] = {}
        for gain in cfg.gains:
            for alpha in cfg.alphas:
                U = build_U(cfg, alpha)
                rows, echecs = [], 0
                for s in seeds:
                    bruit = np.random.default_rng(int(s)).normal(0.0, cfg.bruit_interne, size=(len(profil), 2))
                    a, r = calibrer_dose(cfg, U, gain, profil, bruit)
                    if a is None:
                        echecs += 1
                        continue
                    rows.append({"amplitude": a, **{k: r[k] for k in CLES}})
                out[mode][(gain, alpha)] = (rows, echecs)
                print(f"  [{mode} g={gain:.2f} a={alpha:.2f}] "
                      f"{len(rows)}/{cfg.n_graines} calibrées", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Noyau quantique V51 (qubit ouvert).")
    ap.add_argument("--graines", type=int, default=5)
    ap.add_argument("--cible", type=float, default=0.50)
    ap.add_argument("--rapide", action="store_true", help="3 graines, grille 25 (téléphone)")
    ap.add_argument("--output", type=str, default=None, help="ex. ~/storage/shared/noyau")
    a = ap.parse_args()

    kw = dict(n_graines=a.graines, dose_reelle_cible=a.cible)
    if a.rapide:
        kw.update(n_graines=3, calibration_grille_points=25)
    cfg = Config(**kw)

    t0 = time.perf_counter()
    res = run_plan(cfg)
    duree = time.perf_counter() - t0

    print("=" * 84)
    print(f" NOYAU QUANTIQUE V51 | graines={cfg.n_graines} | cible={cfg.dose_reelle_cible} | tol={cfg.tolerance_bilan:.0e}")
    print("=" * 84)
    export = []
    for mode, d in res.items():
        print(f"\nREGIME : {mode.upper()}")
        print(f"{'Gain':<5}|{'Alpha':<6}|{'Amp':<9}|{'Dose':<9}|{'F_min':<7}|{'F_fin':<7}|{'Pureté':<7}|{'Viol.max'}")
        print("-" * 84)
        for (gain, alpha), (rows, echecs) in d.items():
            if not rows:
                print(f"{gain:4.2f} |{alpha:5.2f} | cible non atteinte ({echecs} échecs)")
                continue
            m = {k: float(np.mean([r[k] for r in rows])) for k in ("amplitude",) + CLES}
            m["max_violation"] = float(np.max([r["max_violation"] for r in rows]))
            print(f"{gain:4.2f} |{alpha:5.2f} |{m['amplitude']:8.3f} |{m['dose_reelle']:8.6f} |"
                  f"{m['F_min']:6.3f} |{m['F_final']:6.3f} |{m['purete_finale']:6.3f} |{m['max_violation']:.1e}")
            export.append({"mode": mode, "gain": gain, "alpha": alpha, "echecs": echecs, **m})
    print("=" * 84)
    print(f"Terminé en {duree:.1f} s.")

    if a.output and export:
        out = Path(a.output).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "noyau_v51.json").write_text(
            json.dumps({"config": asdict(cfg), "synthese": export}, indent=2), encoding="utf-8")
        with (out / "noyau_v51.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(export[0].keys()))
            w.writeheader()
            w.writerows(export)
        print(f"Exports : {out}")


if __name__ == "__main__":
    main()
