#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOYAU DYNAMIQUE CONSERVATIF V50.1 — Bilan Fermé & Calibration par Dichotomie
============================================================================
Modèle couplé à deux états (x_1, x_2) sous matrice de transition W(alpha),
saturation non linéaire (tanh), bruit interne multiplicatif et contrôle
strict de conservation de masse à chaque pas de temps :

    dm = -dr + dc + cr + db + de   (|résidu| <= 1e-10)

Modes supportés :
  - 'chronique' : sollicitation continue uniforme sur n_stress pas
  - 'aigu'      : sollicitation impulsionnelle discrète (10 pics)

Dépendance unique : numpy
"""
from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional
import argparse
import csv
import json
import time
import numpy as np


# ============================================================================
# 1. CONFIGURATION & VALIDATION
# ============================================================================

@dataclass(frozen=True)
class Config:
    n_graines: int = 10
    n_stress: int = 100
    n_recup: int = 20
    seed: int = 42
    dose_reelle_cible: float = 0.50
    tolerance_dose: float = 1e-6
    alphas: tuple[float, ...] = (0.0, 0.20)
    gains: tuple[float, ...] = (0.0, 0.80)
    seuil: float = 0.50
    decay: float = 0.10
    etat_initial: tuple[float, float] = (1.0, 1.0)
    bruit_interne: float = 0.01
    tolerance_bilan: float = 1e-10
    calibration_grille_points: int = 100
    calibration_max_amplitude: float = 1000.0
    calibration_iterations: int = 80

    def __post_init__(self) -> None:
        if self.n_graines < 1 or self.n_stress < 10 or self.n_recup < 0:
            raise ValueError("Dimensions temporelles ou nombre de graines invalides.")
        if self.tolerance_dose <= 0 or self.calibration_grille_points < 2 or self.calibration_max_amplitude <= 0:
            raise ValueError("Paramètres de calibration invalides.")
        if len(self.etat_initial) != 2 or any(not 0.0 <= float(x) <= 1.0 for x in self.etat_initial):
            raise ValueError("L'état initial doit contenir deux valeurs dans [0, 1].")
        if any(not -0.2 <= float(a) <= 0.2 for a in self.alphas):
            raise ValueError("Le paramètre alpha doit appartenir à [-0.2, 0.2].")


# ============================================================================
# 2. OPÉRATEURS DE COUPLAGE & PROFILS DE SOLLICITATION
# ============================================================================

def build_W(alpha: float) -> np.ndarray:
    """Construit la matrice stochastique par colonnes W(alpha)."""
    W = np.array([[0.8 - alpha, 0.2 - alpha], [0.2 + alpha, 0.8 + alpha]], dtype=float)
    if np.any(W < 0.0) or not np.allclose(W.sum(axis=0), 1.0):
        raise ValueError(f"Matrice W invalide pour alpha={alpha}.")
    return W


def construire_profil(mode: str, n_stress: int, n_recup: int) -> np.ndarray:
    """Génère le vecteur temporel de sollicitation ('chronique' ou 'aigu')."""
    p = np.zeros(n_stress + n_recup, dtype=float)
    if mode == "chronique":
        p[:n_stress] = 1.0 / n_stress
    elif mode == "aigu":
        p[np.linspace(0, n_stress - 1, 10, dtype=int)] = 0.1
    else:
        raise ValueError(f"Mode inconnu : {mode}")
    return p


# ============================================================================
# 3. SIMULATION AVEC BILAN CONSERVATIF STRICT
# ============================================================================

def simuler(
    cfg: Config,
    W: np.ndarray,
    gain: float,
    profil: np.ndarray,
    amplitude: float,
    bruit: np.ndarray,
    observer: Optional[Callable] = None,
) -> dict:
    etat = np.asarray(cfg.etat_initial, dtype=float).copy()
    ref = etat.copy()
    profil = np.asarray(profil, dtype=float)
    bruit = np.asarray(bruit, dtype=float)
    n = len(profil)

    if bruit.shape != (n, 2):
        raise ValueError("Forme du tableau de bruit invalide.")

    cles_bilan = (
        "dose_demandee",
        "dose_reelle",
        "delta_perte",
        "delta_couplage",
        "correction_brute",
        "delta_bruit",
        "delta_ecretage",
        "delta_M",
    )
    phases = {p: {k: 0.0 for k in cles_bilan} for p in ("stress", "recup")}

    Fmin = float(np.prod(etat))
    dstress = None
    estress = None
    mr = 0.0
    mc = 0.0

    for t in range(n):
        phase = "stress" if t < cfg.n_stress else "recup"
        avant = etat.copy()
        masse = float(avant.sum())

        # 1. Extraction de dose
        demande = float(np.tanh(profil[t] * amplitude * cfg.decay))
        pd = np.array([demande, 0.0], dtype=float)
        apres = np.clip(avant - pd, 0.0, 1.0)
        perte = avant - apres
        dr = float(perte.sum())
        dd = float(pd.sum())

        # 2. Redistribution conservative par couplage W
        couple = W @ apres
        dc = float(couple.sum() - apres.sum())
        mc = max(mc, abs(dc))

        # 3. Rétroaction correctrice sous seuil
        correction = gain * np.tanh(2.0 * np.maximum(0.0, cfg.seuil - couple))
        cr = float(correction.sum())

        # 4. Fluctuation interne multiplicative
        be = bruit[t] * avant * (1.0 - avant)
        db = float(be.sum())

        # 5. Écrêtage dans [0, 1] et vérification exacte du bilan de masse
        avant_clip = couple + correction + be
        nouvel = np.clip(avant_clip, 0.0, 1.0)
        de = float(nouvel.sum() - avant_clip.sum())
        etat = nouvel

        dm = float(etat.sum() - masse)
        residu = dm - (-dr + dc + cr + db + de)
        mr = max(mr, abs(residu))
        if abs(residu) > cfg.tolerance_bilan:
            raise AssertionError(f"Bilan non fermé à t={t} : résidu={residu}")

        vals = {
            "dose_demandee": dd,
            "dose_reelle": dr,
            "delta_perte": -dr,
            "delta_couplage": dc,
            "correction_brute": cr,
            "delta_bruit": db,
            "delta_ecretage": de,
            "delta_M": dm,
        }
        for k, v in vals.items():
            phases[phase][k] += v

        dist = float(np.linalg.norm(etat - ref))
        F = float(etat[0] * etat[1])
        Fmin = min(Fmin, F)

        if t == cfg.n_stress - 1:
            dstress = dist
            estress = etat.copy()

        if observer is not None:
            observer(
                t,
                {"x1": float(etat[0]), "x2": float(etat[1]), "F": F, "distance_reference": dist},
                {
                    **vals,
                    "residu_bilan": residu,
                    "erreur_couplage": dc,
                    "delta_net_correction_apres_ecretage": float(np.sum(nouvel - couple - be)),
                },
                {"phase": phase, "profil_t": float(profil[t]), "amplitude": float(amplitude), "gain": float(gain)},
            )

    df = float(np.linalg.norm(etat - ref))
    dstress = df if dstress is None else dstress

    return {
        "dose_demandee": sum(x["dose_demandee"] for x in phases.values()),
        "dose_reelle": sum(x["dose_reelle"] for x in phases.values()),
        "F_min": Fmin,
        "F_final": float(np.prod(etat)),
        "x1_final": float(etat[0]),
        "x2_final": float(etat[1]),
        "distance_stress": dstress,
        "distance_finale": df,
        "amelioration_recuperation": dstress - df,
        "phases": phases,
        "max_residu": mr,
        "max_erreur_couplage": mc,
        "etat_stress": estress,
    }


# ============================================================================
# 4. CALIBRATION D'AMPLITUDE PAR GRILLE ET DICHOTOMIE
# ============================================================================

def calibrer_dose(cfg: Config, W: np.ndarray, gain: float, profil: np.ndarray, bruit: np.ndarray) -> tuple[float, dict]:
    """Recherche l'amplitude minimale atteignant dose_reelle_cible à tolerance_dose près."""
    cible = cfg.dose_reelle_cible
    ag = np.linspace(0.0, cfg.calibration_max_amplitude, cfg.calibration_grille_points)
    dg = np.array([simuler(cfg, W, gain, profil, float(a), bruit)["dose_reelle"] for a in ag])
    eg = dg - cible

    nonmono = bool(np.any(np.diff(dg) < -cfg.tolerance_dose))
    brackets = []
    for i in range(len(ag)):
        if abs(eg[i]) <= cfg.tolerance_dose:
            brackets.append((float(ag[i]), float(ag[i]), 0))
        if i < len(ag) - 1 and eg[i] * eg[i + 1] < 0.0:
            brackets.append((float(ag[i]), float(ag[i + 1]), 1))

    if not brackets:
        raise RuntimeError(f"Cible non encadrée sur la grille : min={dg.min():.6g}, max={dg.max():.6g}, cible={cible:.6g}")

    sols = []
    for lo, hi, kind in brackets:
        if kind == 0:
            r = simuler(cfg, W, gain, profil, lo, bruit)
            sols.append((lo, r))
            continue

        flo = simuler(cfg, W, gain, profil, lo, bruit)["dose_reelle"] - cible
        fhi = simuler(cfg, W, gain, profil, hi, bruit)["dose_reelle"] - cible
        if flo * fhi >= 0.0:
            continue

        best = None
        for _ in range(cfg.calibration_iterations):
            mid = 0.5 * (lo + hi)
            r = simuler(cfg, W, gain, profil, mid, bruit)
            fm = r["dose_reelle"] - cible
            best = (mid, r)
            if abs(fm) <= cfg.tolerance_dose:
                break
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm

        if best and abs(best[1]["dose_reelle"] - cible) <= cfg.tolerance_dose:
            sols.append(best)

    if not sols:
        raise RuntimeError("Aucun intervalle détecté n'a convergé à la tolérance demandée.")

    sols.sort(key=lambda x: x[0])
    a, r = sols[0]
    r["calibration"] = {
        "statut": "convergee",
        "non_monotone_sur_grille": nonmono,
        "solutions_detectees": len(sols),
        "selection": "plus_petite_amplitude_detectee",
    }
    return a, r


# ============================================================================
# 5. PLAN EXPÉRIMENTAL & EXPORTS
# ============================================================================

def run_plan(cfg: Config) -> dict:
    master = np.random.default_rng(cfg.seed)
    seeds = master.integers(0, 2**31 - 1, size=cfg.n_graines)
    out = {}

    for mode in ("chronique", "aigu"):
        out[mode] = {}
        profil = construire_profil(mode, cfg.n_stress, cfg.n_recup)
        for gain in cfg.gains:
            out[mode][gain] = {}
            for alpha in cfg.alphas:
                vals = []
                W = build_W(alpha)
                for seed in seeds:
                    rng = np.random.default_rng(int(seed))
                    bruit = rng.normal(0.0, cfg.bruit_interne, size=(len(profil), 2))
                    a, r = calibrer_dose(cfg, W, gain, profil, bruit)
                    vals.append({
                        "amplitude": a,
                        **{
                            k: r[k]
                            for k in (
                                "dose_reelle",
                                "F_min",
                                "F_final",
                                "distance_stress",
                                "distance_finale",
                                "amelioration_recuperation",
                                "max_residu",
                                "max_erreur_couplage",
                            )
                        },
                    })
                out[mode][gain][alpha] = vals
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Noyau Dynamique Conservatif V50.1 — Bilan Fermé.")
    parser.add_argument("--graines", type=int, default=10, help="Nombre de trajectoires stochastiques.")
    parser.add_argument("--cible", type=float, default=0.50, help="Dose réelle cible à calibrer.")
    parser.add_argument("--output", type=str, default=None, help="Dossier d'export optionnel (CSV + JSON).")
    args = parser.parse_args()

    cfg = Config(n_graines=args.graines, dose_reelle_cible=args.cible)
    t0 = time.perf_counter()
    res = run_plan(cfg)
    duree = time.perf_counter() - t0

    print("=" * 88)
    print(f"  NOYAU CONSERVATIF V50.1 (Graines={cfg.n_graines} | Cible={cfg.dose_reelle_cible} | Tolérance bilan={cfg.tolerance_bilan:.0e})")
    print("=" * 88)

    lignes_export = []
    for mode in res:
        print(f"\nRÉGIME : {mode.upper()}")
        print(f"{'Gain':<6} | {'Alpha':<6} | {'Amp moy':<9} | {'Dose moy':<9} | {'F_min moy':<9} | {'F_fin moy':<9} | {'Résidu max'}")
        print("-" * 88)
        for gain in res[mode]:
            for alpha, rows in res[mode][gain].items():
                amp_m = float(np.mean([x["amplitude"] for x in rows]))
                dose_m = float(np.mean([x["dose_reelle"] for x in rows]))
                fmin_m = float(np.mean([x["F_min"] for x in rows]))
                ffin_m = float(np.mean([x["F_final"] for x in rows]))
                res_max = float(np.max([x["max_residu"] for x in rows]))

                print(
                    f"{gain:5.2f}  | {alpha:5.2f}  | {amp_m:8.4f}  | {dose_m:8.6f}  | "
                    f"{fmin_m:8.4f}  | {ffin_m:8.4f}  | {res_max:.2e}"
                )
                lignes_export.append({
                    "mode": mode,
                    "gain": gain,
                    "alpha": alpha,
                    "amplitude_moy": amp_m,
                    "dose_reelle_moy": dose_m,
                    "F_min_moy": fmin_m,
                    "F_final_moy": ffin_m,
                    "max_residu": res_max,
                })

    print("=" * 88)
    print(f"Exécution complète terminée en {duree:.2f} s.")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "noyau_v50_synthese.json").write_text(
            json.dumps({"config": asdict(cfg), "synthese": lignes_export}, indent=2),
            encoding="utf-8",
        )
        with (out / "noyau_v50_synthese.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(lignes_export[0].keys()))
            writer.writeheader()
            writer.writerows(lignes_export)


if __name__ == "__main__":
    main()
