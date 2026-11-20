#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOYAU QUANTIQUE V52 - Deux qubits couplés, équation de Lindblad, intrication
============================================================================
Termux : pkg install python python-numpy   puis   python noyau_quantique_v52.py --rapide

Modèle (jouet, non calé sur un système physique précis) :
  Base |gg>,|ge>,|eg>,|ee>. État initial |eg> (une excitation, sur A).
  Sur chaque pas (dt = 1), générateur de Lindblad constant :
      d rho/dt = -i[H_t, rho] + gamma_t * D[sigma-_A](rho)
      H_t = J (s+_A s-_B + h.c.)            couplage d'échange   (rôle de W)
          + (alpha*echelle/2)(sz_A - sz_B)  désaccord            (rôle de alpha)
          + theta_t * sy_B / 2              rétroaction d'état   (rôle du gain)
          + env*(eta_A sz_A + eta_B sz_B)/2 bruit classique multiplicatif
      gamma_t = tanh(profil_t * amplitude * decay)               (extraction)
  Propagation EXACTE par exponentielle matricielle d'un générateur augmenté
  17x17 [rho ; compteur], dont la dernière composante donne la dose :
      dose_pas = gamma_t * integrale de <n_A> sur le pas.

Tests réels à chaque pas : Tr(rho)=1, rho hermitienne, rho >= 0.
Si gain = 0 : H et le bruit conservent le nombre d'excitations, donc
      Delta<N> + dose = 0   (test indépendant de la dose calculée).
Intrication : concurrence de Wootters (Phys. Rev. Lett. 80, 2245, 1998).

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
I4 = np.eye(4, dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)      # sigma- : |e> -> |g>
SP = SM.T.copy()
NN = np.diag([0.0, 1.0]).astype(complex)
SZ = 2.0 * NN - I2
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
YY = np.kron(SY, SY)


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
    couplage_J: float = 0.30
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
# 2. SUPEROPÉRATEURS (vectorisation par lignes : vec(A rho B) = (A kron B^T) vec(rho))
# ============================================================================

def op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


def super_comm(H: np.ndarray) -> np.ndarray:
    """rho -> -i[H, rho]"""
    return -1j * (np.kron(H, I4) - np.kron(I4, H.T))


def super_diss(L: np.ndarray) -> np.ndarray:
    """rho -> L rho L^dag - (1/2){L^dag L, rho}"""
    LdL = L.conj().T @ L
    return np.kron(L, L.conj()) - 0.5 * np.kron(LdL, I4) - 0.5 * np.kron(I4, LdL.T)


class Modele:
    """Pièces constantes pour un alpha donné."""

    def __init__(self, cfg: Config, alpha: float) -> None:
        sm_a, sp_a = op(SM, I2), op(SP, I2)
        sm_b, sp_b = op(I2, SM), op(I2, SP)
        H0 = cfg.couplage_J * (sp_a @ sm_b + sm_a @ sp_b)
        H0 = H0 + 0.5 * alpha * cfg.echelle_desaccord * (op(SZ, I2) - op(I2, SZ))
        self.S0 = super_comm(H0)
        self.Sy = super_comm(0.5 * op(I2, SY))
        self.Sza = super_comm(0.5 * op(SZ, I2))
        self.Szb = super_comm(0.5 * op(I2, SZ))
        self.DA = super_diss(sm_a)
        nA, nB = op(NN, I2), op(I2, NN)
        self.rowA = nA.T.reshape(-1)               # Tr(nA rho) = rowA . vec(rho)
        self.nTot = nA + nB
        # contrôle : le générateur doit conserver la trace
        tr = np.eye(4, dtype=complex).reshape(-1)
        for S in (self.S0, self.Sy, self.Sza, self.Szb, self.DA):
            if np.linalg.norm(tr @ S) > 1e-12:
                raise ValueError("Superopérateur non préservant la trace.")


def expm_taylor(A: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(A, 1))
    s = 0 if nrm <= 0.5 else int(np.ceil(np.log2(nrm / 0.5)))
    B = A / (2.0 ** s)
    E = np.eye(A.shape[0], dtype=complex)
    T = E.copy()
    for k in range(1, 16):
        T = T @ B / k
        E = E + T
    for _ in range(s):
        E = E @ E
    return E


# ============================================================================
# 3. PROFILS ET MESURES
# ============================================================================

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
    return float(0.5 * np.abs(np.linalg.eigvalsh((a - b + (a - b).conj().T) / 2.0)).sum())


def concurrence(rho: np.ndarray) -> float:
    rt = YY @ rho.conj() @ YY
    ev = np.sqrt(np.abs(np.linalg.eigvals(rho @ rt).real))
    ev = np.sort(ev)[::-1]
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


# ============================================================================
# 4. SIMULATION
# ============================================================================

def simuler(cfg: Config, M: Modele, gain: float, profil: np.ndarray,
            amplitude: float, bruit: np.ndarray) -> dict:
    n = len(profil)
    if bruit.shape != (n, 2):
        raise ValueError("Forme du tableau de bruit invalide.")

    psi0 = np.zeros(4, dtype=complex)
    psi0[2] = 1.0                                   # |eg>
    ref = np.outer(psi0, psi0.conj())
    rho = ref.copy()

    dose_tot, Fmin, Cmax = 0.0, 1.0, 0.0
    dstress = None
    max_viol, max_resN = 0.0, 0.0
    G = np.zeros((17, 17), dtype=complex)

    for t in range(n):
        N0 = float(np.trace(M.nTot @ rho).real)
        P = N0 / 2.0

        gam = float(np.clip(np.tanh(profil[t] * amplitude * cfg.decay), 0.0, 1.0))
        theta = gain * float(np.tanh(2.0 * max(0.0, cfg.seuil - P)))
        env = 2.0 * np.sqrt(max(P * (1.0 - P), 0.0))

        G[:16, :16] = (M.S0 + theta * M.Sy + bruit[t, 0] * env * M.Sza
                       + bruit[t, 1] * env * M.Szb + gam * M.DA)
        G[16, :16] = gam * M.rowA
        E = expm_taylor(G)
        out = E[:, :16] @ rho.reshape(-1)
        rho = out[:16].reshape(4, 4)
        dose = float(out[16].real)
        dose_tot += dose

        # vrais tests de validité
        err_trace = abs(float(np.trace(rho).real) - 1.0)
        err_herm = float(np.linalg.norm(rho - rho.conj().T))
        viol_pos = max(0.0, -float(np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)[0]))
        pas_viol = max(err_trace, err_herm, viol_pos)
        max_viol = max(max_viol, pas_viol)
        if pas_viol > cfg.tolerance_bilan:
            raise AssertionError(f"Validité rompue à t={t} : {pas_viol:.3e}")

        if gain == 0.0:
            N1 = float(np.trace(M.nTot @ rho).real)
            res = abs((N1 - N0) + dose)
            max_resN = max(max_resN, res)
            if res > cfg.tolerance_bilan:
                raise AssertionError(f"Excitations non conservées à t={t} : {res:.3e}")

        F = float(np.real(psi0.conj() @ rho @ psi0))
        Fmin = min(Fmin, F)
        Cmax = max(Cmax, concurrence(rho))
        if t == cfg.n_stress - 1:
            dstress = distance_trace(rho, ref)

    df = distance_trace(rho, ref)
    dstress = df if dstress is None else dstress
    return {
        "dose_reelle": dose_tot,
        "F_min": Fmin,
        "F_final": float(np.real(psi0.conj() @ rho @ psi0)),
        "C_max": Cmax,
        "C_final": concurrence(rho),
        "purete_finale": float(np.trace(rho @ rho).real),
        "distance_stress": dstress,
        "distance_finale": df,
        "amelioration_recuperation": dstress - df,
        "max_violation": max_viol,
        "max_residu_N": max_resN if gain == 0.0 else None,
    }


# ============================================================================
# 5. CALIBRATION (grille + dichotomie)
# ============================================================================

def calibrer_dose(cfg: Config, M, gain, profil, bruit) -> tuple[Optional[float], dict]:
    cible, tol = cfg.dose_reelle_cible, cfg.tolerance_dose
    f = lambda a: simuler(cfg, M, gain, profil, float(a), bruit)
    ag = np.linspace(0.0, cfg.calibration_max_amplitude, cfg.calibration_grille_points)
    dg = np.array([f(a)["dose_reelle"] for a in ag])
    eg = dg - cible
    nonmono = bool(np.any(np.diff(dg) < -tol))

    brackets = []
    for i in range(len(ag)):
        if abs(eg[i]) <= tol:
            brackets.append((ag[i], ag[i], 0))
        if i < len(ag) - 1 and eg[i] * eg[i + 1] < 0.0:
            brackets.append((ag[i], ag[i + 1], 1))
    if not brackets:
        return None, {"statut": "non_encadree"}

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
        return None, {"statut": "non_convergee"}
    sols.sort(key=lambda x: x[0])
    a, r = sols[0]
    r["calibration"] = {"statut": "convergee", "non_monotone": nonmono, "solutions": len(sols)}
    return a, r


# ============================================================================
# 6. PLAN, AFFICHAGE, EXPORT
# ============================================================================

CLES = ("dose_reelle", "F_min", "F_final", "C_max", "C_final", "purete_finale",
        "distance_stress", "distance_finale", "amelioration_recuperation", "max_violation")


def run_plan(cfg: Config) -> dict:
    seeds = np.random.default_rng(cfg.seed).integers(0, 2**31 - 1, size=cfg.n_graines)
    out: dict = {}
    for mode in ("chronique", "aigu"):
        profil = construire_profil(mode, cfg.n_stress, cfg.n_recup)
        out[mode] = {}
        for gain in cfg.gains:
            for alpha in cfg.alphas:
                M = Modele(cfg, alpha)
                rows, echecs = [], 0
                for s in seeds:
                    bruit = np.random.default_rng(int(s)).normal(0.0, cfg.bruit_interne, size=(len(profil), 2))
                    a, r = calibrer_dose(cfg, M, gain, profil, bruit)
                    if a is None:
                        echecs += 1
                        continue
                    rows.append({"amplitude": a, "max_residu_N": r["max_residu_N"], **{k: r[k] for k in CLES}})
                out[mode][(gain, alpha)] = (rows, echecs)
                print(f"  [{mode} g={gain:.2f} a={alpha:.2f}] {len(rows)}/{cfg.n_graines} calibrées", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Noyau quantique V52 (deux qubits, Lindblad).")
    ap.add_argument("--graines", type=int, default=5)
    ap.add_argument("--cible", type=float, default=0.50)
    ap.add_argument("--rapide", action="store_true", help="2 graines, grille 20 (téléphone)")
    ap.add_argument("--output", type=str, default=None, help="ex. ~/storage/shared/noyau")
    a = ap.parse_args()

    kw = dict(n_graines=a.graines, dose_reelle_cible=a.cible)
    if a.rapide:
        kw.update(n_graines=2, calibration_grille_points=20)
    cfg = Config(**kw)

    t0 = time.perf_counter()
    res = run_plan(cfg)
    duree = time.perf_counter() - t0

    print("=" * 92)
    print(f" NOYAU QUANTIQUE V52 | graines={cfg.n_graines} | cible={cfg.dose_reelle_cible} | tol={cfg.tolerance_bilan:.0e}")
    print("=" * 92)
    export = []
    for mode, d in res.items():
        print(f"\nREGIME : {mode.upper()}")
        print(f"{'Gain':<5}|{'Alpha':<6}|{'Amp':<9}|{'Dose':<9}|{'F_min':<7}|{'C_max':<7}|{'C_fin':<7}|{'Pureté':<7}|{'Viol.':<8}|{'Res.N'}")
        print("-" * 92)
        for (gain, alpha), (rows, echecs) in d.items():
            if not rows:
                print(f"{gain:4.2f} |{alpha:5.2f} | cible non atteinte ({echecs} échecs)")
                continue
            m = {k: float(np.mean([r[k] for r in rows])) for k in ("amplitude",) + CLES}
            m["max_violation"] = float(np.max([r["max_violation"] for r in rows]))
            rn = [r["max_residu_N"] for r in rows if r["max_residu_N"] is not None]
            m["max_residu_N"] = float(np.max(rn)) if rn else None
            rn_txt = f"{m['max_residu_N']:.1e}" if rn else "n/a"
            print(f"{gain:4.2f} |{alpha:5.2f} |{m['amplitude']:8.3f} |{m['dose_reelle']:8.6f} |{m['F_min']:6.3f} |"
                  f"{m['C_max']:6.3f} |{m['C_final']:6.3f} |{m['purete_finale']:6.3f} |{m['max_violation']:.1e} |{rn_txt}")
            export.append({"mode": mode, "gain": gain, "alpha": alpha, "echecs": echecs, **m})
    print("=" * 92)
    print(f"Terminé en {duree:.1f} s.")

    if a.output and export:
        out = Path(a.output).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "noyau_v52.json").write_text(
            json.dumps({"config": asdict(cfg), "synthese": export}, indent=2), encoding="utf-8")
        with (out / "noyau_v52.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(export[0].keys()))
            w.writeheader()
            w.writerows(export)
        print(f"Exports : {out}")


if __name__ == "__main__":
    main()
