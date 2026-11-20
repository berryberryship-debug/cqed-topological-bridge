#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ÉDIFICE QUANTIQUE 1.0.0 - fichier unique (ouvrable dans nano), généré depuis src/edifice.
Termux : pkg install python python-numpy nano
Usage  : python edifice.py valider
         python edifice.py campagne --rapide --output ~/noyau
         python edifice.py trajectoire --gain 0.8 --alpha 0.1 --output ~/traj.csv
Sections : 1 core | 2 modele | 3 validation | 4 campagne | 5 cli
(dans nano : Ctrl+W cherche un mot, Ctrl+_ va à une ligne, Ctrl+O enregistre, Ctrl+X quitte)
"""
from __future__ import annotations
import argparse, csv, hashlib, json, logging, os, platform, sys, tempfile, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np

VERSION = __version__ = "1.0.0"
C = K = validation = sys.modules[__name__]   # les anciens modules deviennent des sections


# ============================================================================
# SECTION 1 : core
# ============================================================================
# -*- coding: utf-8 -*-
"""Noyau numérique : opérateurs, superopérateurs, exponentielle matricielle, mesures.

Convention de vectorisation par lignes : vec(A rho B) = (A kron B^T) vec(rho).
Base à un qubit : |g> = indice 0, |e> = indice 1. Qubit 0 = bit de poids fort.
Dépendance unique : numpy.
"""

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)        # sigma- : |e> -> |g>
SP = SM.T.copy()
NN = np.diag([0.0, 1.0]).astype(complex)
SZ = 2.0 * NN - I2                                    # +1 sur |e>, -1 sur |g>


def eye(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex)


def embed(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """Opérateur à un qubit placé sur `site` dans un registre de n qubits."""
    m = np.array([[1.0 + 0j]])
    for k in range(n):
        m = np.kron(m, op if k == site else I2)
    return m


# ---------------------------------------------------------------- superopérateurs

def super_comm(H: np.ndarray) -> np.ndarray:
    """rho -> -i[H, rho]"""
    I = eye(H.shape[0])
    return -1j * (np.kron(H, I) - np.kron(I, H.T))


def super_diss(L: np.ndarray) -> np.ndarray:
    """rho -> L rho L^dag - (1/2){L^dag L, rho}"""
    I = eye(L.shape[0])
    LdL = L.conj().T @ L
    return np.kron(L, L.conj()) - 0.5 * np.kron(LdL, I) - 0.5 * np.kron(I, LdL.T)


def liouvillien(H: np.ndarray, Ls=()) -> np.ndarray:
    S = super_comm(H)
    for L in Ls:
        S = S + super_diss(L)
    return S


def ligne(op: np.ndarray) -> np.ndarray:
    """Forme linéaire rho -> Tr(op rho) sur vec(rho)."""
    return op.T.reshape(-1)


def augmente(S: np.ndarray, row: np.ndarray, poids: float = 1.0) -> np.ndarray:
    """Générateur [rho ; compteur] : d(compteur)/dt = poids * Tr(op rho)."""
    D = S.shape[0]
    G = np.zeros((D + 1, D + 1), dtype=complex)
    G[:D, :D] = S
    G[D, :D] = poids * row
    return G


# ---------------------------------------------------------------- exponentielle

def expm(A: np.ndarray, tol: float = 1e-16, max_terms: int = 80) -> np.ndarray:
    """Exponentielle par Taylor avec mise à l'échelle (||A/2^s||_1 <= 0.5) et quadrature."""
    nrm = float(np.linalg.norm(A, 1))
    s = 0 if nrm <= 0.5 else int(np.ceil(np.log2(nrm / 0.5)))
    B = A / (2.0 ** s)
    E = np.eye(A.shape[0], dtype=complex)
    T = E.copy()
    for k in range(1, max_terms):
        T = T @ B / k
        E = E + T
        if float(np.linalg.norm(T, 1)) < tol:
            break
    else:
        raise RuntimeError("expm : série non convergée")
    for _ in range(s):
        E = E @ E
    return E


def propager(S: np.ndarray, rho: np.ndarray, t: float, row=None, poids: float = 1.0):
    """Évolue rho sous le générateur constant S pendant t. Renvoie (rho, compteur)."""
    d = rho.shape[0]
    if row is None:
        return (expm(S * t) @ rho.reshape(-1)).reshape(d, d), 0.0
    out = expm(augmente(S, row, poids) * t)[:, :-1] @ rho.reshape(-1)
    return out[:-1].reshape(d, d), float(out[-1].real)


# ---------------------------------------------------------------- contrôles

def verifier(rho: np.ndarray) -> tuple[float, float, float]:
    """(erreur de trace, erreur d'hermiticité, violation de positivité)."""
    herm = (rho + rho.conj().T) / 2.0
    return (abs(float(np.trace(rho).real) - 1.0),
            float(np.linalg.norm(rho - rho.conj().T)),
            max(0.0, -float(np.linalg.eigvalsh(herm)[0])))


# ---------------------------------------------------------------- mesures

def _herm(a: np.ndarray) -> np.ndarray:
    return (a + a.conj().T) / 2.0


def sqrt_psd(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(_herm(A))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T


def purete(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)


def entropie_bits(rho: np.ndarray) -> float:
    w = np.clip(np.linalg.eigvalsh(_herm(rho)), 0.0, None)
    w = w[w > 1e-15]
    return float(-(w * np.log2(w)).sum())


def distance_trace(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.linalg.eigvalsh(_herm(a - b))).sum())


def fidelite(a: np.ndarray, b: np.ndarray) -> float:
    """Fidélité d'Uhlmann F = (Tr sqrt(sqrt(a) b sqrt(a)))^2."""
    sa = sqrt_psd(a)
    w = np.clip(np.linalg.eigvalsh(_herm(sa @ b @ sa)), 0.0, None)
    return float(np.sqrt(w).sum() ** 2)


_YY = np.kron(SY, SY)


def concurrence(rho: np.ndarray) -> float:
    """Concurrence de Wootters, deux qubits (Phys. Rev. Lett. 80, 2245, 1998).
    Précision plancher ~1e-8 (racines de valeurs propres proches de 0)."""
    if rho.shape != (4, 4):
        raise ValueError("concurrence : deux qubits requis")
    ev = np.sqrt(np.abs(np.linalg.eigvals(rho @ (_YY @ rho.conj() @ _YY)).real))
    ev = np.sort(ev)[::-1]
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def neg_log_bits(rho: np.ndarray) -> float:
    """Négativité logarithmique log2 ||rho^{T_0}||_1, bipartition qubit 0 | reste."""
    d = rho.shape[0]
    pt = rho.reshape(2, d // 2, 2, d // 2).transpose(2, 1, 0, 3).reshape(d, d)
    return float(np.log2(max(1.0, np.abs(np.linalg.eigvalsh(_herm(pt))).sum())))


# ============================================================================
# SECTION 2 : modele
# ============================================================================
# -*- coding: utf-8 -*-
"""Modèle d'une chaîne de N qubits ouverte (N = 2..4), simulation et calibration.

Sur chaque pas (dt = 1) le générateur de Lindblad est constant :
    d rho/dt = -i[H_t, rho] + gamma_t D[sigma-_0](rho) + (dephasage) sum_k D[sz_k]
    H_t = J sum_k (s+_k s-_{k+1} + h.c.) + (alpha*echelle/2) sum_k (-1)^k sz_k
          + theta_t sy_{N-1}/2 + env * sum_k eta_k sz_k / 2
    gamma_t = tanh(profil_t * amplitude * decay)        (extraction sur le qubit 0)
    theta_t = gain * tanh(2 max(0, seuil - P_t))        (rétroaction d'état)
    env     = 2 sqrt(P_t (1 - P_t)),  P_t = <N>/N       (bruit multiplicatif)
État initial : excitation sur le qubit 0. La propagation est exacte (exponentielle
d'un générateur augmenté [rho ; compteur de dose]).
Modèle jouet : aucune calibration sur un système physique n'est revendiquée.
"""




@dataclass(frozen=True)
class Config:
    n_qubits: int = 2
    n_graines: int = 5
    n_stress: int = 100
    n_recup: int = 20
    seed: int = 42
    modes: tuple[str, ...] = ("chronique", "aigu")
    dose_reelle_cible: float = 0.50
    tolerance_dose: float = 1e-6
    alphas: tuple[float, ...] = (0.0, 0.20)
    gains: tuple[float, ...] = (0.0, 0.80)
    seuil: float = 0.50
    decay: float = 0.10
    couplage_J: float = 0.30
    echelle_desaccord: float = 1.0
    bruit_interne: float = 0.01
    dephasage: float = 0.0
    tolerance_bilan: float = 1e-10
    calibration_grille_points: int = 40
    calibration_max_amplitude: float = 1000.0
    calibration_iterations: int = 80

    def __post_init__(self) -> None:
        if not 2 <= self.n_qubits <= 4:
            raise ValueError("n_qubits doit être dans [2, 4].")
        if self.n_graines < 1 or self.n_stress < 10 or self.n_recup < 0:
            raise ValueError("Dimensions temporelles ou nombre de graines invalides.")
        if not 0.0 < self.dose_reelle_cible < 1.0:
            raise ValueError("La dose cible doit être dans ]0, 1[ (une seule excitation initiale).")
        if self.tolerance_dose <= 0 or self.tolerance_bilan <= 0:
            raise ValueError("Tolérances invalides.")
        if self.calibration_grille_points < 2 or self.calibration_max_amplitude <= 0:
            raise ValueError("Paramètres de calibration invalides.")
        if any(not -0.2 <= float(a) <= 0.2 for a in self.alphas):
            raise ValueError("alpha doit appartenir à [-0.2, 0.2].")
        if any(float(g) < 0.0 or float(g) > 5.0 for g in self.gains):
            raise ValueError("gain doit appartenir à [0, 5].")
        if self.bruit_interne < 0 or self.dephasage < 0 or self.couplage_J < 0:
            raise ValueError("bruit, dephasage et couplage doivent être >= 0.")
        if not set(self.modes) <= {"chronique", "aigu"} or not self.modes:
            raise ValueError("modes : sous-ensemble non vide de {chronique, aigu}.")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        d = dict(d)
        inconnus = set(d) - set(Config.__dataclass_fields__)
        if inconnus:
            raise ValueError(f"Paramètres inconnus : {sorted(inconnus)}")
        for k in ("alphas", "gains", "modes"):
            if k in d:
                d[k] = tuple(d[k])
        return Config(**d)

    def to_json(self, chemin) -> None:
        Path(chemin).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def from_json(chemin) -> "Config":
        return Config.from_dict(json.loads(Path(chemin).read_text(encoding="utf-8")))


def construire_profil(mode: str, n_stress: int, n_recup: int) -> np.ndarray:
    p = np.zeros(n_stress + n_recup, dtype=float)
    if mode == "chronique":
        p[:n_stress] = 1.0 / n_stress
    elif mode == "aigu":
        p[np.linspace(0, n_stress - 1, 10, dtype=int)] = 0.1
    else:
        raise ValueError(f"Mode inconnu : {mode}")
    return p


class Modele:
    """Pièces constantes du générateur pour un alpha donné."""

    def __init__(self, cfg: Config, alpha: float) -> None:
        n = cfg.n_qubits
        self.n, self.d = n, 2 ** n
        sm = [C.embed(C.SM, k, n) for k in range(n)]
        sp = [C.embed(C.SP, k, n) for k in range(n)]
        sz = [C.embed(C.SZ, k, n) for k in range(n)]
        H0 = sum(cfg.couplage_J * (sp[k] @ sm[k + 1] + sm[k] @ sp[k + 1]) for k in range(n - 1))
        H0 = H0 + 0.5 * alpha * cfg.echelle_desaccord * sum((-1) ** k * sz[k] for k in range(n))
        S0 = C.super_comm(H0)
        if cfg.dephasage > 0.0:
            for k in range(n):
                S0 = S0 + C.super_diss(np.sqrt(cfg.dephasage / 2.0) * sz[k])
        self.S0 = S0
        self.Sy = C.super_comm(0.5 * C.embed(C.SY, n - 1, n))
        self.Sz = np.array([C.super_comm(0.5 * z) for z in sz])
        self.DA = C.super_diss(sm[0])
        nk = [C.embed(C.NN, k, n) for k in range(n)]
        self.masque = [np.real(np.diag(m)) for m in nk]
        self.nTot = sum(nk)
        self.rowA = C.ligne(nk[0])
        tr = C.eye(self.d).reshape(-1)
        for S in (self.S0, self.Sy, self.DA, *self.Sz):
            if np.linalg.norm(tr @ S) > 1e-12:
                raise ValueError("Superopérateur non préservant la trace.")
        psi0 = np.zeros(self.d, dtype=complex)
        psi0[1 << (n - 1)] = 1.0                      # qubit 0 excité, les autres au repos
        self.psi0 = psi0


def simuler(cfg: Config, M: Modele, gain: float, profil: np.ndarray,
            amplitude: float, bruit: np.ndarray, enregistrer: bool = False) -> dict:
    n_t, n, D = len(profil), M.n, M.d * M.d
    if bruit.shape != (n_t, n):
        raise ValueError("Forme du tableau de bruit invalide.")

    ref = np.outer(M.psi0, M.psi0.conj())
    rho = ref.copy()
    dose_tot, Fmin = 0.0, 1.0
    Cmax, ENmax = 0.0, 0.0
    dstress = None
    max_viol, max_resN = 0.0, 0.0
    G = np.zeros((D + 1, D + 1), dtype=complex)
    traj = {k: [] for k in ("t", "phase", "gamma", "theta", "dose_cum", "F", "purete", "EN", "C")}
    for k in range(n):
        traj[f"P_q{k}"] = []

    for t in range(n_t):
        N0 = float(np.trace(M.nTot @ rho).real)
        P = N0 / n
        gam = float(np.clip(np.tanh(profil[t] * amplitude * cfg.decay), 0.0, 1.0))
        theta = gain * float(np.tanh(2.0 * max(0.0, cfg.seuil - P)))
        env = 2.0 * np.sqrt(max(P * (1.0 - P), 0.0))

        G[:D, :D] = M.S0 + theta * M.Sy + np.tensordot(bruit[t] * env, M.Sz, axes=1) + gam * M.DA
        G[D, :D] = gam * M.rowA
        out = C.expm(G)[:, :D] @ rho.reshape(-1)
        rho = out[:D].reshape(M.d, M.d)
        dose = float(out[D].real)
        dose_tot += dose

        et, eh, vp = C.verifier(rho)
        pas = max(et, eh, vp)
        max_viol = max(max_viol, pas)
        if pas > cfg.tolerance_bilan:
            raise AssertionError(f"Validité rompue à t={t} : {pas:.3e}")
        if gain == 0.0:
            res = abs((float(np.trace(M.nTot @ rho).real) - N0) + dose)
            max_resN = max(max_resN, res)
            if res > cfg.tolerance_bilan:
                raise AssertionError(f"Excitations non conservées à t={t} : {res:.3e}")

        F = float(np.real(M.psi0.conj() @ rho @ M.psi0))
        Fmin = min(Fmin, F)
        EN = C.neg_log_bits(rho)
        ENmax = max(ENmax, EN)
        Cc = C.concurrence(rho) if n == 2 else None
        if Cc is not None:
            Cmax = max(Cmax, Cc)
        if t == cfg.n_stress - 1:
            dstress = C.distance_trace(rho, ref)

        if enregistrer:
            diag = np.real(np.diag(rho))
            traj["t"].append(t + 1)
            traj["phase"].append("stress" if t < cfg.n_stress else "recup")
            traj["gamma"].append(gam)
            traj["theta"].append(theta)
            traj["dose_cum"].append(dose_tot)
            traj["F"].append(F)
            traj["purete"].append(C.purete(rho))
            traj["EN"].append(EN)
            traj["C"].append(Cc)
            for k in range(n):
                traj[f"P_q{k}"].append(float(diag @ M.masque[k]))

    df = C.distance_trace(rho, ref)
    dstress = df if dstress is None else dstress
    res = {
        "dose_reelle": dose_tot,
        "F_min": Fmin,
        "F_final": float(np.real(M.psi0.conj() @ rho @ M.psi0)),
        "C_max": Cmax if n == 2 else None,
        "C_final": C.concurrence(rho) if n == 2 else None,
        "EN_max": ENmax,
        "EN_final": C.neg_log_bits(rho),
        "purete_finale": C.purete(rho),
        "entropie_finale": C.entropie_bits(rho),
        "distance_stress": dstress,
        "distance_finale": df,
        "amelioration_recuperation": dstress - df,
        "max_violation": max_viol,
        "max_residu_N": max_resN if gain == 0.0 else None,
    }
    if enregistrer:
        res["trajectoire"] = traj
    return res


def calibrer_dose(cfg: Config, M: Modele, gain: float, profil: np.ndarray,
                  bruit: np.ndarray) -> tuple[Optional[float], dict]:
    """Amplitude minimale (parmi les solutions détectées) donnant dose_reelle = cible."""
    cible, tol = cfg.dose_reelle_cible, cfg.tolerance_dose
    f = lambda a: simuler(cfg, M, gain, profil, float(a), bruit)
    ag = np.linspace(0.0, cfg.calibration_max_amplitude, cfg.calibration_grille_points)
    dg = np.array([f(a)["dose_reelle"] for a in ag])
    eg = dg - cible
    nonmono = bool(np.any(np.diff(dg) < -tol))          # effet Zénon possible

    brackets = []
    for i in range(len(ag)):
        if abs(eg[i]) <= tol:
            brackets.append((float(ag[i]), float(ag[i]), 0))
        if i < len(ag) - 1 and eg[i] * eg[i + 1] < 0.0:
            brackets.append((float(ag[i]), float(ag[i + 1]), 1))
    if not brackets:
        return None, {"statut": "non_encadree", "non_monotone": nonmono,
                      "dose_min": float(dg.min()), "dose_max": float(dg.max())}

    sols = []
    for lo, hi, kind in brackets:
        if kind == 0:
            sols.append((lo, f(lo)))
            continue
        flo = f(lo)["dose_reelle"] - cible
        best = None
        for _ in range(cfg.calibration_iterations):
            mid = 0.5 * (lo + hi)
            r = f(mid)
            fm = r["dose_reelle"] - cible
            best = (mid, r)
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
# SECTION 3 : validation
# ============================================================================
# -*- coding: utf-8 -*-
"""Batterie de validation : solutions analytiques, intégrateur indépendant, invariants.

Chaque test renvoie (mesure, seuil) ; il réussit si mesure <= seuil.
"""




def _rng():
    return np.random.default_rng(12345)


def _herm_alea(rng, d):
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return (A + A.conj().T) / 2.0


def _rho_alea(rng, d):
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    r = A @ A.conj().T
    return r / np.trace(r).real


def _rk4(H, Ls, rho, T, n):
    """Intégrateur RK4 direct de l'équation maîtresse (sans superopérateurs)."""
    h = T / n

    def f(r):
        out = -1j * (H @ r - r @ H)
        for L in Ls:
            LdL = L.conj().T @ L
            out = out + L @ r @ L.conj().T - 0.5 * (LdL @ r + r @ LdL)
        return out

    for _ in range(n):
        k1 = f(rho)
        k2 = f(rho + 0.5 * h * k1)
        k3 = f(rho + 0.5 * h * k2)
        k4 = f(rho + h * k3)
        rho = rho + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return rho


def _ket(i, d):
    v = np.zeros(d, dtype=complex)
    v[i] = 1.0
    return np.outer(v, v.conj())


# ------------------------------------------------------------------ tests

def t_expm_unitaire():
    H = _herm_alea(_rng(), 4)
    w, V = np.linalg.eigh(H)
    ref = (V * np.exp(-1j * w * 3.0)) @ V.conj().T
    return float(np.abs(C.expm(-1j * H * 3.0) - ref).max()), 1e-12


def t_rabi():
    Om, T = 0.7, 5.3
    r, _ = C.propager(C.liouvillien(0.5 * Om * C.SX), _ket(0, 2), T)
    return abs(float(r[1, 1].real) - np.sin(Om * T / 2.0) ** 2), 1e-12


def t_amortissement_et_dose():
    g, T = 0.4, 3.7
    S = C.liouvillien(np.zeros((2, 2), dtype=complex), [np.sqrt(g) * C.SM])
    r, dose = C.propager(S, _ket(1, 2), T, row=C.ligne(C.NN), poids=g)
    e1 = abs(float(r[1, 1].real) - np.exp(-g * T))
    e2 = abs(dose - (1.0 - np.exp(-g * T)))
    return max(e1, e2), 1e-12


def t_echange_deux_qubits():
    J, T = 0.3, 2.1
    sp, sm = np.kron(C.SP, C.I2), np.kron(C.SM, C.I2)
    spb, smb = np.kron(C.I2, C.SP), np.kron(C.I2, C.SM)
    H = J * (sp @ smb + sm @ spb)
    r, _ = C.propager(C.liouvillien(H), _ket(2, 4), T)
    return abs(float(r[2, 2].real) - np.cos(J * T) ** 2), 1e-12


def t_semigroupe():
    rng = _rng()
    S = C.liouvillien(_herm_alea(rng, 4),
                      [0.4 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) for _ in range(2)])
    E1, E2 = C.expm(S * 0.7), C.expm(S * 1.4)
    return float(np.abs(E1 @ E1 - E2).max()), 1e-12


def t_rk4_croise():
    rng = _rng()
    H = _herm_alea(rng, 4) * 0.5
    Ls = [0.3 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) for _ in range(2)]
    rho0 = _rho_alea(rng, 4)
    a, _ = C.propager(C.liouvillien(H, Ls), rho0, 1.0)
    b = _rk4(H, Ls, rho0, 1.0, 4000)
    return float(np.abs(a - b).max()), 1e-9


def t_positivite_aleatoire():
    rng = _rng()
    S = C.liouvillien(_herm_alea(rng, 4),
                      [0.5 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) for _ in range(3)])
    E = C.expm(S * 0.3)
    rho, worst = _rho_alea(rng, 4), 0.0
    for _ in range(50):
        rho = (E @ rho.reshape(-1)).reshape(4, 4)
        worst = max(worst, *C.verifier(rho))
    return worst, 1e-12


def t_concurrence():
    psi = np.zeros(4, dtype=complex)
    psi[1], psi[2] = 1 / np.sqrt(2), -1 / np.sqrt(2)         # singulet
    sing = np.outer(psi, psi.conj())
    p = 0.7
    werner = p * sing + (1 - p) * np.eye(4) / 4
    e1 = abs(C.concurrence(sing) - 1.0)
    e2 = abs(C.concurrence(_ket(0, 4)) - 0.0)
    e3 = abs(C.concurrence(werner) - max(0.0, (3 * p - 1) / 2))
    return max(e1, e2, e3), 1e-6                              # plancher ~1e-8, cf. doc


def t_neg_log():
    psi = np.zeros(4, dtype=complex)
    psi[1], psi[2] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    sing = np.outer(psi, psi.conj())
    p = 0.7
    werner = p * sing + (1 - p) * np.eye(4) / 4
    tn = (3 * (1 + p) + abs(1 - 3 * p)) / 4                   # norme de trace de la transposée partielle
    e1 = abs(C.neg_log_bits(sing) - 1.0)
    e2 = abs(C.neg_log_bits(_ket(0, 4)))
    e3 = abs(C.neg_log_bits(werner) - np.log2(tn))
    return max(e1, e2, e3), 1e-12


def t_entropie_fidelite():
    psi = np.zeros(4, dtype=complex)
    psi[1], psi[2] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    sing = np.outer(psi, psi.conj())
    p = 0.7
    werner = p * sing + (1 - p) * np.eye(4) / 4
    e1 = abs(C.entropie_bits(np.eye(4) / 4) - 2.0)
    e2 = abs(C.entropie_bits(sing))
    e3 = abs(C.fidelite(sing, sing) - 1.0)
    e4 = abs(C.fidelite(_ket(0, 4), _ket(3, 4)))
    e5 = abs(C.fidelite(sing, werner) - (p + (1 - p) / 4))     # <psi|rho|psi>
    return max(e1, e2, e3, e4, e5), 1e-9


def _cfg_petit(**kw):
    base = dict(n_stress=30, n_recup=10, calibration_grille_points=20, n_graines=1)
    base.update(kw)
    return Config(**base)


def t_reproductibilite():
    cfg = _cfg_petit()
    M = Modele(cfg, 0.2)
    profil = construire_profil("chronique", cfg.n_stress, cfg.n_recup)
    bruit = np.random.default_rng(7).normal(0, cfg.bruit_interne, size=(len(profil), cfg.n_qubits))
    a = simuler(cfg, M, 0.8, profil, 9.0, bruit)
    b = simuler(cfg, M, 0.8, profil, 9.0, bruit)
    return float(a != b), 0.0


def t_conservation_simulation():
    worst = 0.0
    for nq in (2, 3):
        cfg = _cfg_petit(n_qubits=nq, dephasage=0.02)
        M = Modele(cfg, 0.1)
        for mode in ("chronique", "aigu"):
            profil = construire_profil(mode, cfg.n_stress, cfg.n_recup)
            bruit = np.random.default_rng(3).normal(0, cfg.bruit_interne, size=(len(profil), nq))
            r = simuler(cfg, M, 0.0, profil, 6.0, bruit)
            worst = max(worst, r["max_violation"], r["max_residu_N"])
    return worst, 1e-10


def t_calibration():
    cfg = _cfg_petit()
    M = Modele(cfg, 0.0)
    profil = construire_profil("aigu", cfg.n_stress, cfg.n_recup)
    bruit = np.random.default_rng(5).normal(0, cfg.bruit_interne, size=(len(profil), cfg.n_qubits))
    a, r = calibrer_dose(cfg, M, 0.0, profil, bruit)
    if a is None:
        return 1.0, 0.0
    return abs(r["dose_reelle"] - cfg.dose_reelle_cible), cfg.tolerance_dose


def t_config_invalide():
    mauvais = [dict(alphas=(0.5,)), dict(n_qubits=9), dict(dose_reelle_cible=1.2),
               dict(gains=(-1.0,)), dict(modes=("inconnu",)), dict(bruit_interne=-1.0)]
    acceptes = 0
    for kw in mauvais:
        try:
            Config(**kw)
            acceptes += 1
        except ValueError:
            pass
    return float(acceptes), 0.0


def t_hamiltonien_effectif():
    """Sans rétroaction ni bruit, l'état reste dans le sous-espace à une excitation :
    solution exacte par l'hamiltonien effectif non hermitien H0 - i(gamma/2) n_A."""
    alpha, amp = 0.2, 7.0
    cfg = Config(n_stress=40, n_recup=10, alphas=(alpha,))
    M = Modele(cfg, alpha)
    profil = construire_profil("chronique", cfg.n_stress, cfg.n_recup)
    r = simuler(cfg, M, 0.0, profil, amp, np.zeros((len(profil), 2)))
    sp0, sm0 = C.embed(C.SP, 0, 2), C.embed(C.SM, 0, 2)
    sp1, sm1 = C.embed(C.SP, 1, 2), C.embed(C.SM, 1, 2)
    H0 = cfg.couplage_J * (sp0 @ sm1 + sm0 @ sp1) \
        + 0.5 * alpha * cfg.echelle_desaccord * (C.embed(C.SZ, 0, 2) - C.embed(C.SZ, 1, 2))
    nA = C.embed(C.NN, 0, 2)
    psi = np.zeros(4, dtype=complex)
    psi[2] = 1.0
    for p in profil:
        g = float(np.tanh(p * amp * cfg.decay))
        psi = C.expm(-1j * (H0 - 0.5j * g * nA)) @ psi
    e1 = abs(r["dose_reelle"] - (1.0 - float(np.vdot(psi, psi).real)))
    e2 = abs(r["F_final"] - abs(psi[2]) ** 2)
    return max(e1, e2), 1e-11


def t_config_json():
    cfg = Config(n_qubits=3, alphas=(0.1, -0.1), modes=("aigu",))
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "cfg.json"
        cfg.to_json(f)
        identique = Config.from_json(f) == cfg
    try:
        Config.from_dict({"parametre_inconnu": 1})
        rejet = False
    except ValueError:
        rejet = True
    return float(not (identique and rejet)), 0.0


def t_reprise_campagne():
    cfg = Config(n_graines=1, n_stress=20, n_recup=5, gains=(0.0,), alphas=(0.0,),
                 modes=("aigu",), calibration_grille_points=10)
    with tempfile.TemporaryDirectory() as d:
        sortie = Path(d)
        a = K.executer(cfg, sortie, False, lambda *_: None)
        b = K.executer(cfg, sortie, True, lambda *_: None)
        n_lignes = len((sortie / "points.jsonl").read_text(encoding="utf-8").strip().splitlines())
    ok = len(a) == len(b) == 1 and a[0]["amplitude"] == b[0]["amplitude"] and n_lignes == 2
    return float(not ok), 0.0


TESTS = [
    ("expm : unitaire vs diagonalisation", t_expm_unitaire),
    ("Rabi : P_e = sin^2(Omega t / 2)", t_rabi),
    ("Amortissement : P_e = exp(-g t) et dose exacte", t_amortissement_et_dose),
    ("Échange 2 qubits : P = cos^2(J t)", t_echange_deux_qubits),
    ("Semi-groupe : E(t)E(t) = E(2t)", t_semigroupe),
    ("Croisé : expm vs RK4 direct de l'équation maîtresse", t_rk4_croise),
    ("CPTP : trace, hermiticité, positivité (50 pas)", t_positivite_aleatoire),
    ("Concurrence : singulet, produit, Werner", t_concurrence),
    ("Négativité log. : singulet, produit, Werner", t_neg_log),
    ("Entropie et fidélité : cas connus", t_entropie_fidelite),
    ("Reproductibilité : même graine, même résultat", t_reproductibilite),
    ("Simulation : conservation (2 et 3 qubits, gain 0)", t_conservation_simulation),
    ("Calibration : dose cible atteinte", t_calibration),
    ("Exact : hamiltonien effectif non hermitien (1 excitation)", t_hamiltonien_effectif),
    ("Config : aller-retour JSON, clés inconnues rejetées", t_config_json),
    ("Campagne : reprise après interruption", t_reprise_campagne),
    ("Config : entrées invalides rejetées", t_config_invalide),
]


def lancer() -> bool:
    print("=" * 78)
    print(" VALIDATION")
    print("=" * 78)
    tout_ok = True
    for nom, fn in TESTS:
        t0 = time.perf_counter()
        try:
            mesure, seuil = fn()
            ok = bool(mesure <= seuil)
            etat = "PASS" if ok else "FAIL"
            detail = f"mesure={mesure:.2e} seuil={seuil:.0e}"
        except Exception as exc:                                # un test défaillant ne masque pas les autres
            ok, etat, detail = False, "ERR ", f"{type(exc).__name__}: {exc}"
        tout_ok &= ok
        print(f"[{etat}] {nom:<52} {detail} ({time.perf_counter() - t0:.2f}s)")
    print("=" * 78)
    print("RÉSULTAT :", "tous les tests passent" if tout_ok else "ÉCHEC")
    return tout_ok


# ============================================================================
# SECTION 4 : campagne
# ============================================================================
# -*- coding: utf-8 -*-
"""Campagnes : manifeste de reproductibilité, reprise après interruption, statistiques."""




log = logging.getLogger(__name__)
METRIQUES = ("amplitude", "dose_reelle", "F_min", "F_final", "C_max", "C_final", "EN_max", "EN_final",
             "purete_finale", "entropie_finale", "distance_stress", "distance_finale",
             "amelioration_recuperation", "max_violation", "max_residu_N")


def empreinte(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def manifeste(cfg: Config) -> dict:
    return {"version": VERSION, "config_hash": empreinte(cfg), "config": cfg.to_dict(),
            "numpy": np.__version__, "python": sys.version.split()[0], "plateforme": platform.platform(),
            "debut": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _tâches(cfg: Config):
    seeds = np.random.default_rng(cfg.seed).integers(0, 2**31 - 1, size=cfg.n_graines)
    for mode in cfg.modes:
        for gain in cfg.gains:
            for alpha in cfg.alphas:
                for i, s in enumerate(seeds):
                    yield mode, float(gain), float(alpha), i, int(s)


def _cle(mode, gain, alpha, i) -> str:
    return f"{mode}|{gain:.6g}|{alpha:.6g}|{i}"


def _lire_points(chemin: Path, cfg: Config) -> dict:
    faits: dict = {}
    if not chemin.exists():
        return faits
    with chemin.open(encoding="utf-8") as fh:
        for n, ligne in enumerate(fh):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                obj = json.loads(ligne)
            except json.JSONDecodeError:
                continue                                  # dernière ligne tronquée par une interruption
            if n == 0:
                if obj.get("manifeste", {}).get("config_hash") != empreinte(cfg):
                    raise RuntimeError("Le point de reprise correspond à une autre configuration.")
                continue
            if "cle" in obj:
                faits[obj["cle"]] = obj
    return faits


def executer(cfg: Config, sortie: Path | None, reprendre: bool, parler=None) -> list[dict]:
    parler = parler or log.info
    points = (sortie / "points.jsonl") if sortie else None
    faits: dict = {}
    if sortie:
        sortie.mkdir(parents=True, exist_ok=True)
        if reprendre:
            faits = _lire_points(points, cfg)
        elif points.exists():
            points.unlink()
        if not points.exists():
            points.write_text(json.dumps({"manifeste": manifeste(cfg)}) + "\n", encoding="utf-8")

    taches = list(_tâches(cfg))
    modeles: dict = {}
    profils = {m: construire_profil(m, cfg.n_stress, cfg.n_recup) for m in cfg.modes}
    resultats = []
    for k, (mode, gain, alpha, i, seed) in enumerate(taches, 1):
        cle = _cle(mode, gain, alpha, i)
        if cle in faits:
            resultats.append(faits[cle])
            parler(f"[{k}/{len(taches)}] {cle} : repris")
            continue
        if alpha not in modeles:
            modeles[alpha] = Modele(cfg, alpha)
        profil = profils[mode]
        bruit = np.random.default_rng(seed).normal(0.0, cfg.bruit_interne, size=(len(profil), cfg.n_qubits))
        t0 = time.perf_counter()
        a, r = calibrer_dose(cfg, modeles[alpha], gain, profil, bruit)
        ligne = {"cle": cle, "mode": mode, "gain": gain, "alpha": alpha, "graine": i,
                 "statut": "ok" if a is not None else r.get("statut", "echec"),
                 "duree_s": time.perf_counter() - t0}
        if a is not None:
            ligne["amplitude"] = float(a)
            for m in METRIQUES[1:]:
                v = r.get(m)
                ligne[m] = None if v is None else float(v)
            ligne["non_monotone"] = bool(r["calibration"]["non_monotone"])
        resultats.append(ligne)
        if points:
            with points.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ligne) + "\n")
        parler(f"[{k}/{len(taches)}] {cle} : {ligne['statut']} ({ligne['duree_s']:.1f}s)")
    return resultats


def agreger(resultats: list[dict]) -> list[dict]:
    groupes: dict = {}
    for r in resultats:
        groupes.setdefault((r["mode"], r["gain"], r["alpha"]), []).append(r)
    out = []
    for (mode, gain, alpha), rows in groupes.items():
        ok = [r for r in rows if r["statut"] == "ok"]
        ligne = {"mode": mode, "gain": gain, "alpha": alpha, "n_ok": len(ok), "n_total": len(rows),
                 "non_monotone": sum(1 for r in ok if r.get("non_monotone"))}
        for m in METRIQUES:
            v = [r[m] for r in ok if r.get(m) is not None]
            ligne[m + "_moy"] = float(np.mean(v)) if v else None
            ligne[m + "_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        for m in ("max_violation", "max_residu_N"):
            v = [r[m] for r in ok if r.get(m) is not None]
            ligne[m + "_max"] = float(np.max(v)) if v else None
        out.append(ligne)
    return out


def _f(x, largeur=6, nd=3):
    return " " * (largeur - 1) + "-" if x is None else f"{x:{largeur}.{nd}f}"


def afficher(cfg: Config, agg: list[dict], duree: float) -> None:
    deux = cfg.n_qubits == 2
    cle_int = "C_max" if deux else "EN_max"
    print("=" * 96)
    print(f" ÉDIFICE QUANTIQUE {VERSION} | qubits={cfg.n_qubits} | graines={cfg.n_graines} | "
          f"cible={cfg.dose_reelle_cible} | empreinte={empreinte(cfg)}")
    print("=" * 96)
    mode_courant = None
    for a in agg:
        if a["mode"] != mode_courant:
            mode_courant = a["mode"]
            print(f"\nREGIME : {mode_courant.upper()}")
            print(f"{'Gain':<5}|{'Alpha':<6}|{'n':<5}|{'Amp':>8}|{'F_min':>6}|{cle_int + ' (moy ± σ)':>18}|"
                  f"{'Pureté':>7}|{'Viol.max':>9}|{'Res.N':>8}")
            print("-" * 96)
        if a["n_ok"] == 0:
            print(f"{a['gain']:4.2f} |{a['alpha']:5.2f} |{a['n_ok']}/{a['n_total']} | cible non atteinte")
            continue
        ci, cs = a[cle_int + "_moy"], a[cle_int + "_std"]
        inter = f"{ci:6.3f} ± {cs:5.3f}" if cs is not None else f"{ci:6.3f} ±   -  "
        rn = a["max_residu_N_max"]
        print(f"{a['gain']:4.2f} |{a['alpha']:5.2f} |{a['n_ok']}/{a['n_total']:<3}|{a['amplitude_moy']:8.3f}|"
              f"{_f(a['F_min_moy'])}|{inter:>18}|{_f(a['purete_finale_moy'], 7)}|"
              f"{a['max_violation_max']:9.1e}|{('n/a' if rn is None else f'{rn:.1e}'):>8}")
    print("=" * 96)
    print(f"Terminé en {duree:.1f} s.")


def exporter(cfg: Config, resultats: list[dict], agg: list[dict], sortie: Path) -> None:
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "synthese.json").write_text(
        json.dumps({"manifeste": manifeste(cfg), "synthese": agg}, indent=2), encoding="utf-8")
    for nom, lignes in (("synthese.csv", agg), ("resultats_bruts.csv", resultats)):
        if not lignes:
            continue
        champs = sorted({k for l in lignes for k in l}, key=lambda k: (k not in ("mode", "gain", "alpha"), k))
        with (sortie / nom).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=champs)
            w.writeheader()
            w.writerows(lignes)


# ============================================================================
# SECTION 5 : cli
# ============================================================================
# -*- coding: utf-8 -*-
"""Interface en ligne de commande : valider | config | campagne | trajectoire."""



log = logging.getLogger("edifice")


def _liste(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip() != "")


def _cfg_depuis(a) -> Config:
    """--config fait autorité : les autres options de modèle sont alors ignorées."""
    if getattr(a, "config", None):
        return Config.from_json(Path(a.config).expanduser())
    kw: dict = {"n_qubits": a.qubits, "dose_reelle_cible": a.cible, "seed": a.seed,
                "bruit_interne": a.bruit, "dephasage": a.dephasage}
    if a.graines is not None:
        kw["n_graines"] = a.graines
    if getattr(a, "rapide", False):
        kw.update(n_graines=min(kw.get("n_graines", 2), 2), calibration_grille_points=20)
    if getattr(a, "alphas", None):
        kw["alphas"] = _liste(a.alphas)
    if getattr(a, "gains", None):
        kw["gains"] = _liste(a.gains)
    if getattr(a, "modes", None):
        kw["modes"] = tuple(a.modes.split(","))
    return Config(**kw)


def _commun(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None,
                   help="fichier JSON de configuration (prioritaire sur les options ci-dessous)")
    p.add_argument("--qubits", type=int, default=2, help="2 à 4 (4 qubits : lent sur téléphone)")
    p.add_argument("--cible", type=float, default=0.50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graines", type=int, default=None)
    p.add_argument("--bruit", type=float, default=0.01)
    p.add_argument("--dephasage", type=float, default=0.0)


def cmd_valider(a) -> int:
    return 0 if validation.lancer() else 1


def cmd_config(a) -> int:
    cfg = _cfg_depuis(a)
    out = Path(a.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out)
    print(f"Configuration écrite : {out}")
    return 0


def cmd_campagne(a) -> int:
    cfg = _cfg_depuis(a)
    sortie = Path(a.output).expanduser() if a.output else None
    if a.reprendre and not sortie:
        print("--reprendre exige --output", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    res = K.executer(cfg, sortie, a.reprendre)
    agg = K.agreger(res)
    K.afficher(cfg, agg, time.perf_counter() - t0)
    if sortie:
        K.exporter(cfg, res, agg, sortie)
        print(f"Exports et reprise : {sortie}")
    return 0


def cmd_trajectoire(a) -> int:
    cfg = _cfg_depuis(a)
    M = Modele(cfg, float(a.alpha))
    profil = construire_profil(a.mode, cfg.n_stress, cfg.n_recup)
    bruit = np.random.default_rng(cfg.seed).normal(0.0, cfg.bruit_interne, size=(len(profil), cfg.n_qubits))
    amp = a.amplitude
    if amp is None:
        amp, _ = calibrer_dose(cfg, M, a.gain, profil, bruit)
        if amp is None:
            print("Calibration impossible pour cette configuration.", file=sys.stderr)
            return 1
        print(f"Amplitude calibrée : {amp:.6f}")
    r = simuler(cfg, M, a.gain, profil, float(amp), bruit, enregistrer=True)
    tr = r["trajectoire"]
    out = Path(a.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    champs = list(tr.keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(champs)
        for i in range(len(tr["t"])):
            w.writerow(["" if tr[c][i] is None else tr[c][i] for c in champs])
    print(f"Dose finale {r['dose_reelle']:.6f} | violation max {r['max_violation']:.1e} | écrit : {out}")
    return 0


def construire_parseur() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="edifice", description="Édifice quantique : qubits ouverts, Lindblad.")
    ap.add_argument("--version", action="version", version=f"edifice {__version__}")
    ap.add_argument("-v", "--verbeux", action="store_true", help="journal détaillé")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("valider", help="batterie de validation analytique et croisée").set_defaults(f=cmd_valider)

    pf = sub.add_parser("config", help="écrit un fichier de configuration JSON modifiable")
    _commun(pf)
    pf.add_argument("--output", type=str, required=True)
    pf.set_defaults(f=cmd_config)

    pc = sub.add_parser("campagne", help="plan expérimental calibré, avec reprise")
    _commun(pc)
    pc.add_argument("--rapide", action="store_true", help="2 graines, grille 20")
    pc.add_argument("--alphas", type=str, default=None, help="ex. -0.2,-0.1,0,0.1,0.2")
    pc.add_argument("--gains", type=str, default=None, help="ex. 0,0.4,0.8")
    pc.add_argument("--modes", type=str, default=None, help="chronique,aigu")
    pc.add_argument("--output", type=str, default=None, help="dossier (CSV, JSON, points de reprise)")
    pc.add_argument("--reprendre", action="store_true", help="reprend un calcul interrompu")
    pc.set_defaults(f=cmd_campagne)

    pt = sub.add_parser("trajectoire", help="série temporelle d'une simulation, en CSV")
    _commun(pt)
    pt.add_argument("--mode", choices=("chronique", "aigu"), default="chronique")
    pt.add_argument("--gain", type=float, default=0.0)
    pt.add_argument("--alpha", type=float, default=0.0)
    pt.add_argument("--amplitude", type=float, default=None, help="sinon calibrée sur la dose cible")
    pt.add_argument("--output", type=str, required=True)
    pt.set_defaults(f=cmd_trajectoire)
    return ap


def main(argv=None) -> int:
    a = construire_parseur().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbeux else logging.INFO,
                        format="%(message)s", stream=sys.stdout, force=True)
    try:
        return a.f(a)
    except (ValueError, RuntimeError, AssertionError, OSError, json.JSONDecodeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
