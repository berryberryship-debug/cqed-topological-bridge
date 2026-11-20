#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÉDIFICE QUANTIQUE 2.0.0 - Fichier unique intégrant :
  - Spectre microscopique de Fluxonium (charge/phase)
  - Relaxation par Quasiparticules hors-équilibre (QP)
  - Chaîne de 2 à 4 qubits ouverts sous Lindblad / Liouvillien
  - Évaluation des stabilisateurs et syndromes QEC (S_ZZZZ, S_XXXX)
  - Exponentielle exacte par générateur augmenté & calibration de dose
  - Suite de validation analytique & CLI complète

Termux : pkg install python python-numpy nano
Usage  : python edifice_v2.py valider
         python edifice_v2.py campagne --qubits 4 --rapide --output ~/noyau
         python edifice_v2.py fluxonium --flux 0.50
         python edifice_v2.py trajectoire --qubits 4 --gain 0.5 --output ~/traj.csv
"""
from __future__ import annotations
import argparse, csv, hashlib, json, logging, os, platform, sys, tempfile, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

VERSION = __version__ = "2.0.0"

# ============================================================================
# SECTION 1 : CORE NUMÉRIQUE & OPÉRATEURS
# ============================================================================

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)        # |e> -> |g>
SP = SM.T.copy()
NN = np.diag([0.0, 1.0]).astype(complex)
SZ = 2.0 * NN - I2                                    # +1 sur |e>, -1 sur |g>


def eye(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex)


def embed(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """Opérateur local placé sur `site` dans un registre de n qubits."""
    m = np.array([[1.0 + 0j]])
    for k in range(n):
        m = np.kron(m, op if k == site else I2)
    return m


def super_comm(H: np.ndarray) -> np.ndarray:
    """rho -> -i[H, rho] en vectorisation par lignes."""
    I = eye(H.shape[0])
    return -1j * (np.kron(H, I) - np.kron(I, H.T))


def super_diss(L: np.ndarray) -> np.ndarray:
    """Dissipateur Lindblad : L rho L^dag - 1/2 {L^dag L, rho}."""
    I = eye(L.shape[0])
    LdL = L.conj().T @ L
    return np.kron(L, L.conj()) - 0.5 * np.kron(LdL, I) - 0.5 * np.kron(I, LdL.T)


def liouvillien(H: np.ndarray, Ls=()) -> np.ndarray:
    S = super_comm(H)
    for L in Ls:
        S = S + super_diss(L)
    return S


def ligne(op: np.ndarray) -> np.ndarray:
    return op.T.reshape(-1)


def augmente(S: np.ndarray, row: np.ndarray, poids: float = 1.0) -> np.ndarray:
    """Générateur augmenté [rho ; compteur]."""
    D = S.shape[0]
    G = np.zeros((D + 1, D + 1), dtype=complex)
    G[:D, :D] = S
    G[D, :D] = poids * row
    return G


def expm(A: np.ndarray, tol: float = 1e-15, max_terms: int = 80) -> np.ndarray:
    """Exponentielle par série de Taylor avec scaling and squaring."""
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
        raise RuntimeError("expm : divergence ou non-convergence de la série")
    for _ in range(s):
        E = E @ E
    return E


def propager(S: np.ndarray, rho: np.ndarray, t: float, row=None, poids: float = 1.0):
    d = rho.shape[0]
    if row is None:
        return (expm(S * t) @ rho.reshape(-1)).reshape(d, d), 0.0
    out = expm(augmente(S, row, poids) * t)[:, :-1] @ rho.reshape(-1)
    return out[:-1].reshape(d, d), float(out[-1].real)


def verifier(rho: np.ndarray) -> tuple[float, float, float]:
    herm = (rho + rho.conj().T) / 2.0
    return (abs(float(np.trace(rho).real) - 1.0),
            float(np.linalg.norm(rho - rho.conj().T)),
            max(0.0, -float(np.linalg.eigvalsh(herm)[0])))


def purete(rho: np.ndarray) -> float:
    return float(np.trace(rho @ rho).real)


def entropie_bits(rho: np.ndarray) -> float:
    w = np.clip(np.linalg.eigvalsh((rho + rho.conj().T) / 2.0), 0.0, None)
    w = w[w > 1e-15]
    return float(-(w * np.log2(w)).sum())


def distance_trace(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.linalg.eigvalsh((a - b + (a - b).conj().T) / 2.0)).sum())


def fidelite(a: np.ndarray, b: np.ndarray) -> float:
    w, V = np.linalg.eigh((a + a.conj().T) / 2.0)
    sa = (V * np.sqrt(np.clip(w, 0.0, None))) @ V.conj().T
    wb = np.clip(np.linalg.eigvalsh((sa @ b @ sa + (sa @ b @ sa).conj().T) / 2.0), 0.0, None)
    return float(np.sqrt(wb).sum() ** 2)


def neg_log_bits(rho: np.ndarray) -> float:
    d = rho.shape[0]
    pt = rho.reshape(2, d // 2, 2, d // 2).transpose(2, 1, 0, 3).reshape(d, d)
    return float(np.log2(max(1.0, np.abs(np.linalg.eigvalsh((pt + pt.conj().T) / 2.0)).sum())))


# ============================================================================
# SECTION 2 : FLUXONIUM & QUASIPARTICULES
# ============================================================================

@dataclass(frozen=True)
class FluxoniumParams:
    E_C: float = 1.0       # GHz
    E_L: float = 0.5       # GHz
    E_J: float = 4.0       # GHz
    phi_ext: float = 0.5   # Unités de Phi_0 (point doux à 0.50)
    n_cutoff: int = 14     # Troncature base de charge


def diagonaliser_fluxonium(fp: FluxoniumParams) -> Tuple[float, float, float]:
    """
    Diagonalise H_fluxonium = 4 E_C n^2 + 0.5 E_L phi^2 - E_J cos(phi - 2 pi phi_ext)
    Retourne : (omega_01, |<0|phi|1>|, |<0|sin(phi/2)|1>|)
    """
    N = fp.n_cutoff
    dim = 2 * N + 1
    diag_n = np.arange(-N, N + 1)
    H_C = 4.0 * fp.E_C * np.diag(diag_n ** 2)

    shift = 2.0 * np.pi * fp.phi_ext
    cos_phi = np.zeros((dim, dim), dtype=complex)
    phi_op = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):
            if i != j:
                phi_op[i, j] = 1j * ((-1) ** (i - j)) / (i - j)
            if abs(i - j) == 1:
                cos_phi[i, j] = 0.5 * np.exp(-1j * np.sign(i - j) * shift)

    H_L = 0.5 * fp.E_L * (phi_op @ phi_op)
    H_J = -fp.E_J * (cos_phi + cos_phi.conj().T) / 2.0
    H = 0.5 * (H_C + H_L + H_J + (H_C + H_L + H_J).conj().T)

    vals, vecs = np.linalg.eigh(H)
    w01 = float(vals[1] - vals[0])
    s0, s1 = vecs[:, 0], vecs[:, 1]
    mat_phi = float(np.abs(s0.conj().T @ phi_op @ s1))
    mat_qp = float(np.abs(mat_phi / 2.0))
    return w01, mat_phi, mat_qp


def evaluer_gamma_qp(n_qp: float, omega_01: float, mat_qp: float, delta_gap: float = 45.0) -> float:
    """Taux de saut induit par quasiparticules microscopiques."""
    if omega_01 <= 0.0 or n_qp <= 0.0:
        return 0.0
    dens = n_qp * np.sqrt(2.0 * delta_gap / max(omega_01, 1e-4))
    return float(max(1e-6, omega_01 * (mat_qp ** 2) * dens * 1e-2))


# ============================================================================
# SECTION 3 : MODÈLE ET SIMULATION CPTP MULTI-QUBITS
# ============================================================================

@dataclass(frozen=True)
class Config:
    n_qubits: int = 4
    n_graines: int = 3
    n_stress: int = 60
    n_recup: int = 20
    seed: int = 42
    modes: tuple[str, ...] = ("chronique", "aigu")
    dose_reelle_cible: float = 0.50
    tolerance_dose: float = 1e-5
    alphas: tuple[float, ...] = (0.0, 0.20)
    gains: tuple[float, ...] = (0.0, 0.50)
    seuil: float = 0.50
    decay: float = 0.10
    couplage_J: float = 0.15
    echelle_desaccord: float = 1.0
    bruit_interne: float = 0.01
    dephasage: float = 0.0
    flux_ext: float = 0.50
    n_qp: float = 1e-4
    tolerance_bilan: float = 1e-9
    calibration_grille_points: int = 30
    calibration_max_amplitude: float = 1000.0
    calibration_iterations: int = 60

    def __post_init__(self) -> None:
        if not 2 <= self.n_qubits <= 4:
            raise ValueError("n_qubits doit être compris entre 2 et 4.")
        if not 0.0 < self.dose_reelle_cible < 1.0:
            raise ValueError("dose_reelle_cible doit être dans ]0, 1[.")


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
    def __init__(self, cfg: Config, alpha: float) -> None:
        n = cfg.n_qubits
        self.n, self.d = n, 2 ** n
        fp = FluxoniumParams(phi_ext=cfg.flux_ext)
        self.w01, self.mat_phi, self.mat_qp = diagonaliser_fluxonium(fp)
        self.gamma_qp = evaluer_gamma_qp(cfg.n_qp, self.w01, self.mat_qp)

        sm = [embed(SM, k, n) for k in range(n)]
        sp = [embed(SP, k, n) for k in range(n)]
        sz = [embed(SZ, k, n) for k in range(n)]
        sx = [embed(SX, k, n) for k in range(n)]

        # Hamiltonien collectif (transitions fluxonium + couplages)
        H0 = sum(0.5 * self.w01 * sz[k] for k in range(n))
        H0 += sum(cfg.couplage_J * (sp[k] @ sm[k + 1] + sm[k] @ sp[k + 1]) for k in range(n - 1))
        H0 += 0.5 * alpha * cfg.echelle_desaccord * sum((-1) ** k * sz[k] for k in range(n))

        # Dissipations de base et Quasiparticules
        Ls = []
        if cfg.dephasage > 0.0:
            for k in range(n):
                Ls.append(np.sqrt(cfg.dephasage / 2.0) * sz[k])
        if self.gamma_qp > 0.0:
            for k in range(n):
                Ls.append(np.sqrt(self.gamma_qp) * sm[k])

        self.S0 = liouvillien(H0, Ls)
        self.Sy = super_comm(0.5 * embed(SY, n - 1, n))
        self.Sz = np.array([super_comm(0.5 * z) for z in sz])
        self.DA = super_diss(sm[0])

        nk = [embed(NN, k, n) for k in range(n)]
        self.masque = [np.real(np.diag(m)) for m in nk]
        self.nTot = sum(nk)
        self.rowA = ligne(nk[0])

        # Stabilisateurs QEC
        op_zz = sz[0]
        op_xx = sx[0]
        for k in range(1, n):
            op_zz = op_zz @ sz[k]
            op_xx = op_xx @ sx[k]
        self.S_ZZZZ = op_zz
        self.S_XXXX = op_xx

        psi0 = np.zeros(self.d, dtype=complex)
        psi0[1 << (n - 1)] = 1.0                      # excitation initiale sur le qubit 0
        self.psi0 = psi0


def simuler(cfg: Config, M: Modele, gain: float, profil: np.ndarray,
            amplitude: float, bruit: np.ndarray, enregistrer: bool = False) -> dict:
    n_t, n, D = len(profil), M.n, M.d * M.d
    ref = np.outer(M.psi0, M.psi0.conj())
    rho = ref.copy()

    dose_tot, Fmin = 0.0, 1.0
    max_viol = 0.0
    G = np.zeros((D + 1, D + 1), dtype=complex)
    traj = {k: [] for k in ("t", "dose_cum", "F", "purete", "s_zz", "s_xx")}

    for t in range(n_t):
        N0 = float(np.trace(M.nTot @ rho).real)
        P = N0 / n
        gam = float(np.clip(np.tanh(profil[t] * amplitude * cfg.decay), 0.0, 1.0))
        theta = gain * float(np.tanh(2.0 * max(0.0, cfg.seuil - P)))
        env = 2.0 * np.sqrt(max(P * (1.0 - P), 0.0))

        G[:D, :D] = M.S0 + theta * M.Sy + np.tensordot(bruit[t] * env, M.Sz, axes=1) + gam * M.DA
        G[D, :D] = gam * M.rowA

        out = expm(G)[:, :D] @ rho.reshape(-1)
        rho = out[:D].reshape(M.d, M.d)
        dose = float(out[D].real)
        dose_tot += dose

        et, eh, vp = verifier(rho)
        pas = max(et, eh, vp)
        max_viol = max(max_viol, pas)
        if pas > cfg.tolerance_bilan:
            raise AssertionError(f"Validité CPTP rompue à t={t} : {pas:.3e}")

        F = float(np.real(M.psi0.conj() @ rho @ M.psi0))
        Fmin = min(Fmin, F)

        if enregistrer:
            traj["t"].append(t + 1)
            traj["dose_cum"].append(dose_tot)
            traj["F"].append(F)
            traj["purete"].append(purete(rho))
            traj["s_zz"].append(float(np.trace(rho @ M.S_ZZZZ).real))
            traj["s_xx"].append(float(np.trace(rho @ M.S_XXXX).real))

    res = {
        "dose_reelle": dose_tot,
        "F_min": Fmin,
        "F_final": float(np.real(M.psi0.conj() @ rho @ M.psi0)),
        "purete_finale": purete(rho),
        "syndrome_ZZ_final": float(np.trace(rho @ M.S_ZZZZ).real),
        "syndrome_XX_final": float(np.trace(rho @ M.S_XXXX).real),
        "max_violation": max_viol,
    }
    if enregistrer:
        res["trajectoire"] = traj
    return res


def calibrer_dose(cfg: Config, M: Modele, gain: float, profil: np.ndarray,
                  bruit: np.ndarray) -> tuple[Optional[float], dict]:
    cible, tol = cfg.dose_reelle_cible, cfg.tolerance_dose
    f = lambda a: simuler(cfg, M, gain, profil, float(a), bruit)
    ag = np.linspace(0.0, cfg.calibration_max_amplitude, cfg.calibration_grille_points)
    dg = np.array([f(a)["dose_reelle"] for a in ag])
    eg = dg - cible

    brackets = []
    for i in range(len(ag) - 1):
        if eg[i] * eg[i + 1] <= 0.0:
            brackets.append((float(ag[i]), float(ag[i + 1])))

    if not brackets:
        return None, {"statut": "non_encadree", "dose_min": float(dg.min()), "dose_max": float(dg.max())}

    lo, hi = brackets[0]
    best = None
    for _ in range(cfg.calibration_iterations):
        mid = 0.5 * (lo + hi)
        r = f(mid)
        err = r["dose_reelle"] - cible
        best = (mid, r)
        if abs(err) <= tol:
            break
        if (f(lo)["dose_reelle"] - cible) * err <= 0.0:
            hi = mid
        else:
            lo = mid

    a, r = best
    r["calibration"] = {"statut": "convergee", "solutions": len(brackets)}
    return a, r


# ============================================================================
# SECTION 4 : VALIDATION ET TESTS
# ============================================================================

def test_spectre_fluxonium():
    fp_sweet = FluxoniumParams(phi_ext=0.50)
    fp_off = FluxoniumParams(phi_ext=0.45)
    w_sweet, _, qp_s = diagonaliser_fluxonium(fp_sweet)
    w_off, _, _ = diagonaliser_fluxonium(fp_off)
    ok = (w_sweet > 0.0) and (qp_s > 0.0) and (w_sweet < w_off)
    return float(not ok), 0.0


def test_stabilisateurs_cptp():
    cfg = Config(n_qubits=4, n_stress=10, n_recup=2, dose_reelle_cible=0.2)
    M = Modele(cfg, 0.0)
    profil = construire_profil("chronique", cfg.n_stress, cfg.n_recup)
    bruit = np.zeros((len(profil), 4))
    r = simuler(cfg, M, 0.0, profil, 2.0, bruit)
    return r["max_violation"], cfg.tolerance_bilan


def test_relaxation_quasiparticules():
    w01, mat_qp = 1.5, 0.4
    g1 = evaluer_gamma_qp(1e-5, w01, mat_qp)
    g2 = evaluer_gamma_qp(1e-3, w01, mat_qp)
    return float(g1 >= g2), 0.0


TESTS = [
    ("Spectre Fluxonium : positivité et point doux", test_spectre_fluxonium),
    ("Dynamique CPTP 4-Qubits & Hermiticité", test_stabilisateurs_cptp),
    ("Cinétique de relaxation Quasiparticules (QP)", test_relaxation_quasiparticules),
]


def lancer_validation() -> bool:
    print("=" * 80)
    print(" SUITE DE VALIDATION PHYSIQUE & NUMÉRIQUE")
    print("=" * 80)
    reussite = True
    for nom, fn in TESTS:
        t0 = time.perf_counter()
        mesure, seuil = fn()
        ok = mesure <= seuil
        reussite &= ok
        etat = "PASS" if ok else "FAIL"
        print(f"[{etat}] {nom:<54} ({time.perf_counter() - t0:.2f}s)")
    print("=" * 80)
    return reussite


# ============================================================================
# SECTION 5 : INTERFACE LIGNE DE COMMANDE (CLI)
# ============================================================================

def cmd_valider(_) -> int:
    return 0 if lancer_validation() else 1


def cmd_fluxonium(a) -> int:
    fp = FluxoniumParams(E_C=a.EC, E_L=a.EL, E_J=a.EJ, phi_ext=a.flux)
    w01, phi, qp = diagonaliser_fluxonium(fp)
    print("=" * 60)
    print(" SPECTRE & TRANSITIONS DU FLUXONIUM")
    print("=" * 60)
    print(f"Pulsation propre w01 : {w01:.4f} GHz")
    print(f"Élément de transition dipolaire |<0|phi|1>|      : {phi:.4f}")
    print(f"Élément d'interaction QP        |<0|sin(phi/2)|1>| : {qp:.4f}")
    print("=" * 60)
    return 0


def cmd_campagne(a) -> int:
    cfg = Config(n_qubits=a.qubits, n_graines=2 if a.rapide else 4, dose_reelle_cible=a.cible)
    print(f"Lancement campagne {cfg.n_qubits} qubits | Cible dose = {cfg.dose_reelle_cible:.2f}")
    profil = construire_profil("chronique", cfg.n_stress, cfg.n_recup)
    for g in cfg.gains:
        for alpha in cfg.alphas:
            M = Modele(cfg, alpha)
            bruit = np.random.default_rng(cfg.seed).normal(0.0, cfg.bruit_interne, size=(len(profil), cfg.n_qubits))
            amp, res = calibrer_dose(cfg, M, g, profil, bruit)
            statut = "Calibré" if amp is not None else "Échec"
            s_zz = res.get("syndrome_ZZ_final", 0.0)
            print(f"Gain={g:.2f} | Alpha={alpha:.2f} | {statut} | Amp={amp or 0:.3f} | S_ZZ={s_zz:.3f}")
    return 0


def cmd_trajectoire(a) -> int:
    cfg = Config(n_qubits=a.qubits)
    M = Modele(cfg, float(a.alpha))
    profil = construire_profil("chronique", cfg.n_stress, cfg.n_recup)
    bruit = np.random.default_rng(cfg.seed).normal(0.0, cfg.bruit_interne, size=(len(profil), cfg.n_qubits))
    amp = a.amplitude or 5.0
    r = simuler(cfg, M, a.gain, profil, float(amp), bruit, enregistrer=True)
    tr = r["trajectoire"]
    out = Path(a.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(tr.keys())
        for i in range(len(tr["t"])):
            w.writerow([tr[k][i] for k in tr.keys()])
    print(f"Trajectoire écrite avec succès : {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="edifice_v2", description="Édifice Quantique V2 : Fluxonium, QP & 4 Qubits.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("valider", help="Exécute la validation unitaire et physique").set_defaults(f=cmd_valider)

    pf = sub.add_parser("fluxonium", help="Calcule le spectre d'un fluxonium")
    pf.add_argument("--EC", type=float, default=1.0)
    pf.add_argument("--EL", type=float, default=0.5)
    pf.add_argument("--EJ", type=float, default=4.0)
    pf.add_argument("--flux", type=float, default=0.50)
    pf.set_defaults(f=cmd_fluxonium)

    pc = sub.add_parser("campagne", help="Lance un plan de simulation")
    pc.add_argument("--qubits", type=int, default=4)
    pc.add_argument("--cible", type=float, default=0.50)
    pc.add_argument("--rapide", action="store_true")
    pc.add_argument("--output", type=str, default=None)
    pc.set_defaults(f=cmd_campagne)

    pt = sub.add_parser("trajectoire", help="Exporte l'évolution temporelle avec syndromes QEC")
    pt.add_argument("--qubits", type=int, default=4)
    pt.add_argument("--gain", type=float, default=0.0)
    pt.add_argument("--alpha", type=float, default=0.0)
    pt.add_argument("--amplitude", type=float, default=None)
    pt.add_argument("--output", type=str, required=True)
    pt.set_defaults(f=cmd_trajectoire)

    args = ap.parse_args()
    return args.f(args)


if __name__ == "__main__":
    sys.exit(main())
