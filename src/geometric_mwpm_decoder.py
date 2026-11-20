#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometric_mwpm_decoder.py  (version réécrite)

Prototype de décodage MWPM EXACT sur une chaîne 1D de qubits (code de
répétition, erreurs Z indépendantes), avec poids d'arêtes issus d'un profil
de distances géométriques d_i fourni en entrée.

PÉRIMÈTRE (à ne pas dépasser) :
  - Ce n'est PAS une chirurgie de réseau sur code de surface tourné d=3 :
    c'est une sous-chaîne 1D de qubits « de couture », bruit Z i.i.d.
  - La relation  p_i = p_base + gamma * d_i^2  est PHÉNOMÉNOLOGIQUE.
    Je ne connais aucune dérivation physique indépendante de cette loi
    ni de la valeur de gamma : « Je ne sais pas. »
  - Le profil d_i de la démonstration est SYNTHÉTIQUE, pas une mesure.
  - Pas d'erreurs de mesure, pas d'espace-temps, pas de stabilisateurs X.

Conventions :
  - qubit i  = arête i de la chaîne ; poids w_i = ln((1-p_i)/p_i)
  - check c  (c = 0..n-2) est placé entre les qubits c et c+1
  - frontières virtuelles à gauche (avant le qubit 0) et à droite

Réf. : Dennis, Kitaev, Landahl, Preskill, J. Math. Phys. 43, 4452 (2002)
       (décodage par appariement de poids minimal) ;
       Bengtsson & Życzkowski, Geometry of Quantum States, CUP (2006)
       (angle de Bures = distance de Fubini–Study pour états purs ;
       la « distance de Bures » D_B = sqrt(2(1-|<a|b>|)) en est distincte).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GeometricSurgeryConfig:
    n_qubits: int = 21
    p_base_Z: float = 0.0025
    coupling_gamma: float = 0.35            # phénoménologique, non calibré
    n_shots: int = 25_000
    seed: int = 42
    seam_indices: Tuple[int, ...] = (7, 8, 9, 10, 11)

    def __post_init__(self):
        if not 0.0 < self.p_base_Z < 0.5:
            raise ValueError("p_base_Z doit être dans ]0, 0.5[.")
        if self.coupling_gamma < 0:
            raise ValueError("coupling_gamma doit être >= 0.")
        if self.n_shots <= 0:
            raise ValueError("n_shots doit être > 0.")
        if len(self.seam_indices) < 2:
            raise ValueError("Au moins 2 qubits de couture sont requis.")
        if min(self.seam_indices) < 0 or max(self.seam_indices) >= self.n_qubits:
            raise ValueError("seam_indices hors de [0, n_qubits).")


def decode_chain_mwpm(syndrome: np.ndarray, qubit_weights: np.ndarray) -> np.ndarray:
    """
    MWPM exact (programmation dynamique sur sous-ensembles de défauts)
    pour une chaîne de n qubits, n-1 checks, deux frontières ouvertes.
    Retourne le vecteur de correction (n,) en {0,1}.
    """
    w = np.asarray(qubit_weights, dtype=np.float64)
    s = np.asarray(syndrome, dtype=np.int8)
    n = len(w)
    if len(s) != n - 1:
        raise ValueError(f"Syndrome de longueur {n-1} attendu, reçu {len(s)}.")
    if np.any(w < 0):
        raise ValueError("Les poids doivent être >= 0 (p_i <= 1/2).")

    defects = np.flatnonzero(s)
    m = len(defects)
    corr = np.zeros(n, dtype=np.int8)
    if m == 0:
        return corr
    if m > 20:
        raise ValueError("Trop de défauts pour la DP exacte (m > 20).")

    cum = np.concatenate(([0.0], np.cumsum(w)))   # cum[k] = somme w[:k]

    def c_pair(a: int, b: int) -> float:           # a < b : qubits a+1..b
        return float(cum[b + 1] - cum[a + 1])

    def c_left(c: int) -> float:                   # qubits 0..c
        return float(cum[c + 1])

    def c_right(c: int) -> float:                  # qubits c+1..n-1
        return float(cum[n] - cum[c + 1])

    full = (1 << m) - 1

    @lru_cache(maxsize=None)
    def solve(mask: int) -> Tuple[float, Tuple]:
        if mask == full:
            return 0.0, ()
        i = next(k for k in range(m) if not (mask >> k) & 1)
        ci = int(defects[i])
        # i -> frontière la moins coûteuse
        cl, cr = c_left(ci), c_right(ci)
        if cl <= cr:
            best = (cl, (("L", ci),))
        else:
            best = (cr, (("R", ci),))
        sub, mv = solve(mask | (1 << i))
        best = (best[0] + sub, best[1] + mv)
        # i -> autre défaut j
        for j in range(i + 1, m):
            if (mask >> j) & 1:
                continue
            cj = int(defects[j])
            sub, mv = solve(mask | (1 << i) | (1 << j))
            tot = c_pair(ci, cj) + sub
            if tot < best[0] - 1e-12:
                best = (tot, (("P", ci, cj),) + mv)
        return best

    _, moves = solve(0)
    for mv in moves:
        if mv[0] == "P":
            corr[mv[1] + 1: mv[2] + 1] ^= 1
        elif mv[0] == "L":
            corr[: mv[1] + 1] ^= 1
        else:
            corr[mv[1] + 1:] ^= 1
    return corr


def syndrome_of(err: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(err[:-1], err[1:])


class GeometricLatticeSurgeryDecoder:
    """Décodeur MWPM 1D à poids dérivés d'un profil géométrique d_i."""

    def __init__(self, cfg: GeometricSurgeryConfig, geometric_distances: Sequence[float]):
        d = np.asarray(geometric_distances, dtype=np.float64)
        if d.shape != (cfg.n_qubits,):
            raise ValueError(f"Attendu {cfg.n_qubits} distances, reçu {d.shape}.")
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("Distances : valeurs finies et >= 0 requises.")
        self.cfg = cfg
        self.d = d
        # Loi phénoménologique (voir en-tête)
        p = cfg.p_base_Z + cfg.coupling_gamma * d**2
        self.p_Z_nodes = np.clip(p, 1e-6, 0.49)
        self.edge_weights = np.log((1.0 - self.p_Z_nodes) / self.p_Z_nodes)

    # ------------------------------------------------------------------
    def _failure_table(self, p: np.ndarray, w: np.ndarray) -> Dict[bytes, bool]:
        """Décode chaque motif d'erreur possible (2^n) ; renvoie échec logique."""
        n = len(p)
        table: Dict[bytes, bool] = {}
        cache: Dict[bytes, np.ndarray] = {}
        for code in range(1 << n):
            err = np.array([(code >> k) & 1 for k in range(n)], dtype=np.int8)
            syn = syndrome_of(err)
            key = syn.tobytes()
            if key not in cache:
                cache[key] = decode_chain_mwpm(syn, w)
            corr = cache[key]
            res = np.bitwise_xor(err, corr)
            assert not syndrome_of(res).any(), "Correction incompatible avec le syndrome."
            table[err.tobytes()] = bool(res[0] == 1)   # résidu = tout 0 ou tout 1
        return table

    def _exact_and_mc(self, p: np.ndarray, w: np.ndarray,
                      errors: np.ndarray) -> Tuple[float, float]:
        n = len(p)
        table = self._failure_table(p, w)
        # exact : somme des probabilités des motifs en échec
        p_exact = 0.0
        for key, fail in table.items():
            if fail:
                e = np.frombuffer(key, dtype=np.int8)
                p_exact += float(np.prod(np.where(e == 1, p, 1.0 - p)))
        # Monte-Carlo : mêmes tirages pour les deux décodeurs
        fails = sum(table[errors[s].tobytes()] for s in range(errors.shape[0]))
        return p_exact, fails / errors.shape[0]

    def benchmark_vs_uniform(self) -> Dict[str, float]:
        cfg = self.cfg
        idx = list(cfg.seam_indices)
        p = self.p_Z_nodes[idx]
        w_geo = self.edge_weights[idx]
        w_uni = np.ones_like(w_geo)

        rng = np.random.default_rng(cfg.seed)
        errors = (rng.random((cfg.n_shots, len(idx))) < p).astype(np.int8)

        ex_u, mc_u = self._exact_and_mc(p, w_uni, errors)
        ex_g, mc_g = self._exact_and_mc(p, w_geo, errors)

        def se(x: float) -> float:                 # erreur-type binomiale
            return float(np.sqrt(max(x * (1 - x), 0.0) / cfg.n_shots))

        return {
            "p_L_uniform_exact": ex_u,
            "p_L_geometric_exact": ex_g,
            "gain_exact": ex_u / ex_g if ex_g > 0 else float("nan"),
            "p_L_uniform_mc": mc_u, "p_L_uniform_mc_se": se(mc_u),
            "p_L_geometric_mc": mc_g, "p_L_geometric_mc_se": se(mc_g),
            "max_node_error_Z": float(p.max()),
            "min_node_error_Z": float(p.min()),
        }


if __name__ == "__main__":
    # Profil SYNTHÉTIQUE (illustration, non mesuré)
    x = np.linspace(-1.0, 1.0, 21)
    d_demo = 0.14 * np.abs(np.sin(np.pi * x))
    d_demo[11] = 0.24

    dec = GeometricLatticeSurgeryDecoder(GeometricSurgeryConfig(), d_demo)
    r = dec.benchmark_vs_uniform()

    print("--- Chaîne 1D de couture : MWPM exact, poids géométriques (synthétique) ---")
    print(f"  Erreur Z locale (min / max) : {r['min_node_error_Z']*100:.3f}% / {r['max_node_error_Z']*100:.3f}%")
    print(f"  Échec logique uniforme  : exact {r['p_L_uniform_exact']*100:.4f}%  | MC {r['p_L_uniform_mc']*100:.4f}% ± {r['p_L_uniform_mc_se']*100:.4f}")
    print(f"  Échec logique géométr.  : exact {r['p_L_geometric_exact']*100:.4f}%  | MC {r['p_L_geometric_mc']*100:.4f}% ± {r['p_L_geometric_mc_se']*100:.4f}")
    print(f"  Rapport exact uniforme/géométrique : {r['gain_exact']:.3f}")
    print("  (valable pour ce modèle jouet et ce profil synthétique uniquement)")
