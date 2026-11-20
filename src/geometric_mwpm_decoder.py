#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module : geometric_mwpm_decoder.py
Décodeur de chirurgie de réseau (21 qubits, d=3) pondéré par la distance
géodésique de Bures D_FS(pi, Phi_i) issue de la métrique de Fubini-Study.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class GeometricSurgeryConfig:
    n_qubits: int = 21
    p_base_Z: float = 0.0025
    p_base_X: float = 0.0005
    bures_coupling_gamma: float = 0.35
    n_shots: int = 25_000
    seed: int = 42


class GeometricLatticeSurgeryDecoder:
    """
    Décodeur MWPM (Minimum-Weight Perfect Matching) à poids géométriques
    pour la fusion de deux patchs d=3 (2 x 9 qubits) via 3 ancillaires d'interface.
    """

    def __init__(self, cfg: GeometricSurgeryConfig, bures_distances: np.ndarray):
        if len(bures_distances) != cfg.n_qubits:
            raise ValueError(f"Attendu {cfg.n_qubits} distances de Bures, reçu {len(bures_distances)}.")
        self.cfg = cfg
        self.d_bures = np.asarray(bures_distances, dtype=np.float64)
        self.p_Z_nodes, self.edge_weights = self._compute_geometric_weights()

    def _compute_geometric_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convertit la distance géodésique D_FS en log-vraisemblances d'arêtes."""
        p_geom = self.cfg.p_base_Z + self.cfg.bures_coupling_gamma * (self.d_bures ** 2)
        p_geom = np.clip(p_geom, 1e-6, 0.49)
        weights = -np.log(p_geom / (1.0 - p_geom))
        return p_geom, weights

    def _decode_1d_chain_mwpm(self, syndrome: np.ndarray, chain_weights: np.ndarray) -> np.ndarray:
        """
        Résout l'appariement parfait de poids minimal (MWPM) sur la chaîne
        d'interface en incluant les bords ouverts (frontières de patch).
        """
        n_edges = len(chain_weights)
        defects = np.where(syndrome == 1)[0]
        m = len(defects)
        if m == 0:
            return np.zeros(n_edges, dtype=np.int8)

        # Matrice de coût étendue (m défauts + m nœuds virtuels de frontière)
        size = 2 * m
        INF = 1e6
        cost_matrix = np.full((size, size), INF, dtype=np.float64)

        # Coûts entre paires de défauts réels
        for i in range(m):
            for j in range(i + 1, m):
                u, v = defects[i], defects[j]
                w_uv = float(np.sum(chain_weights[u:v]))
                cost_matrix[i, j] = cost_matrix[j, i] = w_uv

        # Coûts d'appariement d'un défaut réel vers la frontière la plus proche
        for i in range(m):
            u = defects[i]
            w_left = float(np.sum(chain_weights[:u]))
            w_right = float(np.sum(chain_weights[u:]))
            w_bound = min(w_left, w_right)
            cost_matrix[i, m + i] = cost_matrix[m + i, i] = w_bound

        # Coût nul entre nœuds virtuels de frontière
        for i in range(m):
            for j in range(i + 1, m):
                cost_matrix[m + i, m + j] = cost_matrix[m + j, m + i] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        correction = np.zeros(n_edges, dtype=np.int8)

        visited = set()
        for r, c in zip(row_ind, col_ind):
            if r >= m or r in visited:
                continue
            visited.add(r)
            if c < m:
                visited.add(c)
                u, v = sorted((defects[r], defects[c]))
                correction[u:v] ^= 1
            else:
                u = defects[r]
                w_left = float(np.sum(chain_weights[:u]))
                w_right = float(np.sum(chain_weights[u:]))
                if w_left <= w_right:
                    correction[:u] ^= 1
                else:
                    correction[u:] ^= 1

        return correction

    def benchmark_vs_uniform(self) -> Dict[str, float]:
        """
        Compare le taux d'échec logique entre le décodeur à poids uniformes
        et le décodeur pondéré par la géométrie de Fubini-Study.
        """
        rng = np.random.default_rng(self.cfg.seed)
        # Sous-chaîne critique de 5 qubits traversant l'interface (patch A -> ancillaires -> patch B)
        idx_seam = [7, 8, 9, 10, 11]
        p_seam = self.p_Z_nodes[idx_seam]
        w_geom = self.edge_weights[idx_seam]
        w_unif = np.ones_like(w_geom)

        errors = rng.random((self.cfg.n_shots, len(idx_seam))) < p_seam
        errors = errors.astype(np.int8)

        fail_uniform = 0
        fail_geometric = 0

        for s in range(self.cfg.n_shots):
            err = errors[s]
            syn = np.bitwise_xor(err[:-1], err[1:])

            corr_u = self._decode_1d_chain_mwpm(syn, w_unif)
            corr_g = self._decode_1d_chain_mwpm(syn, w_geom)

            if np.sum(np.bitwise_xor(err, corr_u)) % 2 == 1:
                fail_uniform += 1
            if np.sum(np.bitwise_xor(err, corr_g)) % 2 == 1:
                fail_geometric += 1

        p_L_unif = fail_uniform / self.cfg.n_shots
        p_L_geom = fail_geometric / self.cfg.n_shots
        gain = p_L_unif / max(p_L_geom, 1e-9)

        return {
            "p_L_uniform": p_L_unif,
            "p_L_geometric": p_L_geom,
            "geometric_gain_factor": gain,
            "max_node_error_Z": float(np.max(p_seam)),
            "min_node_error_Z": float(np.min(p_seam))
        }


if __name__ == "__main__":
    # Profil de distance géodésique D_FS(pi, Phi_i) avec asymétrie sur la couture d'interface
    x = np.linspace(-1.0, 1.0, 21)
    d_bures_demo = 0.14 * np.abs(np.sin(np.pi * x))
    d_bures_demo[11] = 0.24  # Décalage de flux localisé sur le nœud d'interface 11

    decoder = GeometricLatticeSurgeryDecoder(GeometricSurgeryConfig(), d_bures_demo)
    res = decoder.benchmark_vs_uniform()

    print("--- Chirurgie de Réseau 21q : Décodeur Géométrique Fubini-Study ---")
    print(f"  Erreur locale Z (min / max) : {res['min_node_error_Z']*100:.3f}% / {res['max_node_error_Z']*100:.3f}%")
    print(f"  Échec logique MWPM uniforme : {res['p_L_uniform']*100:.4f}%")
    print(f"  Échec logique MWPM Bures    : {res['p_L_geometric']*100:.4f}%")
    print(f"  Gain de suppression logique : {res['geometric_gain_factor']:.2f}x")
