#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PONT LOURD UNIFIÉ — Architecture Complète (100% NumPy, un seul fichier)
=======================================================================

Couplage multi-échelle :
  1. Champ spatial périodique (Moiré) sur réseau replié 21 qubits (zigzag)
  2. Noyau dynamique conservatif régulant x_qp (bilan de masse fermé)
  3. Solveur microscopique Fluxonium (spectre, T1, T_phi,
     métrique de Fubini-Study symétrique, holonomie de Berry)
  4. Simulateur Monte-Carlo vectorisé de chirurgie de réseau
     (code de surface tourné d=3, merge/split, mesure logique)

Note d'honnêteté scientifique
-----------------------------
- Les étapes 1-4 sont des modèles physiques ou numériques explicites.
- L'indicateur `readiness_score` est un COMPOSITE HEURISTIQUE de
  classement (T1, T_phi, g_FS), pas une observable cQED standard.
  Il est clairement étiqueté comme tel et ne doit pas être présenté
  comme une grandeur physique.

Dépendance unique : numpy
"""

from __future__ import annotations

import os

# Verrouillage des threads BLAS pour exécution fluide (mobile et desktop)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ============================================================================
# 0. UTILITAIRES
# ============================================================================

def clamp01(x: float) -> float:
    """Contraint une valeur dans [0, 1]."""
    return float(np.clip(np.real(x), 0.0, 1.0))


def sigmoid(x: float) -> float:
    """Sigmoïde numérique stable."""
    x = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-x))


def safe_divide(num: float, den: float) -> float:
    """Division sûre."""
    if abs(den) < 1e-14:
        return 0.0
    return num / den


# ============================================================================
# 1. GÉOMÉTRIE SPATIALE & CHAMP DE MOIRÉ (21 QUBITS REPLIÉS EN ZIGZAG)
# ============================================================================

@dataclass(frozen=True)
class MoireField:
    """Champ d'interférence à deux ondes planes avec modulation temporelle."""

    k1: Tuple[float, float] = (2.1, 0.8)
    k2: Tuple[float, float] = (-0.9, 2.3)
    amplitude: float = 0.03
    phi1: float = 0.0
    phi2: float = 0.0
    temporal_frequency: float = 0.05
    temporal_phase: float = 0.0

    def sample(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: float = 1.0,
    ) -> np.ndarray:
        """Échantillonne le champ de moiré."""
        phase1 = self.k1[0] * x + self.k1[1] * y + self.phi1
        phase2 = self.k2[0] * x + self.k2[1] * y + self.phi2
        spatial = np.cos(phase1) + np.cos(phase2)
        if self.temporal_frequency == 0.0:
            temporal = 1.0
        else:
            temporal = math.cos(
                2.0 * math.pi * self.temporal_frequency * t
                + self.temporal_phase
            )
        return self.amplitude * spatial * temporal


def coordonnees_reseau_21q(
    pas_x: float = 0.45,
    pas_y: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordonnées physiques (mm) des 21 qubits sur la topologie en zigzag (3 x 7).
    """
    xs, ys = [], []
    for col in range(7):
        for row in range(3):
            xs.append(col * pas_x)
            ys.append(row * pas_y + (0.15 if col % 2 == 1 else 0.0))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def gradient_flux_substrat(
    xs: np.ndarray,
    ys: np.ndarray,
    field: MoireField,
    t: float = 1.0,
    sigma_device: float = 1.8,
) -> np.ndarray:
    """
    Calcule le décalage de flux local delta_Phi_i(t), pondéré
    par une enveloppe gaussienne centrée sur la puce.
    """
    x0, y0 = float(np.mean(xs)), float(np.mean(ys))
    r2 = ((xs - x0) ** 2 + (ys - y0) ** 2) / (sigma_device ** 2)
    enveloppe = np.exp(-0.5 * r2)
    return field.sample(xs, ys, t) * enveloppe


# ============================================================================
# 2. NOYAU DYNAMIQUE CONSERVATIF (BILAN FERMÉ)
# ============================================================================

@dataclass(frozen=True)
class ConfigNoyau:
    """Paramètres du noyau conservatif."""

    n_steps: int = 40
    alpha: float = 0.12
    gain: float = 0.65
    seuil: float = 0.5
    decay: float = 0.10
    bruit_interne: float = 0.005
    tolerance_bilan: float = 1e-10
    seed: int = 42


@dataclass(frozen=True)
class BilanConservatif:
    """Résultat du noyau conservatif."""

    charge_quasiparticules: float
    stress_controle: float
    max_residu_bilan: float


def evoluer_noyau_conservatif(
    cfg: ConfigNoyau,
    mode: str = "aigu",
) -> BilanConservatif:
    """
    Évolution couplée à deux états avec fermeture stricte du bilan de masse :

        dm = -dr + dc + cr + db + de   (résidu <= tolerance_bilan)
    """
    rng = np.random.default_rng(cfg.seed)
    W = np.array(
        [
            [0.8 - cfg.alpha, 0.2 - cfg.alpha],
            [0.2 + cfg.alpha, 0.8 + cfg.alpha],
        ],
        dtype=float,
    )
    etat = np.array([1.0, 1.0], dtype=float)

    profil = np.zeros(cfg.n_steps)
    if mode == "chronique":
        profil[:] = 1.0 / cfg.n_steps
    elif mode == "aigu":
        profil[np.linspace(0, cfg.n_steps - 1, 8, dtype=int)] = 0.12
    else:
        raise ValueError(f"Mode inconnu : {mode}")

    bruit = rng.normal(
        0.0, cfg.bruit_interne, size=(cfg.n_steps, 2)
    )
    max_residu = 0.0

    for t in range(cfg.n_steps):
        avant = etat.copy()
        masse = float(avant.sum())

        demande = math.tanh(profil[t] * 1.8 * cfg.decay)
        apres = np.clip(
            avant - np.array([demande, 0.0]), 0.0, 1.0
        )
        dr = float((avant - apres).sum())

        couple = W @ apres
        dc = float(couple.sum() - apres.sum())

        correction = cfg.gain * np.tanh(
            2.0 * np.maximum(0.0, cfg.seuil - couple)
        )
        cr = float(correction.sum())

        be = bruit[t] * avant * (1.0 - avant)
        db = float(be.sum())

        avant_clip = couple + correction + be
        nouvel = np.clip(avant_clip, 0.0, 1.0)
        de = float(nouvel.sum() - avant_clip.sum())
        etat = nouvel

        dm = float(etat.sum() - masse)
        residu = abs(dm - (-dr + dc + cr + db + de))
        max_residu = max(max_residu, residu)
        if residu > cfg.tolerance_bilan:
            raise AssertionError(
                f"Bilan non fermé à t={t} : {residu}"
            )

    x_qp_eff = 1.0e-8 * (
        1.0 + 4.0 * (1.0 - float(etat[0]))
    )
    return BilanConservatif(
        charge_quasiparticules=x_qp_eff,
        stress_controle=float(1.0 - np.prod(etat)),
        max_residu_bilan=max_residu,
    )


# ============================================================================
# 3. PHYSIQUE FLUXONIUM, GÉOMÉTRIE QUANTIQUE & HOLONOMIE
# ============================================================================

@dataclass(frozen=True)
class FluxoniumParams:
    """Paramètres physiques du fluxonium."""

    EC: float = 1.0          # GHz
    EL: float = 0.5          # GHz
    EJ_base: float = 5.0     # GHz
    N_basis: int = 35
    Q_diel: float = 6.0e5
    Delta: float = 50.0      # GHz
    T: float = 0.020         # K
    A_phi: float = 2.0e-6    # Phi0
    kB_h: float = 20.8366    # GHz/K


class SolveurFluxoniumMicroscopique:
    """
    Résolution spectrale, temps de cohérence et géométrie
    de Fubini-Study (100% NumPy).

    Métrique de Fubini-Study (différence symétrique) :
        g_FS(phi) = (1 - |<psi(phi-eps)|psi(phi+eps)>|^2) / (2 eps)^2
    """

    def __init__(self, params: FluxoniumParams):
        self.p = params
        phi_zpf = (2.0 * params.EC / params.EL) ** 0.25
        a = np.zeros(
            (params.N_basis, params.N_basis), dtype=complex
        )
        for k in range(1, params.N_basis):
            a[k - 1, k] = math.sqrt(k)
        adag = a.conj().T

        self.phi_op = phi_zpf * (a + adag)
        self.n_op = -1j / (2.0 * phi_zpf) * (a - adag)
        self.phi_vals, self.phi_vecs = np.linalg.eigh(self.phi_op)
        self.H_statique = (
            4.0 * params.EC * (self.n_op @ self.n_op)
            + 0.5 * params.EL * (self.phi_op @ self.phi_op)
        )

    def _diag(
        self,
        EJ: float,
        phi_ext: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalise le hamiltonien pour un (EJ, phi_ext) donné."""
        cos_op = (
            self.phi_vecs
            * np.cos(self.phi_vals - phi_ext)
        ) @ self.phi_vecs.conj().T
        H = self.H_statique - EJ * cos_op
        evals, evecs = np.linalg.eigh(
            0.5 * (H + H.conj().T)
        )
        return evals[:3] - evals[0], evecs[:, :3]

    def phase_berry_boucle(
        self,
        EJ: float,
        phi_centre: float,
        rayon: float = 0.25 * math.pi,
    ) -> float:
        """
        Holonomie discrète de Pancharatnam-Berry sur une
        excursion fermée en flux.
        """
        chemin = np.linspace(
            phi_centre - rayon, phi_centre + rayon, 21
        )
        etats = [
            self._diag(EJ, float(p))[1][:, 0] for p in chemin
        ]
        phase = 0.0
        for u, v in zip(etats[:-1], etats[1:]):
            ov = np.vdot(u, v)
            if abs(ov) < 1e-14:
                return 0.0
            phase += np.angle(ov)
        ov_close = np.vdot(etats[-1], etats[0])
        if abs(ov_close) >= 1e-14:
            phase += np.angle(ov_close)
        return float(np.angle(np.exp(1j * phase)))

    def caracteriser_noeud(
        self,
        EJ_local: float,
        phi_ext: float,
        x_qp: float,
        d_phi: float = 1e-3,
    ) -> Dict[str, float]:
        """
        Caractérise un nœud du réseau : spectre, cohérence, métrique FS.

        g_FS est calculé par différence SYMÉTRIQUE (plus précis
        que la version one-sided).
        """
        E0, V0 = self._diag(EJ_local, phi_ext)
        Ep, Vp = self._diag(EJ_local, phi_ext + d_phi)
        Em, Vm = self._diag(EJ_local, phi_ext - d_phi)

        f01 = float(E0[1])
        f12 = float(E0[2] - E0[1])
        alpha_rel = (
            (f12 - f01) / f01 if abs(f01) > 1e-12 else 0.0
        )
        d_omega = float((Ep[1] - Em[1]) / (2.0 * d_phi))

        n01 = abs(
            np.vdot(V0[:, 0], self.n_op @ V0[:, 1])
        )
        phi01 = abs(
            np.vdot(V0[:, 0], self.phi_op @ V0[:, 1])
        )
        sin_op = (
            self.phi_vecs * np.sin(0.5 * self.phi_vals)
        ) @ self.phi_vecs.conj().T
        sin01 = abs(
            np.vdot(V0[:, 0], sin_op @ V0[:, 1])
        )

        x = f01 / (self.p.kB_h * self.p.T)
        coth = (
            1.0 / np.tanh(0.5 * x)
            if x > 1e-8
            else 2.0 / x
        )
        g_diel = (
            16.0
            * self.p.EC
            * (n01 ** 2)
            / self.p.Q_diel
            * coth
        )

        f_qp = (
            math.sqrt(
                max(0.0, 2.0 * self.p.Delta / max(f01, 1e-9))
            )
            * x_qp
            * (1.0 + math.exp(-min(700.0, x)))
        )
        g_qp = (
            (8.0 * EJ_local / math.pi) * (sin01 ** 2)
            + (8.0 * self.p.EL / math.pi)
            * ((0.5 * phi01) ** 2)
        ) * f_qp
        T1_us = (1.0 / (g_diel + g_qp)) * 1e-3

        g_phi = (
            abs(d_omega)
            * math.sqrt(2.0 * math.log(2.0))
            * (2.0 * math.pi * self.p.A_phi)
        )
        Tphi_us = (
            (1.0 / g_phi) * 1e-3 if g_phi > 1e-12 else 1e6
        )
        T2_star_us = 1.0 / (
            0.5 / T1_us + 1.0 / Tphi_us
        )

        # Métrique de Fubini-Study : différence SYMÉTRIQUE
        overlap_sym = abs(
            np.vdot(Vm[:, 0], Vp[:, 0])
        )
        F_loc = clamp01(overlap_sym ** 2)
        g_FS = max(
            0.0, (1.0 - F_loc) / ((2.0 * d_phi) ** 2)
        )

        return {
            "f01_GHz": f01,
            "alpha_rel": alpha_rel,
            "T1_us": T1_us,
            "Tphi_us": Tphi_us,
            "T2_star_us": T2_star_us,
            "F_loc": F_loc,
            "g_FS": g_FS,
        }

    @staticmethod
    def readiness_score(
        obs_moy: Dict[str, float],
    ) -> float:
        """
        Indicateur HEURISTIQUE de classement des configurations.

        Composite de T1, T_phi et g_FS. Ce n'est PAS une observable
        cQED standard ; il sert uniquement à ordonner les scénarios.
        """
        coh_spec = clamp01(
            0.5 * sigmoid(math.log10(obs_moy["T1_us"] + 1e-12) - 1.0)
            + 0.5 * sigmoid(
                math.log10(obs_moy["T2_star_us"] + 1e-12) - 1.0
            )
        )
        coh_geom = clamp01(
            math.exp(-0.001 * min(obs_moy["g_FS"], 2000.0))
        )
        return clamp01(0.6 * coh_spec + 0.4 * coh_geom)


# ============================================================================
# 4. CHIRURGIE DE RÉSEAU VECTORISÉE (21 QUBITS, CODE DE SURFACE TOURNÉ d=3)
# ============================================================================

class SimChirurgie21Q:
    """
    Simulateur Monte-Carlo vectorisé d'un cycle complet de
    chirurgie de réseau : Init -> Merge -> Split -> Measure Z_L1.

    Limitation documentée : le decodeur par table de syndromes
    corrige seulement les erreurs à un qubit (poids 1). C'est
    suffisant pour une démonstration, pas pour un seuil réaliste.
    """

    def __init__(
        self,
        pX: np.ndarray,
        pZ: np.ndarray,
        pro: float,
        seed: int = 123,
    ):
        self.pX = np.asarray(pX, dtype=float)
        self.pZ = np.asarray(pZ, dtype=float)
        self.pro = float(pro)
        self.rng = np.random.default_rng(seed)

        self.HZ = self._mat(
            [[0, 1], [3, 4, 6, 7], [1, 2, 4, 5], [7, 8]]
        )
        self.HX = self._mat(
            [[3, 6], [0, 1, 3, 4], [4, 5, 7, 8], [2, 5]]
        )
        self.dZ, self.vZ = self._decodeur_syndromes(self.HZ)
        self.dX, self.vX = self._decodeur_syndromes(self.HX)

    @staticmethod
    def _mat(checks: List[List[int]]) -> np.ndarray:
        """Construit la matrice de contrôle."""
        H = np.zeros((len(checks), 9), dtype=np.int8)
        for r, c in enumerate(checks):
            H[r, c] = 1
        return H

    @staticmethod
    def _decodeur_syndromes(
        H: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Table de syndromes pour erreurs à 1 qubit."""
        table = np.zeros((16, 9), dtype=np.int8)
        valide = np.zeros(16, dtype=bool)
        poids = np.array([1, 2, 4, 8], dtype=np.int8)
        valide[0] = True
        for q in range(9):
            e = np.zeros(9, dtype=np.int8)
            e[q] = 1
            idx = int(((H @ e) % 2) @ poids)
            if not valide[idx]:
                table[idx] = e
                valide[idx] = True
        return table, valide

    def _injecter_bruit(
        self,
        eX: np.ndarray,
        eZ: np.ndarray,
        idx: np.ndarray,
    ) -> None:
        """Injecte du bruit X et Z sur les qubits indexés."""
        eX[:, idx] ^= (
            self.rng.random((eX.shape[0], len(idx)))
            < self.pX[idx]
        ).astype(np.int8)
        eZ[:, idx] ^= (
            self.rng.random((eZ.shape[0], len(idx)))
            < self.pZ[idx]
        ).astype(np.int8)

    def _corriger_patch(
        self,
        eX: np.ndarray,
        eZ: np.ndarray,
        offset: int,
    ) -> None:
        """Corrige un patch de 9 qubits via la table de syndromes."""
        poids = np.array([1, 2, 4, 8], dtype=np.int8)
        xl = eX[:, offset:offset + 9]
        zl = eZ[:, offset:offset + 9]

        iz = (
            (
                (xl @ self.HZ.T) % 2
            )
            ^ (
                self.rng.random((xl.shape[0], 4)) < self.pro
            ).astype(np.int8)
        ) @ poids
        ix = (
            (
                (zl @ self.HX.T) % 2
            )
            ^ (
                self.rng.random((zl.shape[0], 4)) < self.pro
            ).astype(np.int8)
        ) @ poids

        mz, mx = self.vZ[iz], self.vX[ix]
        if np.any(mz):
            xl[mz] ^= self.dZ[iz][mz]
        if np.any(mx):
            zl[mx] ^= self.dX[ix][mx]

    def executer_lote(
        self,
        shots: int = 5000,
        rounds_merge: int = 3,
    ) -> float:
        """Exécute une lot de shots et retourne le taux de succès logique."""
        p1 = np.arange(0, 9)
        interface = np.arange(9, 12)
        p2 = np.arange(12, 21)

        eX = np.zeros((shots, 21), dtype=np.int8)
        eZ = np.zeros((shots, 21), dtype=np.int8)

        # (a) Initialisation et stabilisation spatiale
        self._injecter_bruit(
            eX, eZ, np.concatenate((p1, p2))
        )
        self._corriger_patch(eX, eZ, 0)
        self._corriger_patch(eX, eZ, 12)

        # (c) Merge : d cycles de mesure de parité avec vote majoritaire
        lectures = np.zeros(
            (shots, rounds_merge), dtype=np.int8
        )
        for r in range(rounds_merge):
            eX[:, interface] = 0
            eZ[:, interface] = 0
            self._injecter_bruit(eX, eZ, interface)
            lectures[:, r] = (
                (
                    np.sum(eZ[:, interface], axis=1) % 2
                )
                ^ (self.rng.random(shots) < self.pro)
            ).astype(np.int8)
        mesure_merge = (
            np.sum(lectures, axis=1) > rounds_merge // 2
        ).astype(np.int8)

        # (d) Split et stabilisation spatiale
        self._injecter_bruit(
            eX, eZ, np.concatenate((p1, p2))
        )
        self._corriger_patch(eX, eZ, 0)
        self._corriger_patch(eX, eZ, 12)

        # (e) Mesure logique Z_L1 et contrôle homologique
        mesure_Z1 = (
            np.sum(eX[:, [0, 3, 6]], axis=1) % 2
        ).astype(np.int8)
        ex_ok = (
            (
                np.sum(eX[:, [12, 15, 18]], axis=1) % 2
            )
            ^ mesure_Z1
        ) == 0
        ez_ok = (
            (
                np.sum(eZ[:, [12, 13, 14]], axis=1) % 2
            )
            ^ mesure_merge
        ) == 0
        return float(np.mean(ex_ok & ez_ok))


# ============================================================================
# 5. ORCHESTRATION GLOBALE & EXPORTS
# ============================================================================

def executer_pipeline_complet(
    shots: int = 5000,
    basis: int = 35,
    rounds_merge: int = 3,
    output_dir: str | None = None,
) -> List[Dict]:
    """Exécute le pipeline complet et retourne les résultats."""
    xs, ys = coordonnees_reseau_21q()
    champ_moire = MoireField()
    delta_flux = gradient_flux_substrat(
        xs, ys, champ_moire, t=1.0
    )

    bilan_dyn = evoluer_noyau_conservatif(
        ConfigNoyau(), mode="aigu"
    )
    solveur = SolveurFluxoniumMicroscopique(
        FluxoniumParams(N_basis=basis)
    )

    print("=" * 94)
    print(
        "  PONT LOURD UNIFIÉ : MOIRÉ (21Q) -> NOYAU -> FLUXONIUM -> LATTICE SURGERY (d=3)"
    )
    print("=" * 94)
    print(
        f"x_qp = {bilan_dyn.charge_quasiparticules:.2e} | "
        f"Résidu bilan = {bilan_dyn.max_residu_bilan:.1e} | "
        f"Basis = {basis} | Shots = {shots}"
    )
    print("-" * 94)
    print(
        f"{'EJ (GHz)':<9} | {'tau (ns)':<8} | {'T1 (µs)':<8} | "
        f"{'Tphi (µs)':<9} | {'g_FS':<8} | {'Readiness':<9} | {'Succès Logique'}"
    )
    print("-" * 94)

    configurations = [
        (3.0, 400.0, 0.0040),
        (5.0, 300.0, 0.0020),
        (5.0, 150.0, 0.0010),
        (7.5, 250.0, 0.0015),
        (9.0, 200.0, 0.0010),
    ]

    resultats: List[Dict] = []
    for EJ_nom, tau_ns, pro in configurations:
        tau_us = tau_ns * 1e-3
        obs_noeuds: List[Dict[str, float]] = []
        pX = np.zeros(21)
        pZ = np.zeros(21)

        for i in range(21):
            EJ_i = EJ_nom * (
                1.0 + 0.015 * math.cos(i * 0.7)
            )
            phi_i = math.pi + float(delta_flux[i])
            obs = solveur.caracteriser_noeud(
                EJ_i, phi_i,
                bilan_dyn.charge_quasiparticules,
            )
            obs_noeuds.append(obs)

            px = 0.5 * (
                1.0 - math.exp(-tau_us / obs["T1_us"])
            )
            pz = (
                0.5 * (1.0 - math.exp(-tau_us / obs["Tphi_us"]))
                + 0.5 * px
            )
            pX[i] = float(np.clip(px, 0.0, 1.0))
            pZ[i] = float(np.clip(pz, 0.0, 1.0))

        obs_moy = {
            k: float(
                np.mean([o[k] for o in obs_noeuds])
            )
            for k in obs_noeuds[0]
        }
        berry = solveur.phase_berry_boucle(EJ_nom, math.pi)
        readiness = solveur.readiness_score(obs_moy)
        succes_logique = SimChirurgie21Q(
            pX, pZ, pro
        ).executer_lote(
            shots=shots, rounds_merge=rounds_merge
        )

        ligne = {
            "EJ_GHz": EJ_nom,
            "tau_cycle_ns": tau_ns,
            "p_readout": pro,
            "T1_moy_us": round(obs_moy["T1_us"], 2),
            "Tphi_moy_us": round(obs_moy["Tphi_us"], 2),
            "g_FS_moy": round(obs_moy["g_FS"], 3),
            "berry_phase_rad": round(berry, 4),
            "readiness_score": round(readiness, 4),
            "probabilite_succes_logique": round(
                succes_logique, 5
            ),
        }
        resultats.append(ligne)

        print(
            f"{EJ_nom:7.2f}   | {tau_ns:6.1f}   | "
            f"{obs_moy['T1_us']:6.1f}   | "
            f"{obs_moy['Tphi_us']:7.1f}   | "
            f"{obs_moy['g_FS']:6.3f}   | "
            f"{readiness:7.3f}   | "
            f"{succes_logique * 100:8.3f} %"
        )

    print("=" * 94)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bilan_pont_lourd.json").write_text(
            json.dumps(
                {
                    "bilan_conservatif": asdict(bilan_dyn),
                    "resultats": resultats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with (out / "bilan_pont_lourd.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(resultats[0].keys())
            )
            writer.writeheader()
            writer.writerows(resultats)
        LOGGER.info(
            "Résultats exportés dans %s", out
        )

    return resultats


# ============================================================================
# 6. TESTS INTÉGRÉS
# ============================================================================

def test_bilan_ferme():
    """Vérifie que le bilan de masse est fermé."""
    bilan = evoluer_noyau_conservatif(ConfigNoyau())
    assert bilan.max_residu_bilan < 1e-10, (
        f"Bilan non fermé : {bilan.max_residu_bilan}"
    )


def test_g_FS_positive():
    """Vérifie que g_FS est positive."""
    solveur = SolveurFluxoniumMicroscopique(
        FluxoniumParams()
    )
    obs = solveur.caracteriser_noeud(5.0, math.pi, 1e-8)
    assert obs["g_FS"] >= 0.0, "g_FS ne doit pas être négative."


def test_sweet_spot_g_FS_minimal():
    """Vérifie que g_FS est minimale au sweet spot."""
    solveur = SolveurFluxoniumMicroscopique(
        FluxoniumParams()
    )
    g_sweet = solveur.caracteriser_noeud(
        5.0, math.pi, 1e-8
    )["g_FS"]
    g_off = solveur.caracteriser_noeud(
        5.0, math.pi * 0.95, 1e-8
    )["g_FS"]
    assert g_sweet <= g_off, (
        "g_FS devrait être minimale au sweet spot."
    )


def test_surgery_success_in_range():
    """Vérifie que le taux de succès logique est dans [0, 1]."""
    pX = np.full(21, 0.001)
    pZ = np.full(21, 0.001)
    sim = SimChirurgie21Q(pX, pZ, 0.001)
    succes = sim.executer_lote(shots=1000)
    assert 0.0 <= succes <= 1.0, (
        f"Taux de succès hors plage : {succes}"
    )


def run_all_tests():
    """Lance tous les tests intégrés."""
    test_bilan_ferme()
    test_g_FS_positive()
    test_sweet_spot_g_FS_minimal()
    test_surgery_success_in_range()
    LOGGER.info("Tous les tests sont passés.")


# ============================================================================
# 7. CLI
# ============================================================================

def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Pont Lourd Unifié : Moiré -> Noyau -> Fluxonium -> "
            "Lattice Surgery."
        )
    )
    parser.add_argument(
        "--shots", type=int, default=5000,
        help="Nombre de trajectoires Monte-Carlo."
    )
    parser.add_argument(
        "--basis", type=int, default=35,
        help="Dimension de la base de Fock du fluxonium."
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Nombre de cycles de mesure lors du Merge."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Dossier d'export optionnel (JSON + CSV)."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Lance les tests intégrés."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.test:
        run_all_tests()
        return

    executer_pipeline_complet(
        shots=args.shots,
        basis=args.basis,
        rounds_merge=args.rounds,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
