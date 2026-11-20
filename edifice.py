#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ÉDIFICE QUANTIQUE 1.1.0
=======================

Fichier unique exécutable dans Termux / Linux / macOS / Windows.

Origine conceptuelle :
    src/edifice

Usage :
    python edifice.py valider
    python edifice.py config --output ~/noyau/config.json
    python edifice.py campagne --rapide --output ~/noyau
    python edifice.py campagne --reprendre --output ~/noyau
    python edifice.py trajectoire --gain 0.8 --alpha 0.1 \
        --output ~/traj.csv

Sections :
    1. core
    2. modele
    3. validation
    4. campagne
    5. cli

Dépendance externe unique :
    numpy

NOTE PHYSIQUE
-------------
Ce programme décrit une chaîne de qubits ouverts avec :
    - Hamiltonien XY ;
    - désaccord alterné ;
    - rétroaction sur le dernier qubit ;
    - bruit multiplicatif ;
    - déphasage optionnel ;
    - dissipation de type sigma- sur le qubit 0 ;
    - propagation par exponentielle matricielle du générateur de Lindblad.

Il ne constitue pas, dans cette version, un Hamiltonien de fluxonium
avec degrés de liberté de phase, énergie Josephson EJ, énergie
inductive EL ou énergie de charge EC.

Convention de registre :
    qubit 0 = bit de poids fort.

Convention de vectorisation :
    vec(A rho B) = (A kron B^T) vec(rho)
avec vectorisation par lignes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


VERSION = __version__ = "1.1.0"

log = logging.getLogger("edifice")


# ============================================================================
# SECTION 1 : CORE
# ============================================================================

I2 = np.eye(2, dtype=complex)

SX = np.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=complex,
)

SY = np.array(
    [
        [0, -1j],
        [1j, 0],
    ],
    dtype=complex,
)

SM = np.array(
    [
        [0, 1],
        [0, 0],
    ],
    dtype=complex,
)

SP = SM.T.copy()

NN = np.diag([0.0, 1.0]).astype(complex)

# Convention :
# |g> -> -1
# |e> -> +1
SZ = 2.0 * NN - I2


def eye(d: int) -> np.ndarray:
    """Matrice identité complexe de dimension d."""
    return np.eye(d, dtype=complex)


def embed(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """
    Place un opérateur mono-qubit sur 'site' dans un registre de n qubits.

    Convention :
        site = 0 -> bit de poids fort.

    Exemple pour n=2 :
        embed(op, 0, 2) = op ⊗ I
        embed(op, 1, 2) = I ⊗ op
    """
    if op.shape != (2, 2):
        raise ValueError("embed() exige un opérateur 2x2.")
    if not 0 <= site < n:
        raise ValueError(f"site={site} hors limites pour n={n}.")

    m = np.array([[1.0 + 0j]])

    for k in range(n):
        m = np.kron(m, op if k == site else I2)

    return m


# ---------------------------------------------------------------------------
# Superopérateurs
# ---------------------------------------------------------------------------

def super_comm(H: np.ndarray) -> np.ndarray:
    """
    Superopérateur du commutateur :

        rho -> -i [H, rho]

    Avec vectorisation par lignes :
        vec(A rho B) = (A ⊗ B^T) vec(rho)
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H doit être une matrice carrée.")

    I = eye(H.shape[0])

    return -1j * (
        np.kron(H, I)
        - np.kron(I, H.T)
    )


def super_diss(L: np.ndarray) -> np.ndarray:
    """
    Dissipateur de Lindblad :

        D[L](rho)
        = L rho L†
          - 1/2 {L†L, rho}
    """
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L doit être une matrice carrée.")

    I = eye(L.shape[0])
    LdL = L.conj().T @ L

    return (
        np.kron(L, L.conj())
        - 0.5 * np.kron(LdL, I)
        - 0.5 * np.kron(I, LdL.T)
    )


def liouvillien(H: np.ndarray, Ls=()) -> np.ndarray:
    """Construit le générateur de Lindblad complet."""
    S = super_comm(H)

    for L in Ls:
        S = S + super_diss(L)

    return S


def ligne(op: np.ndarray) -> np.ndarray:
    """
    Forme linéaire associée à :

        Tr(op rho)

    sous vectorisation par lignes.
    """
    return op.T.reshape(-1)


def augmente(
    S: np.ndarray,
    row: np.ndarray,
    poids: float = 1.0,
) -> np.ndarray:
    """
    Générateur augmenté [rho ; compteur].

    Le dernier état scalaire vérifie :

        d(compteur)/dt = poids * Tr(op rho)
    """
    D = S.shape[0]

    if row.shape != (D,):
        raise ValueError(
            f"Dimension incompatible : row={row.shape}, attendu {(D,)}."
        )

    G = np.zeros((D + 1, D + 1), dtype=complex)

    G[:D, :D] = S
    G[D, :D] = poids * row

    return G


# ---------------------------------------------------------------------------
# Exponentielle matricielle
# ---------------------------------------------------------------------------

def expm(
    A: np.ndarray,
    tol: float = 1e-16,
    max_terms: int = 80,
) -> np.ndarray:
    """
    Exponentielle matricielle par série de Taylor avec scaling/squaring.

    Le calcul est :

        exp(A)
        = [exp(A / 2^s)]^(2^s)

    avec une série de Taylor sur A / 2^s.

    Cette implémentation est volontairement indépendante de scipy afin
    de conserver numpy comme dépendance externe unique.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expm() exige une matrice carrée.")

    if max_terms < 2:
        raise ValueError("max_terms doit être >= 2.")

    nrm = float(np.linalg.norm(A, 1))

    if nrm == 0.0:
        return np.eye(A.shape[0], dtype=complex)

    s = (
        0
        if nrm <= 0.5
        else int(np.ceil(np.log2(nrm / 0.5)))
    )

    B = A / (2.0 ** s)

    E = np.eye(A.shape[0], dtype=complex)
    T = E.copy()

    for k in range(1, max_terms):
        T = T @ B / k
        E = E + T

        if float(np.linalg.norm(T, 1)) < tol:
            break
    else:
        raise RuntimeError(
            "expm : série non convergée "
            f"(norme={nrm:.3e}, max_terms={max_terms})."
        )

    for _ in range(s):
        E = E @ E

    return E


def propager(
    S: np.ndarray,
    rho: np.ndarray,
    t: float,
    row=None,
    poids: float = 1.0,
):
    """
    Évolution sous générateur constant S pendant une durée t.

    Sans compteur :
        retourne (rho_final, 0.0)

    Avec compteur :
        retourne (rho_final, compteur)
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho doit être carrée.")

    if t < 0:
        raise ValueError("t doit être >= 0.")

    d = rho.shape[0]

    if S.shape != (d * d, d * d):
        raise ValueError(
            f"S incompatible avec rho : "
            f"S={S.shape}, rho={rho.shape}."
        )

    rho_vec = rho.reshape(-1)

    if row is None:
        out = expm(S * t) @ rho_vec

        return (
            out.reshape(d, d),
            0.0,
        )

    G = augmente(S, row, poids)

    out = expm(G * t) @ np.concatenate(
        [rho_vec, np.zeros(1, dtype=complex)]
    )

    return (
        out[:-1].reshape(d, d),
        float(out[-1].real),
    )


# ---------------------------------------------------------------------------
# Contrôles physiques / numériques
# ---------------------------------------------------------------------------

def verifier(rho: np.ndarray) -> tuple[float, float, float]:
    """
    Retourne :

        (erreur de trace,
         erreur d'hermiticité,
         violation de positivité)

    La positivité est mesurée par la partie négative de la plus petite
    valeur propre de la partie hermitienne.
    """
    herm = (rho + rho.conj().T) / 2.0

    trace_error = abs(
        float(np.trace(rho).real) - 1.0
    )

    herm_error = float(
        np.linalg.norm(rho - rho.conj().T)
    )

    min_eval = float(
        np.linalg.eigvalsh(herm)[0]
    )

    positivity_violation = max(
        0.0,
        -min_eval,
    )

    return (
        trace_error,
        herm_error,
        positivity_violation,
    )


# ---------------------------------------------------------------------------
# Mesures
# ---------------------------------------------------------------------------

def _herm(a: np.ndarray) -> np.ndarray:
    """Projection hermitienne."""
    return (a + a.conj().T) / 2.0


def sqrt_psd(A: np.ndarray) -> np.ndarray:
    """Racine carrée hermitienne positive d'une matrice PSD."""
    w, V = np.linalg.eigh(_herm(A))

    return (
        V * np.sqrt(np.clip(w, 0.0, None))
    ) @ V.conj().T


def purete(rho: np.ndarray) -> float:
    """Tr(rho²)."""
    return float(
        np.trace(rho @ rho).real
    )


def entropie_bits(rho: np.ndarray) -> float:
    """Entropie de von Neumann en bits."""
    w = np.clip(
        np.linalg.eigvalsh(_herm(rho)),
        0.0,
        None,
    )

    w = w[w > 1e-15]

    if len(w) == 0:
        return 0.0

    return float(
        -(w * np.log2(w)).sum()
    )


def distance_trace(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Distance de trace :

        D(a,b) = 1/2 ||a-b||_1
    """
    return float(
        0.5
        * np.abs(
            np.linalg.eigvalsh(
                _herm(a - b)
            )
        ).sum()
    )


def fidelite(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Fidélité d'Uhlmann :

        F(a,b)
        = [Tr sqrt(sqrt(a) b sqrt(a))]^2
    """
    sa = sqrt_psd(a)

    w = np.clip(
        np.linalg.eigvalsh(
            _herm(sa @ b @ sa)
        ),
        0.0,
        None,
    )

    return float(
        np.sqrt(w).sum() ** 2
    )


_YY = np.kron(SY, SY)


def concurrence(rho: np.ndarray) -> float:
    """
    Concurrence de Wootters pour deux qubits.

    Référence :
        W. K. Wootters,
        Phys. Rev. Lett. 80, 2245 (1998).

    Cette mesure n'est définie ici que pour une matrice 4x4.
    """
    if rho.shape != (4, 4):
        raise ValueError(
            "concurrence : deux qubits requis."
        )

    R = rho @ (
        _YY @ rho.conj() @ _YY
    )

    ev = np.sqrt(
        np.abs(
            np.linalg.eigvals(R).real
        )
    )

    ev = np.sort(ev)[::-1]

    return float(
        max(
            0.0,
            ev[0] - ev[1] - ev[2] - ev[3],
        )
    )


def neg_log_bits(rho: np.ndarray) -> float:
    """
    Négativité logarithmique :

        log2 ||rho^(T0)||_1

    Bipartition :
        qubit 0 | reste
    """
    d = rho.shape[0]

    if d % 2 != 0:
        raise ValueError(
            "Dimension impaire incompatible avec une bipartition 2 x d."
        )

    pt = (
        rho.reshape(
            2,
            d // 2,
            2,
            d // 2,
        )
        .transpose(2, 1, 0, 3)
        .reshape(d, d)
    )

    trace_norm = float(
        np.abs(
            np.linalg.eigvalsh(
                _herm(pt)
            )
        ).sum()
    )

    return float(
        np.log2(max(1.0, trace_norm))
    )


# ============================================================================
# SECTION 2 : MODELE
# ============================================================================

@dataclass(frozen=True)
class Config:
    """
    Configuration complète du modèle.
    """

    n_qubits: int = 2

    n_graines: int = 5

    n_stress: int = 100

    n_recup: int = 20

    seed: int = 42

    modes: tuple[str, ...] = (
        "chronique",
        "aigu",
    )

    dose_reelle_cible: float = 0.50

    tolerance_dose: float = 1e-6

    alphas: tuple[float, ...] = (
        0.0,
        0.20,
    )

    gains: tuple[float, ...] = (
        0.0,
        0.80,
    )

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
            raise ValueError(
                "n_qubits doit être dans [2, 4]."
            )

        if self.n_graines < 1:
            raise ValueError(
                "n_graines doit être >= 1."
            )

        if self.n_stress < 10:
            raise ValueError(
                "n_stress doit être >= 10."
            )

        if self.n_recup < 0:
            raise ValueError(
                "n_recup doit être >= 0."
            )

        if not 0.0 < self.dose_reelle_cible < 1.0:
            raise ValueError(
                "La dose cible doit être dans ]0, 1[."
            )

        if self.tolerance_dose <= 0:
            raise ValueError(
                "tolerance_dose doit être > 0."
            )

        if self.tolerance_bilan <= 0:
            raise ValueError(
                "tolerance_bilan doit être > 0."
            )

        if self.calibration_grille_points < 2:
            raise ValueError(
                "calibration_grille_points doit être >= 2."
            )

        if self.calibration_max_amplitude <= 0:
            raise ValueError(
                "calibration_max_amplitude doit être > 0."
            )

        if self.calibration_iterations < 1:
            raise ValueError(
                "calibration_iterations doit être >= 1."
            )

        if any(
            not -0.2 <= float(a) <= 0.2
            for a in self.alphas
        ):
            raise ValueError(
                "alpha doit appartenir à [-0.2, 0.2]."
            )

        if any(
            float(g) < 0.0 or float(g) > 5.0
            for g in self.gains
        ):
            raise ValueError(
                "gain doit appartenir à [0, 5]."
            )

        if (
            self.bruit_interne < 0
            or self.dephasage < 0
            or self.couplage_J < 0
        ):
            raise ValueError(
                "bruit, dephasage et couplage doivent être >= 0."
            )

        if (
            not set(self.modes)
            <= {"chronique", "aigu"}
            or not self.modes
        ):
            raise ValueError(
                "modes : sous-ensemble non vide de "
                "{chronique, aigu}."
            )

    def to_dict(self) -> dict:
        """Convertit la configuration en dictionnaire JSON-compatible."""
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        """Construit Config en rejetant les clés inconnues."""
        d = dict(d)

        inconnus = (
            set(d)
            - set(Config.__dataclass_fields__)
        )

        if inconnus:
            raise ValueError(
                f"Paramètres inconnus : {sorted(inconnus)}"
            )

        for k in (
            "alphas",
            "gains",
            "modes",
        ):
            if k in d:
                d[k] = tuple(d[k])

        return Config(**d)

    def to_json(self, chemin) -> None:
        """Écrit la configuration sur disque."""
        Path(chemin).write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def from_json(chemin) -> "Config":
        """Charge une configuration JSON."""
        return Config.from_dict(
            json.loads(
                Path(chemin).read_text(
                    encoding="utf-8"
                )
            )
        )


def construire_profil(
    mode: str,
    n_stress: int,
    n_recup: int,
) -> np.ndarray:
    """
    Construit le profil temporel d'exposition.

    chronique :
        amplitude répartie uniformément sur la phase stress.

    aigu :
        dix impulsions de 0.1 pendant la phase stress.
    """
    p = np.zeros(
        n_stress + n_recup,
        dtype=float,
    )

    if mode == "chronique":

        p[:n_stress] = (
            1.0 / n_stress
        )

    elif mode == "aigu":

        indices = np.linspace(
            0,
            n_stress - 1,
            10,
            dtype=int,
        )

        p[indices] = 0.1

    else:
        raise ValueError(
            f"Mode inconnu : {mode}"
        )

    return p


class Modele:
    """
    Pièces constantes du générateur pour un alpha donné.
    """

    def __init__(
        self,
        cfg: Config,
        alpha: float,
    ) -> None:

        n = cfg.n_qubits

        self.n = n
        self.d = 2 ** n

        sm = [
            embed(SM, k, n)
            for k in range(n)
        ]

        sp = [
            embed(SP, k, n)
            for k in range(n)
        ]

        sz = [
            embed(SZ, k, n)
            for k in range(n)
        ]

        H0 = sum(
            (
                cfg.couplage_J
                * (
                    sp[k] @ sm[k + 1]
                    + sm[k] @ sp[k + 1]
                )
            )
            for k in range(n - 1)
        )

        H0 = H0 + (
            0.5
            * alpha
            * cfg.echelle_desaccord
            * sum(
                (-1) ** k * sz[k]
                for k in range(n)
            )
        )

        S0 = super_comm(H0)

        if cfg.dephasage > 0.0:

            for k in range(n):

                S0 = S0 + super_diss(
                    np.sqrt(
                        cfg.dephasage / 2.0
                    ) * sz[k]
                )

        self.S0 = S0

        self.Sy = super_comm(
            0.5 * embed(
                SY,
                n - 1,
                n,
            )
        )

        self.Sz = np.array(
            [
                super_comm(
                    0.5 * z
                )
                for z in sz
            ]
        )

        # Dissipation sur le qubit 0.
        self.DA = super_diss(sm[0])

        nk = [
            embed(NN, k, n)
            for k in range(n)
        ]

        self.masque = [
            np.real(np.diag(m))
            for m in nk
        ]

        self.nTot = sum(nk)

        self.rowA = ligne(nk[0])

        # Vérification trace :
        # Tr(S rho) = 0 pour tout rho.
        tr = eye(self.d).reshape(-1)

        for S in (
            self.S0,
            self.Sy,
            self.DA,
            *self.Sz,
        ):

            if (
                np.linalg.norm(
                    tr @ S
                )
                > 1e-12
            ):
                raise ValueError(
                    "Superopérateur non préservant la trace."
                )

        # État initial :
        # excitation sur le qubit 0.
        #
        # Comme q0 est le bit de poids fort :
        # indice = 2^(n-1)
        psi0 = np.zeros(
            self.d,
            dtype=complex,
        )

        psi0[
            1 << (n - 1)
        ] = 1.0

        self.psi0 = psi0


def simuler(
    cfg: Config,
    M: Modele,
    gain: float,
    profil: np.ndarray,
    amplitude: float,
    bruit: np.ndarray,
    enregistrer: bool = False,
) -> dict:
    """
    Simule une trajectoire complète.

    Le générateur est constant pendant chaque pas dt=1.
    """

    n_t = len(profil)
    n = M.n
    D = M.d * M.d

    if bruit.shape != (
        n_t,
        n,
    ):
        raise ValueError(
            "Forme du tableau de bruit invalide."
        )

    if amplitude < 0:
        raise ValueError(
            "amplitude doit être >= 0."
        )

    ref = np.outer(
        M.psi0,
        M.psi0.conj(),
    )

    rho = ref.copy()

    dose_tot = 0.0

    Fmin = 1.0

    Cmax = 0.0

    ENmax = 0.0

    dstress = None

    max_viol = 0.0

    max_resN = 0.0

    G = np.zeros(
        (
            D + 1,
            D + 1,
        ),
        dtype=complex,
    )

    traj = {
        k: []
        for k in (
            "t",
            "phase",
            "gamma",
            "theta",
            "dose_cum",
            "F",
            "purete",
            "EN",
            "C",
        )
    }

    for k in range(n):
        traj[f"P_q{k}"] = []

    for t in range(n_t):

        N0 = float(
            np.trace(
                M.nTot @ rho
            ).real
        )

        P = N0 / n

        gam = float(
            np.clip(
                np.tanh(
                    profil[t]
                    * amplitude
                    * cfg.decay
                ),
                0.0,
                1.0,
            )
        )

        theta = (
            gain
            * float(
                np.tanh(
                    2.0
                    * max(
                        0.0,
                        cfg.seuil - P,
                    )
                )
            )
        )

        env = (
            2.0
            * np.sqrt(
                max(
                    P * (1.0 - P),
                    0.0,
                )
            )
        )

        # Générateur du pas.
        G[:D, :D] = (
            M.S0
            + theta * M.Sy
            + np.tensordot(
                bruit[t] * env,
                M.Sz,
                axes=1,
            )
            + gam * M.DA
        )

        # Le compteur mesure :
        # dDose/dt = gamma * <N_0>
        G[D, :D] = (
            gam * M.rowA
        )

        out = (
            expm(G)
            @ rho.reshape(-1).tolist()
            if False
            else expm(G) @ np.concatenate(
                [
                    rho.reshape(-1),
                    np.zeros(
                        1,
                        dtype=complex,
                    ),
                ]
            )
        )

        rho = (
            out[:D]
            .reshape(
                M.d,
                M.d,
            )
        )

        dose = float(
            out[D].real
        )

        dose_tot += dose

        et, eh, vp = verifier(rho)

        pas = max(
            et,
            eh,
            vp,
        )

        max_viol = max(
            max_viol,
            pas,
        )

        if (
            pas
            > cfg.tolerance_bilan
        ):
            raise AssertionError(
                f"Validité rompue à t={t} : "
                f"{pas:.3e}"
            )

        # Sans rétroaction :
        # le Hamiltonien conserve le nombre total d'excitations ;
        # la seule perte est le compteur dissipatif.
        if gain == 0.0:

            res = abs(
                (
                    float(
                        np.trace(
                            M.nTot @ rho
                        ).real
                    )
                    - N0
                )
                + dose
            )

            max_resN = max(
                max_resN,
                res,
            )

            if (
                res
                > cfg.tolerance_bilan
            ):
                raise AssertionError(
                    "Excitations non conservées "
                    f"à t={t} : {res:.3e}"
                )

        F = float(
            np.real(
                M.psi0.conj()
                @ rho
                @ M.psi0
            )
        )

        Fmin = min(
            Fmin,
            F,
        )

        EN = neg_log_bits(rho)

        ENmax = max(
            ENmax,
            EN,
        )

        Cc = (
            concurrence(rho)
            if n == 2
            else None
        )

        if Cc is not None:
            Cmax = max(
                Cmax,
                Cc,
            )

        if (
            t
            == cfg.n_stress - 1
        ):
            dstress = (
                distance_trace(
                    rho,
                    ref,
                )
            )

        if enregistrer:

            diag = np.real(
                np.diag(rho)
            )

            traj["t"].append(
                t + 1
            )

            traj["phase"].append(
                (
                    "stress"
                    if t < cfg.n_stress
                    else "recup"
                )
            )

            traj["gamma"].append(
                gam
            )

            traj["theta"].append(
                theta
            )

            traj["dose_cum"].append(
                dose_tot
            )

            traj["F"].append(
                F
            )

            traj["purete"].append(
                purete(rho)
            )

            traj["EN"].append(
                EN
            )

            traj["C"].append(
                Cc
            )

            for k in range(n):

                traj[f"P_q{k}"].append(
                    float(
                        diag
                        @ M.masque[k]
                    )
                )

    df = distance_trace(
        rho,
        ref,
    )

    if dstress is None:
        dstress = df

    result = {
        "dose_reelle": dose_tot,

        "F_min": Fmin,

        "F_final": float(
            np.real(
                M.psi0.conj()
                @ rho
                @ M.psi0
            )
        ),

        "C_max": (
            Cmax
            if n == 2
            else None
        ),

        "C_final": (
            concurrence(rho)
            if n == 2
            else None
        ),

        "EN_max": ENmax,

        "EN_final": neg_log_bits(rho),

        "purete_finale": purete(rho),

        "entropie_finale": entropie_bits(rho),

        "distance_stress": dstress,

        "distance_finale": df,

        "amelioration_recuperation":
            dstress - df,

        "max_violation": max_viol,

        "max_residu_N": (
            max_resN
            if gain == 0.0
            else None
        ),
    }

    if enregistrer:
        result["trajectoire"] = traj

    return result


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrer_dose(
    cfg: Config,
    M: Modele,
    gain: float,
    profil: np.ndarray,
    bruit: np.ndarray,
) -> tuple[Optional[float], dict]:
    """
    Recherche une amplitude telle que :

        dose_reelle = dose_reelle_cible

    Méthode :
        1. balayage initial ;
        2. détection des encadrements ;
        3. bissection sur chaque encadrement ;
        4. choix de la plus petite solution détectée.

    La fonction ne suppose PAS que dose(amplitude) est monotone.

    Elle signale explicitement :
        non_monotone=True

    si la grille détecte une décroissance de dose supérieure
    à la tolérance.
    """

    cible = cfg.dose_reelle_cible
    tol = cfg.tolerance_dose

    def f(a: float) -> dict:
        return simuler(
            cfg,
            M,
            gain,
            profil,
            float(a),
            bruit,
        )

    ag = np.linspace(
        0.0,
        cfg.calibration_max_amplitude,
        cfg.calibration_grille_points,
    )

    dg = np.array(
        [
            f(a)["dose_reelle"]
            for a in ag
        ],
        dtype=float,
    )

    eg = dg - cible

    nonmono = bool(
        np.any(
            np.diff(dg)
            < -tol
        )
    )

    # ------------------------------------------------------------------
    # Recherche des brackets
    # ------------------------------------------------------------------
    #
    # On accepte :
    #
    #   eg[i] == 0
    #
    # ainsi que :
    #
    #   eg[i] * eg[i+1] < 0
    #
    # Cela évite de perdre une solution située exactement
    # sur un point de la grille.
    #
    brackets = []

    for i in range(len(ag)):

        if abs(eg[i]) <= tol:
            brackets.append(
                (
                    float(ag[i]),
                    float(ag[i]),
                    0,
                )
            )

        if i < len(ag) - 1:

            if (
                eg[i]
                * eg[i + 1]
                < 0.0
            ):
                brackets.append(
                    (
                        float(ag[i]),
                        float(ag[i + 1]),
                        1,
                    )
                )

    # ------------------------------------------------------------------
    # Aucune racine encadrée
    # ------------------------------------------------------------------
    if not brackets:

        idx_min = int(
            np.argmin(
                np.abs(eg)
            )
        )

        return None, {
            "statut": "non_encadree",
            "non_monotone": nonmono,
            "dose_min": float(
                dg.min()
            ),
            "dose_max": float(
                dg.max()
            ),
            "meilleure_amp": float(
                ag[idx_min]
            ),
            "erreur_cible": float(
                eg[idx_min]
            ),
        }

    # ------------------------------------------------------------------
    # Résolution des brackets
    # ------------------------------------------------------------------

    solutions = []

    for lo, hi, kind in brackets:

        # Solution déjà présente sur la grille.
        if kind == 0:

            r = f(lo)

            if (
                abs(
                    r["dose_reelle"]
                    - cible
                )
                <= tol
            ):
                solutions.append(
                    (
                        lo,
                        r,
                    )
                )

            continue

        flo = (
            f(lo)["dose_reelle"]
            - cible
        )

        best = None

        for _ in range(
            cfg.calibration_iterations
        ):

            mid = (
                0.5
                * (lo + hi)
            )

            r = f(mid)

            fm = (
                r["dose_reelle"]
                - cible
            )

            best = (
                mid,
                r,
            )

            if abs(fm) <= tol:
                break

            if flo * fm <= 0.0:

                hi = mid

            else:

                lo = mid
                flo = fm

        if (
            best is not None
            and abs(
                best[1]["dose_reelle"]
                - cible
            )
            <= tol
        ):
            solutions.append(
                best
            )

    # ------------------------------------------------------------------
    # Aucune convergence
    # ------------------------------------------------------------------

    if not solutions:

        return None, {
            "statut": "non_convergee",
            "non_monotone": nonmono,
            "dose_min": float(
                dg.min()
            ),
            "dose_max": float(
                dg.max()
            ),
        }

    # Plus petite amplitude parmi les solutions trouvées.
    solutions.sort(
        key=lambda x: x[0]
    )

    amplitude, resultat = (
        solutions[0]
    )

    resultat["calibration"] = {
        "statut": "convergee",
        "non_monotone": nonmono,
        "solutions": len(
            solutions
        ),
    }

    return (
        amplitude,
        resultat,
    )


# ============================================================================
# SECTION 3 : VALIDATION
# ============================================================================

def _rng():
    return np.random.default_rng(
        12345
    )


def _herm_alea(
    rng,
    d,
):
    A = (
        rng.normal(
            size=(d, d)
        )
        + 1j
        * rng.normal(
            size=(d, d)
        )
    )

    return (
        A + A.conj().T
    ) / 2.0


def _rho_alea(
    rng,
    d,
):
    A = (
        rng.normal(
            size=(d, d)
        )
        + 1j
        * rng.normal(
            size=(d, d)
        )
    )

    r = A @ A.conj().T

    return r / np.trace(
        r
    ).real


def _rk4(
    H,
    Ls,
    rho,
    T,
    n,
):
    """
    RK4 direct de l'équation maîtresse.

    Sert d'intégrateur indépendant de contrôle
    pour l'exponentielle du Liouvillien.
    """
    h = T / n

    def f(r):

        out = (
            -1j
            * (
                H @ r
                - r @ H
            )
        )

        for L in Ls:

            LdL = (
                L.conj().T
                @ L
            )

            out = (
                out
                + L
                @ r
                @ L.conj().T
                - 0.5
                * (
                    LdL @ r
                    + r @ LdL
                )
            )

        return out

    for _ in range(n):

        k1 = f(rho)

        k2 = f(
            rho
            + 0.5 * h * k1
        )

        k3 = f(
            rho
            + 0.5 * h * k2
        )

        k4 = f(
            rho
            + h * k3
        )

        rho = (
            rho
            + h / 6.0
            * (
                k1
                + 2 * k2
                + 2 * k3
                + k4
            )
        )

    return rho


def _ket(
    i,
    d,
):
    v = np.zeros(
        d,
        dtype=complex,
    )

    v[i] = 1.0

    return np.outer(
        v,
        v.conj(),
    )


# ---------------------------------------------------------------------------
# Tests analytiques
# ---------------------------------------------------------------------------

def t_expm_unitaire():

    H = _herm_alea(
        _rng(),
        4,
    )

    w, V = np.linalg.eigh(H)

    ref = (
        V
        * np.exp(
            -1j
            * w
            * 3.0
        )
    ) @ V.conj().T

    erreur = float(
        np.abs(
            expm(
                -1j * H * 3.0
            )
            - ref
        ).max()
    )

    return erreur, 1e-12


def t_rabi():

    Om, T = 0.7, 5.3

    r, _ = propager(
        liouvillien(
            0.5
            * Om
            * SX
        ),
        _ket(
            0,
            2,
        ),
        T,
    )

    erreur = abs(
        float(
            r[1, 1].real
        )
        - np.sin(
            Om * T / 2.0
        ) ** 2
    )

    return erreur, 1e-12


def t_amortissement_et_dose():

    g, T = 0.4, 3.7

    S = liouvillien(
        np.zeros(
            (2, 2),
            dtype=complex,
        ),
        [
            np.sqrt(g)
            * SM
        ],
    )

    r, dose = propager(
        S,
        _ket(1, 2),
        T,
        row=ligne(NN),
        poids=g,
    )

    e1 = abs(
        float(
            r[1, 1].real
        )
        - np.exp(
            -g * T
        )
    )

    e2 = abs(
        dose
        - (
            1.0
            - np.exp(
                -g * T
            )
        )
    )

    return max(e1, e2), 1e-12


def t_echange_deux_qubits():

    J, T = 0.3, 2.1

    sp = np.kron(
        SP,
        I2,
    )

    sm = np.kron(
        SM,
        I2,
    )

    spb = np.kron(
        I2,
        SP,
    )

    smb = np.kron(
        I2,
        SM,
    )

    H = J * (
        sp @ smb
        + sm @ spb
    )

    r, _ = propager(
        liouvillien(H),
        _ket(2, 4),
        T,
    )

    erreur = abs(
        float(
            r[2, 2].real
        )
        - np.cos(
            J * T
        ) ** 2
    )

    return erreur, 1e-12


def t_semigroupe():

    rng = _rng()

    S = liouvillien(
        _herm_alea(
            rng,
            4,
        ),
        [
            0.4
            * (
                rng.normal(
                    size=(4, 4)
                )
                + 1j
                * rng.normal(
                    size=(4, 4)
                )
            )
            for _ in range(2)
        ],
    )

    E1 = expm(
        S * 0.7
    )

    E2 = expm(
        S * 1.4
    )

    erreur = float(
        np.abs(
            E1 @ E1
            - E2
        ).max()
    )

    return erreur, 1e-12


def t_rk4_croise():

    rng = _rng()

    H = (
        _herm_alea(
            rng,
            4,
        )
        * 0.5
    )

    Ls = [
        0.3
        * (
            rng.normal(
                size=(4, 4)
            )
            + 1j
            * rng.normal(
                size=(4, 4)
            )
        )
        for _ in range(2)
    ]

    rho0 = _rho_alea(
        rng,
        4,
    )

    a, _ = propager(
        liouvillien(
            H,
            Ls,
        ),
        rho0,
        1.0,
    )

    b = _rk4(
        H,
        Ls,
        rho0,
        1.0,
        4000,
    )

    erreur = float(
        np.abs(
            a - b
        ).max()
    )

    return erreur, 1e-9


def t_positivite_aleatoire():

    rng = _rng()

    S = liouvillien(
        _herm_alea(
            rng,
            4,
        ),
        [
            0.5
            * (
                rng.normal(
                    size=(4, 4)
                )
                + 1j
                * rng.normal(
                    size=(4, 4)
                )
            )
            for _ in range(3)
        ],
    )

    E = expm(
        S * 0.3
    )

    rho = _rho_alea(
        rng,
        4,
    )

    worst = 0.0

    for _ in range(50):

        rho = (
            E
            @ rho.reshape(-1)
        ).reshape(
            4,
            4,
        )

        worst = max(
            worst,
            *verifier(rho),
        )

    return worst, 1e-12


def t_concurrence():

    psi = np.zeros(
        4,
        dtype=complex,
    )

    psi[1] = (
        1 / np.sqrt(2)
    )

    psi[2] = (
        -1 / np.sqrt(2)
    )

    sing = np.outer(
        psi,
        psi.conj(),
    )

    p = 0.7

    werner = (
        p * sing
        + (
            1 - p
        )
        * np.eye(4)
        / 4
    )

    e1 = abs(
        concurrence(
            sing
        )
        - 1.0
    )

    e2 = abs(
        concurrence(
            _ket(
                0,
                4,
            )
        )
    )

    e3 = abs(
        concurrence(
            werner
        )
        - max(
            0.0,
            (3 * p - 1)
            / 2,
        )
    )

    return (
        max(
            e1,
            e2,
            e3,
        ),
        1e-6,
    )


def t_neg_log():

    psi = np.zeros(
        4,
        dtype=complex,
    )

    psi[1] = (
        1 / np.sqrt(2)
    )

    psi[2] = (
        -1 / np.sqrt(2)
    )

    sing = np.outer(
        psi,
        psi.conj(),
    )

    p = 0.7

    werner = (
        p * sing
        + (
            1 - p
        )
        * np.eye(4)
        / 4
    )

    tn = (
        3 * (1 + p)
        + abs(
            1 - 3 * p
        )
    ) / 4

    e1 = abs(
        neg_log_bits(
            sing
        )
        - 1.0
    )

    e2 = abs(
        neg_log_bits(
            _ket(
                0,
                4,
            )
        )
    )

    e3 = abs(
        neg_log_bits(
            werner
        )
        - np.log2(tn)
    )

    return (
        max(
            e1,
            e2,
            e3,
        ),
        1e-12,
    )


def t_entropie_fidelite():

    psi = np.zeros(
        4,
        dtype=complex,
    )

    psi[1] = (
        1 / np.sqrt(2)
    )

    psi[2] = (
        -1 / np.sqrt(2)
    )

    sing = np.outer(
        psi,
        psi.conj(),
    )

    p = 0.7

    werner = (
        p * sing
        + (
            1 - p
        )
        * np.eye(4)
        / 4
    )

    e1 = abs(
        entropie_bits(
            np.eye(4) / 4
        )
        - 2.0
    )

    e2 = abs(
        entropie_bits(
            sing
        )
    )

    e3 = abs(
        fidelite(
            sing,
            sing,
        )
        - 1.0
    )

    e4 = abs(
        fidelite(
            _ket(
                0,
                4,
            ),
            _ket(
                3,
                4,
            ),
        )
    )

    e5 = abs(
        fidelite(
            sing,
            werner,
        )
        - (
            p
            + (
                1 - p
            ) / 4
        )
    )

    return (
        max(
            e1,
            e2,
            e3,
            e4,
            e5,
        ),
        1e-9,
    )


# ---------------------------------------------------------------------------
# Tests modèle
# ---------------------------------------------------------------------------

def _cfg_petit(**kw):

    base = dict(
        n_stress=30,
        n_recup=10,
        calibration_grille_points=20,
        n_graines=1,
    )

    base.update(kw)

    return Config(**base)


def t_reproductibilite():

    cfg = _cfg_petit()

    M = Modele(
        cfg,
        0.2,
    )

    profil = construire_profil(
        "chronique",
        cfg.n_stress,
        cfg.n_recup,
    )

    bruit = np.random.default_rng(
        7
    ).normal(
        0,
        cfg.bruit_interne,
        size=(
            len(profil),
            cfg.n_qubits,
        ),
    )

    a = simuler(
        cfg,
        M,
        0.8,
        profil,
        9.0,
        bruit,
    )

    b = simuler(
        cfg,
        M,
        0.8,
        profil,
        9.0,
        bruit,
    )

    return (
        float(a != b),
        0.0,
    )


def t_conservation_simulation():

    worst = 0.0

    for nq in (
        2,
        3,
    ):

        cfg = _cfg_petit(
            n_qubits=nq,
            dephasage=0.02,
        )

        M = Modele(
            cfg,
            0.1,
        )

        for mode in (
            "chronique",
            "aigu",
        ):

            profil = construire_profil(
                mode,
                cfg.n_stress,
                cfg.n_recup,
            )

            bruit = np.random.default_rng(
                3
            ).normal(
                0,
                cfg.bruit_interne,
                size=(
                    len(profil),
                    nq,
                ),
            )

            r = simuler(
                cfg,
                M,
                0.0,
                profil,
                6.0,
                bruit,
            )

            worst = max(
                worst,
                r["max_violation"],
                r["max_residu_N"],
            )

    return worst, 1e-10


def t_calibration():

    cfg = _cfg_petit()

    M = Modele(
        cfg,
        0.0,
    )

    profil = construire_profil(
        "aigu",
        cfg.n_stress,
        cfg.n_recup,
    )

    bruit = np.random.default_rng(
        5
    ).normal(
        0,
        cfg.bruit_interne,
        size=(
            len(profil),
            cfg.n_qubits,
        ),
    )

    a, r = calibrer_dose(
        cfg,
        M,
        0.0,
        profil,
        bruit,
    )

    if a is None:
        return 1.0, 0.0

    return (
        abs(
            r["dose_reelle"]
            - cfg.dose_reelle_cible
        ),
        cfg.tolerance_dose,
    )


def t_config_invalide():

    mauvais = [
        dict(
            alphas=(0.5,)
        ),
        dict(
            n_qubits=9
        ),
        dict(
            dose_reelle_cible=1.2
        ),
        dict(
            gains=(-1.0,)
        ),
        dict(
            modes=("inconnu",)
        ),
        dict(
            bruit_interne=-1.0
        ),
    ]

    acceptes = 0

    for kw in mauvais:

        try:
            Config(**kw)
            acceptes += 1

        except ValueError:
            pass

    return (
        float(acceptes),
        0.0,
    )


def t_hamiltonien_effectif():
    """
    Sans rétroaction ni bruit, l'état reste dans le sous-espace
    à une excitation.

    Évolution équivalente :

        H_eff = H0 - i gamma/2 n_A

    """
    alpha = 0.2
    amp = 7.0

    cfg = Config(
        n_stress=40,
        n_recup=10,
        alphas=(alpha,),
    )

    M = Modele(
        cfg,
        alpha,
    )

    profil = construire_profil(
        "chronique",
        cfg.n_stress,
        cfg.n_recup,
    )

    r = simuler(
        cfg,
        M,
        0.0,
        profil,
        amp,
        np.zeros(
            (
                len(profil),
                2,
            )
        ),
    )

    sp0 = embed(
        SP,
        0,
        2,
    )

    sm0 = embed(
        SM,
        0,
        2,
    )

    sp1 = embed(
        SP,
        1,
        2,
    )

    sm1 = embed(
        SM,
        1,
        2,
    )

    H0 = (
        cfg.couplage_J
        * (
            sp0 @ sm1
            + sm0 @ sp1
        )
        + 0.5
        * alpha
        * cfg.echelle_desaccord
        * (
            embed(
                SZ,
                0,
                2,
            )
            - embed(
                SZ,
                1,
                2,
            )
        )
    )

    nA = embed(
        NN,
        0,
        2,
    )

    psi = np.zeros(
        4,
        dtype=complex,
    )

    psi[2] = 1.0

    for p in profil:

        g = float(
            np.tanh(
                p
                * amp
                * cfg.decay
            )
        )

        psi = (
            expm(
                -1j
                * (
                    H0
                    - 0.5j
                    * g
                    * nA
                )
            )
            @ psi
        )

    e1 = abs(
        r["dose_reelle"]
        - (
            1.0
            - float(
                np.vdot(
                    psi,
                    psi,
                ).real
            )
        )
    )

    e2 = abs(
        r["F_final"]
        - abs(
            psi[2]
        ) ** 2
    )

    return (
        max(
            e1,
            e2,
        ),
        1e-11,
    )


def t_config_json():

    cfg = Config(
        n_qubits=3,
        alphas=(
            0.1,
            -0.1,
        ),
        modes=(
            "aigu",
        ),
    )

    with tempfile.TemporaryDirectory() as d:

        f = Path(d) / "cfg.json"

        cfg.to_json(f)

        identique = (
            Config.from_json(f)
            == cfg
        )

    try:

        Config.from_dict(
            {
                "parametre_inconnu": 1
            }
        )

        rejet = False

    except ValueError:

        rejet = True

    return (
        float(
            not (
                identique
                and rejet
            )
        ),
        0.0,
    )


def t_reprise_campagne():

    cfg = Config(
        n_graines=1,
        n_stress=20,
        n_recup=5,
        gains=(0.0,),
        alphas=(0.0,),
        modes=("aigu",),
        calibration_grille_points=10,
    )

    with tempfile.TemporaryDirectory() as d:

        sortie = Path(d)

        a = executer(
            cfg,
            sortie,
            False,
            lambda *_: None,
        )

        b = executer(
            cfg,
            sortie,
            True,
            lambda *_: None,
        )

        n_lignes = len(
            (
                sortie
                / "points.jsonl"
            )
            .read_text(
                encoding="utf-8"
            )
            .strip()
            .splitlines()
        )

    ok = (
        len(a)
        == len(b)
        == 1
        and a[0]["amplitude"]
        == b[0]["amplitude"]
        and n_lignes
        == 2
    )

    return (
        float(not ok),
        0.0,
    )


TESTS = [
    (
        "expm : unitaire vs diagonalisation",
        t_expm_unitaire,
    ),
    (
        "Rabi : P_e = sin^2(Omega t / 2)",
        t_rabi,
    ),
    (
        "Amortissement : P_e = exp(-g t) et dose exacte",
        t_amortissement_et_dose,
    ),
    (
        "Échange 2 qubits : P = cos^2(J t)",
        t_echange_deux_qubits,
    ),
    (
        "Semi-groupe : E(t)E(t) = E(2t)",
        t_semigroupe,
    ),
    (
        "Croisé : expm vs RK4 direct",
        t_rk4_croise,
    ),
    (
        "CPTP : trace, hermiticité, positivité",
        t_positivite_aleatoire,
    ),
    (
        "Concurrence : singulet, produit, Werner",
        t_concurrence,
    ),
    (
        "Négativité log. : singulet, produit, Werner",
        t_neg_log,
    ),
    (
        "Entropie et fidélité : cas connus",
        t_entropie_fidelite,
    ),
    (
        "Reproductibilité : même graine, même résultat",
        t_reproductibilite,
    ),
    (
        "Simulation : conservation",
        t_conservation_simulation,
    ),
    (
        "Calibration : dose cible atteinte",
        t_calibration,
    ),
    (
        "Exact : hamiltonien effectif",
        t_hamiltonien_effectif,
    ),
    (
        "Config : aller-retour JSON",
        t_config_json,
    ),
    (
        "Campagne : reprise après interruption",
        t_reprise_campagne,
    ),
    (
        "Config : entrées invalides rejetées",
        t_config_invalide,
    ),
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

            ok = bool(
                mesure <= seuil
            )

            etat = (
                "PASS"
                if ok
                else "FAIL"
            )

            detail = (
                f"mesure={mesure:.2e} "
                f"seuil={seuil:.0e}"
            )

        except Exception as exc:

            ok = False

            etat = "ERR "

            detail = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        tout_ok &= ok

        print(
            f"[{etat}] "
            f"{nom:<52} "
            f"{detail} "
            f"({time.perf_counter() - t0:.2f}s)"
        )

    print("=" * 78)

    print(
        "RÉSULTAT :",
        (
            "tous les tests passent"
            if tout_ok
            else "ÉCHEC"
        ),
    )

    return tout_ok


# ============================================================================
# SECTION 4 : CAMPAGNE
# ============================================================================

METRIQUES = (
    "amplitude",
    "dose_reelle",
    "F_min",
    "F_final",
    "C_max",
    "C_final",
    "EN_max",
    "EN_final",
    "purete_finale",
    "entropie_finale",
    "distance_stress",
    "distance_finale",
    "amelioration_recuperation",
    "max_violation",
    "max_residu_N",
)


def empreinte(
    cfg: Config,
) -> str:
    """Empreinte SHA-256 abrégée de la configuration."""
    payload = json.dumps(
        cfg.to_dict(),
        sort_keys=True,
    ).encode()

    return hashlib.sha256(
        payload
    ).hexdigest()[:16]


def manifeste(
    cfg: Config,
) -> dict:

    return {
        "version": VERSION,
        "config_hash": empreinte(cfg),
        "config": cfg.to_dict(),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "plateforme": platform.platform(),
        "debut": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    }


def _taches(cfg: Config):

    seeds = (
        np.random.default_rng(
            cfg.seed
        )
        .integers(
            0,
            2**31 - 1,
            size=cfg.n_graines,
        )
    )

    for mode in cfg.modes:

        for gain in cfg.gains:

            for alpha in cfg.alphas:

                for i, s in enumerate(
                    seeds
                ):

                    yield (
                        mode,
                        float(gain),
                        float(alpha),
                        i,
                        int(s),
                    )


def _cle(
    mode,
    gain,
    alpha,
    i,
) -> str:

    return (
        f"{mode}|"
        f"{gain:.6g}|"
        f"{alpha:.6g}|"
        f"{i}"
    )


def _lire_points(
    chemin: Path,
    cfg: Config,
) -> dict:

    faits: dict = {}

    if not chemin.exists():
        return faits

    with chemin.open(
        encoding="utf-8"
    ) as fh:

        for n, ligne in enumerate(
            fh
        ):

            ligne = ligne.strip()

            if not ligne:
                continue

            try:
                obj = json.loads(
                    ligne
                )

            except json.JSONDecodeError:
                # Dernière ligne potentiellement tronquée
                # par une interruption.
                continue

            if n == 0:

                if (
                    obj.get(
                        "manifeste",
                        {},
                    ).get(
                        "config_hash"
                    )
                    != empreinte(cfg)
                ):
                    raise RuntimeError(
                        "Le point de reprise "
                        "correspond à une autre "
                        "configuration."
                    )

                continue

            if "cle" in obj:
                faits[
                    obj["cle"]
                ] = obj

    return faits


def executer(
    cfg: Config,
    sortie: Path | None,
    reprendre: bool,
    parler=None,
) -> list[dict]:

    parler = (
        parler
        or log.info
    )

    points = (
        sortie
        / "points.jsonl"
        if sortie
        else None
    )

    faits: dict = {}

    if sortie:

        sortie.mkdir(
            parents=True,
            exist_ok=True,
        )

        if reprendre:

            faits = _lire_points(
                points,
                cfg,
            )

        elif points.exists():

            points.unlink()

        if not points.exists():

            points.write_text(
                json.dumps(
                    {
                        "manifeste":
                            manifeste(
                                cfg
                            )
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    taches = list(
        _taches(cfg)
    )

    modeles: dict = {}

    profils = {
        m:
            construire_profil(
                m,
                cfg.n_stress,
                cfg.n_recup,
            )
        for m in cfg.modes
    }

    resultats = []

    for k, (
        mode,
        gain,
        alpha,
        i,
        seed,
    ) in enumerate(
        taches,
        1,
    ):

        cle = _cle(
            mode,
            gain,
            alpha,
            i,
        )

        if cle in faits:

            resultats.append(
                faits[cle]
            )

            parler(
                f"[{k}/{len(taches)}] "
                f"{cle} : repris"
            )

            continue

        if alpha not in modeles:

            modeles[alpha] = Modele(
                cfg,
                alpha,
            )

        profil = profils[
            mode
        ]

        bruit = (
            np.random.default_rng(
                seed
            )
            .normal(
                0.0,
                cfg.bruit_interne,
                size=(
                    len(profil),
                    cfg.n_qubits,
                ),
            )
        )

        t0 = time.perf_counter()

        a, r = calibrer_dose(
            cfg,
            modeles[alpha],
            gain,
            profil,
            bruit,
        )

        ligne = {
            "cle": cle,
            "mode": mode,
            "gain": gain,
            "alpha": alpha,
            "graine": i,
            "statut": (
                "ok"
                if a is not None
                else r.get(
                    "statut",
                    "echec",
                )
            ),
            "duree_s":
                time.perf_counter()
                - t0,
        }

        if a is not None:

            ligne["amplitude"] = float(
                a
            )

            for m in METRIQUES[1:]:

                v = r.get(m)

                ligne[m] = (
                    None
                    if v is None
                    else float(v)
                )

            ligne[
                "non_monotone"
            ] = bool(
                r[
                    "calibration"
                ][
                    "non_monotone"
                ]
            )

        else:

            ligne[
                "meilleure_amp"
            ] = r.get(
                "meilleure_amp"
            )

            ligne[
                "erreur_cible"
            ] = r.get(
                "erreur_cible"
            )

            ligne[
                "non_monotone"
            ] = bool(
                r.get(
                    "non_monotone",
                    False,
                )
            )

        resultats.append(
            ligne
        )

        if points:

            with points.open(
                "a",
                encoding="utf-8",
            ) as fh:

                fh.write(
                    json.dumps(
                        ligne
                    )
                    + "\n"
                )

        parler(
            f"[{k}/{len(taches)}] "
            f"{cle} : "
            f"{ligne['statut']} "
            f"({ligne['duree_s']:.1f}s)"
        )

    return resultats


def agreger(
    resultats: list[dict],
) -> list[dict]:

    groupes: dict = {}

    for r in resultats:

        groupes.setdefault(
            (
                r["mode"],
                r["gain"],
                r["alpha"],
            ),
            [],
        ).append(r)

    out = []

    for (
        mode,
        gain,
        alpha,
    ), rows in groupes.items():

        ok = [
            r
            for r in rows
            if r["statut"]
            == "ok"
        ]

        ligne = {
            "mode": mode,
            "gain": gain,
            "alpha": alpha,
            "n_ok": len(ok),
            "n_total": len(rows),
            "non_monotone":
                sum(
                    1
                    for r in ok
                    if r.get(
                        "non_monotone"
                    )
                ),
        }

        for m in METRIQUES:

            v = [
                r[m]
                for r in ok
                if r.get(m)
                is not None
            ]

            ligne[
                m + "_moy"
            ] = (
                float(
                    np.mean(v)
                )
                if v
                else None
            )

            ligne[
                m + "_std"
            ] = (
                float(
                    np.std(
                        v,
                        ddof=1,
                    )
                )
                if len(v) > 1
                else None
            )

        for m in (
            "max_violation",
            "max_residu_N",
        ):

            v = [
                r[m]
                for r in ok
                if r.get(m)
                is not None
            ]

            ligne[
                m + "_max"
            ] = (
                float(
                    np.max(v)
                )
                if v
                else None
            )

        out.append(
            ligne
        )

    return out


def _f(
    x,
    largeur=6,
    nd=3,
):

    return (
        " "
        * (
            largeur - 1
        )
        + "-"
        if x is None
        else f"{x:{largeur}.{nd}f}"
    )


def afficher(
    cfg: Config,
    agg: list[dict],
    duree: float,
) -> None:

    deux = (
        cfg.n_qubits
        == 2
    )

    cle_int = (
        "C_max"
        if deux
        else "EN_max"
    )

    print("=" * 96)

    print(
        f" ÉDIFICE QUANTIQUE "
        f"{VERSION} | "
        f"qubits={cfg.n_qubits} | "
        f"graines={cfg.n_graines} | "
        f"cible={cfg.dose_reelle_cible} | "
        f"empreinte={empreinte(cfg)}"
    )

    print("=" * 96)

    mode_courant = None

    for a in agg:

        if (
            a["mode"]
            != mode_courant
        ):

            mode_courant = (
                a["mode"]
            )

            print(
                f"\nREGIME : "
                f"{mode_courant.upper()}"
            )

            print(
                f"{'Gain':<5}|"
                f"{'Alpha':<6}|"
                f"{'n':<5}|"
                f"{'Amp':>8}|"
                f"{'F_min':>6}|"
                f"{cle_int + ' (moy ± σ)':>18}|"
                f"{'Pureté':>7}|"
                f"{'Viol.max':>9}|"
                f"{'Res.N':>8}"
            )

            print("-" * 96)

        if a["n_ok"] == 0:

            print(
                f"{a['gain']:4.2f} |"
                f"{a['alpha']:5.2f} |"
                f"{a['n_ok']}/{a['n_total']} | "
                f"cible non atteinte"
            )

            continue

        ci = a[
            cle_int
            + "_moy"
        ]

        cs = a[
            cle_int
            + "_std"
        ]

        inter = (
            f"{ci:6.3f} ± "
            f"{cs:5.3f}"
            if cs is not None
            else
            f"{ci:6.3f} ±   -  "
        )

        rn = a[
            "max_residu_N_max"
        ]

        print(
            f"{a['gain']:4.2f} |"
            f"{a['alpha']:5.2f} |"
            f"{a['n_ok']}/"
            f"{a['n_total']:<3}|"
            f"{a['amplitude_moy']:8.3f}|"
            f"{_f(a['F_min_moy'])}|"
            f"{inter:>18}|"
            f"{_f(a['purete_finale_moy'], 7)}|"
            f"{a['max_violation_max']:9.1e}|"
            f"{('n/a' if rn is None else f'{rn:.1e}'):>8}"
        )

    print("=" * 96)

    print(
        f"Terminé en "
        f"{duree:.1f} s."
    )


def exporter(
    cfg: Config,
    resultats: list[dict],
    agg: list[dict],
    sortie: Path,
) -> None:

    sortie.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        sortie
        / "synthese.json"
    ).write_text(
        json.dumps(
            {
                "manifeste":
                    manifeste(
                        cfg
                    ),
                "synthese":
                    agg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for nom, lignes in (
        (
            "synthese.csv",
            agg,
        ),
        (
            "resultats_bruts.csv",
            resultats,
        ),
    ):

        if not lignes:
            continue

        champs = sorted(
            {
                k
                for l in lignes
                for k in l
            },
            key=lambda k: (
                k
                not in (
                    "mode",
                    "gain",
                    "alpha",
                ),
                k,
            ),
        )

        with (
            sortie
            / nom
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as fh:

            w = csv.DictWriter(
                fh,
                fieldnames=champs,
            )

            w.writeheader()

            w.writerows(
                lignes
            )


# ============================================================================
# SECTION 5 : CLI
# ============================================================================

def _liste(
    s: str,
) -> tuple[float, ...]:

    return tuple(
        float(x)
        for x in s.split(",")
        if x.strip()
        != ""
    )


def _cfg_depuis(a) -> Config:
    """
    --config est prioritaire :
    les autres paramètres de modèle sont alors ignorés.
    """

    if getattr(
        a,
        "config",
        None,
    ):

        return Config.from_json(
            Path(
                a.config
            ).expanduser()
        )

    kw: dict = {
        "n_qubits":
            a.qubits,
        "dose_reelle_cible":
            a.cible,
        "seed":
            a.seed,
        "bruit_interne":
            a.bruit,
        "dephasage":
            a.dephasage,
    }

    if a.graines is not None:

        kw[
            "n_graines"
        ] = a.graines

    if getattr(
        a,
        "rapide",
        False,
    ):

        kw.update(
            n_graines=min(
                kw.get(
                    "n_graines",
                    2,
                ),
                2,
            ),
            calibration_grille_points=20,
        )

    if getattr(
        a,
        "alphas",
        None,
    ):

        kw[
            "alphas"
        ] = _liste(
            a.alphas
        )

    if getattr(
        a,
        "gains",
        None,
    ):

        kw[
            "gains"
        ] = _liste(
            a.gains
        )

    if getattr(
        a,
        "modes",
        None,
    ):

        kw[
            "modes"
        ] = tuple(
            a.modes.split(",")
        )

    return Config(
        **kw
    )


def _commun(
    p: argparse.ArgumentParser,
) -> None:

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "fichier JSON de configuration "
            "(prioritaire sur les options ci-dessous)"
        ),
    )

    p.add_argument(
        "--qubits",
        type=int,
        default=2,
        help=(
            "2 à 4 "
            "(4 qubits : plus lent)"
        ),
    )

    p.add_argument(
        "--cible",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--graines",
        type=int,
        default=None,
    )

    p.add_argument(
        "--bruit",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--dephasage",
        type=float,
        default=0.0,
    )


def cmd_valider(a) -> int:

    return (
        0
        if lancer()
        else 1
    )


def cmd_config(a) -> int:

    cfg = _cfg_depuis(a)

    out = (
        Path(
            a.output
        )
        .expanduser()
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.to_json(
        out
    )

    print(
        f"Configuration écrite : "
        f"{out}"
    )

    return 0


def cmd_campagne(a) -> int:

    cfg = _cfg_depuis(a)

    sortie = (
        Path(
            a.output
        ).expanduser()
        if a.output
        else None
    )

    if (
        a.reprendre
        and not sortie
    ):

        print(
            "--reprendre exige "
            "--output",
            file=sys.stderr,
        )

        return 2

    t0 = time.perf_counter()

    res = executer(
        cfg,
        sortie,
        a.reprendre,
    )

    agg = agreger(
        res
    )

    afficher(
        cfg,
        agg,
        time.perf_counter()
        - t0,
    )

    if sortie:

        exporter(
            cfg,
            res,
            agg,
            sortie,
        )

        print(
            f"Exports et reprise : "
            f"{sortie}"
        )

    return 0


def cmd_trajectoire(a) -> int:

    cfg = _cfg_depuis(a)

    M = Modele(
        cfg,
        float(
            a.alpha
        ),
    )

    profil = construire_profil(
        a.mode,
        cfg.n_stress,
        cfg.n_recup,
    )

    bruit = (
        np.random.default_rng(
            cfg.seed
        )
        .normal(
            0.0,
            cfg.bruit_interne,
            size=(
                len(profil),
                cfg.n_qubits,
            ),
        )
    )

    amp = a.amplitude

    if amp is None:

        amp, _ = calibrer_dose(
            cfg,
            M,
            a.gain,
            profil,
            bruit,
        )

        if amp is None:

            print(
                "Calibration impossible "
                "pour cette configuration.",
                file=sys.stderr,
            )

            return 1

        print(
            f"Amplitude calibrée : "
            f"{amp:.6f}"
        )

    r = simuler(
        cfg,
        M,
        a.gain,
        profil,
        float(amp),
        bruit,
        enregistrer=True,
    )

    tr = r[
        "trajectoire"
    ]

    out = (
        Path(
            a.output
        )
        .expanduser()
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    champs = list(
        tr.keys()
    )

    with out.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:

        w = csv.writer(
            fh
        )

        w.writerow(
            champs
        )

        for i in range(
            len(
                tr["t"]
            )
        ):

            w.writerow(
                [
                    ""
                    if tr[c][i]
                    is None
                    else tr[c][i]
                    for c in champs
                ]
            )

    print(
        f"Dose finale "
        f"{r['dose_reelle']:.6f} | "
        f"violation max "
        f"{r['max_violation']:.1e} | "
        f"écrit : {out}"
    )

    return 0


def construire_parseur() -> argparse.ArgumentParser:

    ap = argparse.ArgumentParser(
        prog="edifice",
        description=(
            "Édifice quantique : "
            "qubits ouverts, Lindblad."
        ),
    )

    ap.add_argument(
        "--version",
        action="version",
        version=f"edifice {__version__}",
    )

    ap.add_argument(
        "-v",
        "--verbeux",
        action="store_true",
        help="journal détaillé",
    )

    sub = (
        ap.add_subparsers(
            dest="cmd",
            required=True,
        )
    )

    # ------------------------------------------------------------------
    # valider
    # ------------------------------------------------------------------

    sub.add_parser(
        "valider",
        help=(
            "batterie de validation "
            "analytique et croisée"
        ),
    ).set_defaults(
        f=cmd_valider
    )

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------

    pf = sub.add_parser(
        "config",
        help=(
            "écrit un fichier "
            "de configuration JSON"
        ),
    )

    _commun(pf)

    pf.add_argument(
        "--output",
        type=str,
        required=True,
    )

    pf.set_defaults(
        f=cmd_config
    )

    # ------------------------------------------------------------------
    # campagne
    # ------------------------------------------------------------------

    pc = sub.add_parser(
        "campagne",
        help=(
            "plan expérimental calibré "
            "avec reprise"
        ),
    )

    _commun(pc)

    pc.add_argument(
        "--rapide",
        action="store_true",
        help=(
            "2 graines, grille 20"
        ),
    )

    pc.add_argument(
        "--alphas",
        type=str,
        default=None,
        help=(
            "ex. -0.2,-0.1,0,0.1,0.2"
        ),
    )

    pc.add_argument(
        "--gains",
        type=str,
        default=None,
        help=(
            "ex. 0,0.4,0.8"
        ),
    )

    pc.add_argument(
        "--modes",
        type=str,
        default=None,
        help=(
            "chronique,aigu"
        ),
    )

    pc.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "dossier de sortie"
        ),
    )

    pc.add_argument(
        "--reprendre",
        action="store_true",
        help=(
            "reprend un calcul interrompu"
        ),
    )

    pc.set_defaults(
        f=cmd_campagne
    )

    # ------------------------------------------------------------------
    # trajectoire
    # ------------------------------------------------------------------

    pt = sub.add_parser(
        "trajectoire",
        help=(
            "série temporelle "
            "d'une simulation"
        ),
    )

    _commun(pt)

    pt.add_argument(
        "--mode",
        choices=(
            "chronique",
            "aigu",
        ),
        default="chronique",
    )

    pt.add_argument(
        "--gain",
        type=float,
        default=0.0,
    )

    pt.add_argument(
        "--alpha",
        type=float,
        default=0.0,
    )

    pt.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help=(
            "sinon calibrée "
            "sur la dose cible"
        ),
    )

    pt.add_argument(
        "--output",
        type=str,
        required=True,
    )

    pt.set_defaults(
        f=cmd_trajectoire
    )

    return ap


def main(
    argv=None,
) -> int:

    a = (
        construire_parseur()
        .parse_args(argv)
    )

    logging.basicConfig(
        level=(
            logging.DEBUG
            if a.verbeux
            else logging.INFO
        ),
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )

    try:

        return a.f(a)

    except (
        ValueError,
        RuntimeError,
        AssertionError,
        OSError,
        json.JSONDecodeError,
    ) as exc:

        print(
            f"Erreur : {exc}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    sys.exit(
        main()
  )
