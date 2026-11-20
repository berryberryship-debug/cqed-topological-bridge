#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOYAU FLUXONIUM 4Q
=================

Trois blocs implémentés :
1. Solveur Fluxonium en base harmonique LC.
2. Relaxation par quasiparticules froides dans la petite jonction.
3. Dynamique Lindblad à quatre qubits et mesures projectives
   des stabilisateurs du code détecteur [[4,2,2]].

CONVENTIONS
-----------
EC, EL, EJ, Delta : énergies divisées par h, en GHz.
flux : flux externe en unités de Phi0.
phase externe : 2*pi*flux.
dt_s : secondes.
J_Hz : fréquence de couplage en Hz.
Gamma_qp : taux descendant en s^-1.

MODÈLE QUASIPARTICULES
---------------------
Gamma_down =
    (8/pi) * (EJ/hbar) * x_qp * sqrt(2*Delta/f01)
    * |<0|sin((phi - phase_ext)/2)|1>|²

Les rapports Delta/f01 utilisent ici des fréquences en GHz.

Hypothèses :
- quasiparticules froides ;
- énergie typique des quasiparticules << h*f01 << gap ;
- petite jonction uniquement ;
- aucun canal de superinductance ;
- aucun taux montant ;
- aucun plancher artificiel.

DYNAMIQUE ET MESURES
-------------------
Le générateur de Lindblad est constant.
Propagation numérique par scipy.sparse.linalg.expm_multiply.

Après chaque intervalle :
- calcul des quatre probabilités de syndromes ;
- évolution d'une branche avec mesure non sélective ;
- évolution d'une branche post-sélectionnée sur (++).

Aucune correction d'erreur inconnue n'est annoncée.
Le code [[4,2,2]] est utilisé pour la détection.

ATTENTION AU RÉFÉRENTIEL
-----------------------
Le Hamiltonien local est conservé dans le référentiel du laboratoire.
L'évolution cohérente peut donc modifier XXXX sans dissipation.
Une détection n'est pas automatiquement une signature de QP.

La fidélité conditionnelle est une fidélité à l'état GHZ initial,
pas une fidélité de porte ni une fidélité logique universelle.

Dépendances :
    numpy
    scipy

Aucun import depuis les autres fichiers du projet.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import expm_multiply


# ============================================================================
# 1. CONSTANTES ET VALIDATION
# ============================================================================

I2 = np.eye(2, dtype=complex)

X = np.array(
    [[0.0, 1.0], [1.0, 0.0]],
    dtype=complex,
)

Y = np.array(
    [[0.0, -1j], [1j, 0.0]],
    dtype=complex,
)

Z = np.diag([1.0, -1.0]).astype(complex)

SM = np.array(
    [[0.0, 1.0], [0.0, 0.0]],
    dtype=complex,
)

N = np.diag([0.0, 1.0]).astype(complex)

TWOPI = 2.0 * math.pi


def check(condition, message):
    """Assertion explicite, conservée même avec python -O."""
    if not condition:
        raise AssertionError(message)


def nonnegative(name, value):
    """Vérifie une valeur finie et non négative."""
    if not math.isfinite(float(value)) or value < 0.0:
        raise ValueError(
            f"{name} doit être fini et supérieur ou égal à zéro."
        )


def integer(name, value, minimum):
    """Vérifie un entier, sans accepter les booléens."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(
            f"{name} doit être un entier supérieur ou égal à {minimum}."
        )


@dataclass(frozen=True)
class Config:
    """Configuration physique et numérique."""

    EC: float = 1.0
    EL: float = 0.5
    EJ: float = 4.0

    flux: float = 0.5

    basis: int = 45
    padding: int = 35

    x_qp: float = 1.0e-8
    gap_GHz: float = 45.0

    J_Hz: float = 0.0

    steps: int = 20
    dt_s: float = 100.0e-9

    tolerance: float = 1.0e-9

    def __post_init__(self):
        for key in (
            "EC",
            "EL",
            "EJ",
            "gap_GHz",
            "dt_s",
            "tolerance",
        ):
            value = getattr(self, key)

            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{key} doit être fini et strictement positif."
                )

        nonnegative("x_qp", self.x_qp)

        for key in ("flux", "J_Hz"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} doit être fini.")

        integer("basis", self.basis, 10)
        integer("padding", self.padding, 0)
        integer("steps", self.steps, 1)


# ============================================================================
# 2. SOLVEUR FLUXONIUM
# ============================================================================

def solve_fluxonium(cfg):
    """
    Résout le fluxonium dans une base harmonique LC.

    H/h =
        4 EC n²
        + EL phi²/2
        - EJ cos(phi - 2*pi*flux).

    La partie LC est diagonale analytiquement.

    Les fonctions trigonométriques sont calculées dans une base
    auxiliaire plus grande, puis projetées dans la base de travail.

    Retour :
        f01_GHz
        f12_GHz
        phi01
        qp01
        ground_tail_last5
    """
    dim = cfg.basis
    aux = dim + cfg.padding

    zpf = (2.0 * cfg.EC / cfg.EL) ** 0.25

    a = np.diag(
        np.sqrt(np.arange(1, aux)),
        1,
    )

    phi = zpf * (a + a.T)

    values, vectors = np.linalg.eigh(phi)

    projected_vectors = vectors[:dim, :]

    phase = TWOPI * cfg.flux

    cos_op = (
        projected_vectors * np.cos(values - phase)
    ) @ projected_vectors.T

    qp_op = (
        projected_vectors * np.sin((values - phase) / 2.0)
    ) @ projected_vectors.T

    f_lc_GHz = math.sqrt(8.0 * cfg.EC * cfg.EL)

    H_lc = np.diag(
        f_lc_GHz * (np.arange(dim) + 0.5)
    )

    H = H_lc - cfg.EJ * cos_op

    energies, states = np.linalg.eigh(H)

    phi_op = phi[:dim, :dim]

    state0 = states[:, 0]
    state1 = states[:, 1]

    return {
        "f01_GHz": float(energies[1] - energies[0]),
        "f12_GHz": float(energies[2] - energies[1]),

        "phi01": float(
            abs(np.vdot(state0, phi_op @ state1))
        ),

        "qp01": float(
            abs(np.vdot(state0, qp_op @ state1))
        ),

        "ground_tail_last5": float(
            np.sum(abs(state0[-5:]) ** 2)
        ),
    }


# ============================================================================
# 3. RELAXATION PAR QUASIPARTICULES
# ============================================================================

def qp_rate(cfg, spectral):
    """
    Taux QP descendant de la petite jonction, en s^-1.

    Aucun plancher ni coefficient de réglage arbitraire.

    x_qp = 0 donne exactement Gamma_qp = 0.
    """
    f01 = spectral["f01_GHz"]

    if not math.isfinite(f01) or f01 <= 0.0:
        raise ValueError("f01 doit être fini et strictement positif.")

    EJ_over_hbar = TWOPI * 1.0e9 * cfg.EJ

    return (
        (8.0 / math.pi)
        * EJ_over_hbar
        * cfg.x_qp
        * math.sqrt(2.0 * cfg.gap_GHz / f01)
        * spectral["qp01"] ** 2
    )


# ============================================================================
# 4. OPÉRATEURS À QUATRE QUBITS
# ============================================================================

def kron_all(operators):
    """Produit tensoriel d'une séquence d'opérateurs."""
    result = np.array([[1.0]], dtype=complex)

    for operator in operators:
        result = np.kron(result, operator)

    return result


def local(operator, index):
    """Place un opérateur à un qubit sur le qubit index."""
    operators = [I2] * 4
    operators[index] = operator

    return kron_all(operators)


def stabilizers():
    """Stabilisateurs du code détecteur [[4,2,2]]."""
    S_Z = kron_all([Z] * 4)
    S_X = kron_all([X] * 4)

    return S_Z, S_X


def projectors():
    """
    Projecteurs conjoints sur les quatre secteurs de syndrome.

    P(z,x) = (I + z ZZZZ)(I + x XXXX)/4.
    """
    S_Z, S_X = stabilizers()
    identity = np.eye(16, dtype=complex)

    return {
        (z, x): (
            (identity + z * S_Z)
            @ (identity + x * S_X)
        ) / 4.0

        for z in (1, -1)
        for x in (1, -1)
    }


def ghz_density():
    """Matrice densité de (|0000> + |1111>)/sqrt(2)."""
    ket = np.zeros(16, dtype=complex)

    ket[0] = 1.0 / math.sqrt(2.0)
    ket[-1] = 1.0 / math.sqrt(2.0)

    return np.outer(ket, ket.conj())


def measure(rho):
    """
    Branches non normalisées d'une mesure projective.

    La trace de chaque branche est sa probabilité.
    """
    return {
        key: projector @ rho @ projector

        for key, projector in projectors().items()
    }


# ============================================================================
# 5. CONTRÔLES DE MATRICE DENSITÉ
# ============================================================================

def density_check(rho, tolerance, normalized=True):
    """Vérifie finitude, hermiticité, positivité et trace."""
    if not np.all(np.isfinite(rho)):
        raise AssertionError("État non fini.")

    check(
        np.linalg.norm(rho - rho.conj().T) <= tolerance,
        "Hermiticité perdue.",
    )

    trace = np.trace(rho)

    check(
        abs(trace.imag) <= tolerance,
        "Trace complexe.",
    )

    hermitian_part = (rho + rho.conj().T) / 2.0

    check(
        np.linalg.eigvalsh(hermitian_part)[0] >= -tolerance,
        "Positivité perdue.",
    )

    if normalized:
        check(
            abs(trace.real - 1.0) <= tolerance,
            "Trace non conservée.",
        )
    else:
        check(
            -tolerance <= trace.real <= 1.0 + tolerance,
            "Poids de branche invalide.",
        )


# ============================================================================
# 6. GÉNÉRATEUR DE LINDBLAD
# ============================================================================

def liouvillian(cfg, spectral, gamma):
    """
    Générateur de Lindblad à quatre qubits.

    Vectorisation par colonnes :
        vec(A rho B) = (B.T kron A) vec(rho).

    Hamiltonien exprimé en rad/s.

    Le terme local est +h*f01*|1><1| :
    |1> est donc bien l'état excité.

    Le couplage XX est un couplage effectif donné en entrée.
    Il n'est pas dérivé ici d'une géométrie de circuit.
    """
    H = np.zeros((16, 16), dtype=complex)

    omega01 = TWOPI * 1.0e9 * spectral["f01_GHz"]

    for qubit in range(4):
        H += omega01 * local(N, qubit)

    for qubit in range(3):
        H += (
            TWOPI
            * cfg.J_Hz
            * (
                local(X, qubit)
                @ local(X, qubit + 1)
            )
        )

    H = sp.csr_matrix(H)

    identity = sp.eye(
        16,
        format="csr",
        dtype=complex,
    )

    generator = -1j * (
        sp.kron(identity, H)
        - sp.kron(H.T, identity)
    )

    for qubit in range(4):
        collapse = sp.csr_matrix(
            math.sqrt(gamma) * local(SM, qubit)
        )

        product = collapse.conj().T @ collapse

        generator += (
            sp.kron(collapse.conj(), collapse)
            - 0.5 * sp.kron(identity, product)
            - 0.5 * sp.kron(product.T, identity)
        )

    return generator.tocsr()


# ============================================================================
# 7. SIMULATION ET MESURES PROJECTIVES
# ============================================================================

def simulate(cfg):
    """
    Simule deux branches.

    rho :
        état inconditionnel après mesure non sélective.

    kept :
        état non normalisé des trajectoires ayant obtenu
        (++), à toutes les mesures précédentes.

    trace(kept) :
        probabilité cumulée de survie à la post-sélection.
    """
    spectral = solve_fluxonium(cfg)
    gamma = qp_rate(cfg, spectral)

    generator = (
        liouvillian(cfg, spectral, gamma)
        * cfg.dt_s
    )

    generator_trace = generator.diagonal().sum()

    rho = ghz_density()
    kept = rho.copy()

    initial = rho.copy()

    history = []

    S_Z, S_X = stabilizers()

    for step in range(1, cfg.steps + 1):
        columns = np.column_stack(
            [
                rho.reshape(-1, order="F"),
                kept.reshape(-1, order="F"),
            ]
        )

        evolved = expm_multiply(
            generator,
            columns,
            traceA=generator_trace,
        )

        pre_measure = evolved[:, 0].reshape(
            (16, 16),
            order="F",
        )

        kept_pre_measure = evolved[:, 1].reshape(
            (16, 16),
            order="F",
        )

        density_check(
            pre_measure,
            cfg.tolerance,
        )

        branches = measure(pre_measure)

        probabilities = {
            f"Z{z:+d}_X{x:+d}": float(
                np.trace(branch).real
            )

            for (z, x), branch in branches.items()
        }

        rho = sum(
            branches.values(),
            np.zeros((16, 16), dtype=complex),
        )

        P_plus = projectors()[(1, 1)]

        kept = (
            P_plus
            @ kept_pre_measure
            @ P_plus
        )

        density_check(
            rho,
            cfg.tolerance,
        )

        density_check(
            kept,
            cfg.tolerance,
            normalized=False,
        )

        survival = float(np.trace(kept).real)

        if survival > 1.0e-15:
            fidelity = float(
                np.trace(initial @ kept).real
                / survival
            )
        else:
            fidelity = None

        history.append(
            {
                "step": step,

                "time_s": step * cfg.dt_s,

                "pre_measure_ZZZZ": float(
                    np.trace(S_Z @ pre_measure).real
                ),

                "pre_measure_XXXX": float(
                    np.trace(S_X @ pre_measure).real
                ),

                "syndrome_probabilities": probabilities,

                "survival_all_plus": survival,

                "GHZ_fidelity_conditioned_all_plus": fidelity,

                "unconditional_purity": float(
                    np.trace(rho @ rho).real
                ),
            }
        )

    return {
        "config": asdict(cfg),

        "spectral": spectral,

        "Gamma_qp_down_s_inv": gamma,

        "model": (
            "cold-QP small junction; "
            "ideal projective detection, no recovery"
        ),

        "history": history,
    }


# ============================================================================
# 8. TESTS INTÉGRÉS
# ============================================================================

def run_tests():
    """
    Tests structurels et numériques.

    Ils ne remplacent pas une validation expérimentale ni une
    étude de convergence sur tout le domaine des paramètres.
    """
    cfg = Config(
        basis=35,
        padding=35,
    )

    spectral = solve_fluxonium(cfg)

    check(
        spectral["f01_GHz"] > 0.0,
        "Transition invalide.",
    )

    # Convergence locale entre deux tailles de base.
    spectral_larger = solve_fluxonium(
        replace(
            cfg,
            basis=55,
            padding=45,
        )
    )

    np.testing.assert_allclose(
        spectral["f01_GHz"],
        spectral_larger["f01_GHz"],
        rtol=1.0e-3,
    )

    # Symétrie du spectre autour du demi-quantum de flux.
    left = solve_fluxonium(
        replace(cfg, flux=0.49)
    )

    right = solve_fluxonium(
        replace(cfg, flux=0.51)
    )

    np.testing.assert_allclose(
        left["f01_GHz"],
        right["f01_GHz"],
        atol=1.0e-10,
    )

    # Aucun taux QP à densité nulle.
    check(
        qp_rate(
            replace(cfg, x_qp=0.0),
            spectral,
        ) == 0.0,
        "QP non nulles à densité nulle.",
    )

    # Proportionnalité du taux à la densité.
    np.testing.assert_allclose(
        qp_rate(
            replace(cfg, x_qp=2.0e-8),
            spectral,
        ),
        2.0 * qp_rate(cfg, spectral),
    )

    # Complétude des projecteurs.
    syndrome_projectors = projectors()

    np.testing.assert_allclose(
        sum(syndrome_projectors.values()),
        np.eye(16),
        atol=1.0e-12,
    )

    # Idempotence de chaque projecteur.
    for projector in syndrome_projectors.values():
        np.testing.assert_allclose(
            projector @ projector,
            projector,
            atol=1.0e-12,
        )

    # Détection de chaque erreur Pauli simple.
    rho = ghz_density()

    error_cases = (
        (X, (-1, 1)),
        (Z, (1, -1)),
        (Y, (-1, -1)),
    )

    for operator, target_syndrome in error_cases:
        for qubit in range(4):
            error = local(operator, qubit)

            corrupted = (
                error
                @ rho
                @ error.conj().T
            )

            branches = measure(corrupted)

            np.testing.assert_allclose(
                np.trace(branches[target_syndrome]),
                1.0,
                atol=1.0e-12,
            )

            np.testing.assert_allclose(
                np.trace(branches[(1, 1)]),
                0.0,
                atol=1.0e-12,
            )

            # Inverser une erreur connue est vérifiable.
            # Cela n'implémente pas un décodeur d'erreur inconnue.
            restored = (
                error
                @ corrupted
                @ error.conj().T
            )

            np.testing.assert_allclose(
                restored,
                rho,
                atol=1.0e-12,
            )

    # Complétude des Kraus d'amortissement.
    for qubit in range(4):
        probability = 0.2

        K0 = local(
            np.diag(
                [
                    1.0,
                    math.sqrt(1.0 - probability),
                ]
            ),
            qubit,
        )

        K1 = local(
            math.sqrt(probability) * SM,
            qubit,
        )

        np.testing.assert_allclose(
            K0.conj().T @ K0
            + K1.conj().T @ K1,
            np.eye(16),
            atol=1.0e-12,
        )

    # À une période entière du qubit, GHZ revient à lui-même
    # sans bruit et sans interaction.
    dt_period = 1.0 / (
        spectral["f01_GHz"] * 1.0e9
    )

    noiseless = simulate(
        replace(
            cfg,
            x_qp=0.0,
            steps=2,
            dt_s=dt_period,
        )
    )

    np.testing.assert_allclose(
        noiseless["history"][-1]["survival_all_plus"],
        1.0,
        atol=1.0e-9,
    )

    # Une dissipation non nulle produit des rejets.
    noisy = simulate(
        replace(
            cfg,
            steps=2,
            dt_s=dt_period,
            x_qp=1.0e-4,
        )
    )

    check(
        noisy["history"][-1]["survival_all_plus"] < 1.0,
        "Détection QP absente.",
    )

    print(
        "Tests réussis : spectre, convergence locale, QP, "
        "projecteurs, 12 erreurs Pauli, Kraus et dynamique."
    )


# ============================================================================
# 9. INTERFACE EN LIGNE DE COMMANDE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quatre qubits effectifs issus du Fluxonium : "
            "QP et détection projective [[4,2,2]]."
        )
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Nombre d'intervalles et de mesures.",
    )

    parser.add_argument(
        "--dt-ns",
        type=float,
        default=100.0,
        help="Durée de chaque intervalle en ns.",
    )

    parser.add_argument(
        "--basis",
        type=int,
        default=45,
        help="Dimension de la base harmonique.",
    )

    parser.add_argument(
        "--flux",
        type=float,
        default=0.5,
        help="Flux externe en unités de Phi0.",
    )

    parser.add_argument(
        "--xqp",
        type=float,
        default=1.0e-8,
        help="Densité normalisée de quasiparticules.",
    )

    parser.add_argument(
        "--J-Hz",
        type=float,
        default=0.0,
        help="Couplage XX effectif en Hz.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Fichier JSON de sortie.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Exécuter les tests intégrés.",
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    cfg = Config(
        steps=args.steps,
        dt_s=args.dt_ns * 1.0e-9,
        basis=args.basis,
        flux=args.flux,
        x_qp=args.xqp,
        J_Hz=args.J_Hz,
    )

    result = simulate(cfg)

    summary = {
        key: value
        for key, value in result.items()
        if key != "history"
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )

    print("Dernier intervalle :")

    print(
        json.dumps(
            result["history"][-1],
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )

    if args.output:
        path = Path(args.output).expanduser()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        print(f"Résultats enregistrés : {path}")


if __name__ == "__main__":
    main()
