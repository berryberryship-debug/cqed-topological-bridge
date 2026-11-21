#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLUXONIUM DYNAMICS — MULTILEVEL QUANTUM CONTROL
================================================

Évolution temporelle d'un Fluxonium dans sa base propre.

Hamiltonien :

    H(t) = H0
         + delta_phi(t) * dH/dphi_ext
         + EL/2 * delta_phi(t)^2 * I
         + u_n(t) * n

avec :

    H0 = diag(E0, E1, ..., EN)

    dH/dphi_ext = EL * (phi_ext * I - phi_matrix)

Le terme quadratique est conservé.

Conventions :
    EJ, EC, EL       : GHz
    phi_ext          : radians
    delta_phi        : phase réduite
    u_n              : GHz
    temps            : secondes

Évolution :

    dU/dt = -i * 2*pi*1e9 * H(t) * U

Limites :
    - base spectrale tronquée ;
    - commandes effectives non calibrées matériellement ;
    - évolution unitaire sans dissipation ;
    - le bruit et la décohérence sont traités séparément.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp


# ============================================================
# CONSTANTES
# ============================================================

GHZ_TO_RAD_S = 2.0 * np.pi * 1e9


# ============================================================
# IMPULSIONS
# ============================================================

@dataclass(frozen=True)
class GaussianPulse:

    amplitude: float
    center_s: float
    sigma_s: float

    def __post_init__(self):

        if not np.isfinite(self.amplitude):
            raise ValueError("Amplitude non finie.")

        if not np.isfinite(self.center_s):
            raise ValueError("center_s non fini.")

        if not np.isfinite(self.sigma_s) or self.sigma_s <= 0:
            raise ValueError("sigma_s doit être positif.")

    def __call__(self, t: float) -> float:

        x = (t - self.center_s) / self.sigma_s

        return float(
            self.amplitude * np.exp(-0.5 * x**2)
        )


@dataclass(frozen=True)
class SquarePulse:

    amplitude: float
    start_s: float
    stop_s: float

    def __post_init__(self):

        if not np.isfinite(self.amplitude):
            raise ValueError("Amplitude non finie.")

        if not np.isfinite(self.start_s):
            raise ValueError("start_s non fini.")

        if not np.isfinite(self.stop_s):
            raise ValueError("stop_s non fini.")

        if self.stop_s <= self.start_s:
            raise ValueError("stop_s doit être supérieur à start_s.")

    def __call__(self, t: float) -> float:

        return float(
            self.amplitude
            if self.start_s <= t <= self.stop_s
            else 0.0
        )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class DynamicsConfig:

    t_start_s: float
    t_stop_s: float

    rtol: float = 1e-9
    atol: float = 1e-11

    method: str = "DOP853"

    max_step_s: Optional[float] = None

    def __post_init__(self):

        if not np.isfinite(self.t_start_s):
            raise ValueError("t_start_s non fini.")

        if not np.isfinite(self.t_stop_s):
            raise ValueError("t_stop_s non fini.")

        if self.t_stop_s <= self.t_start_s:
            raise ValueError("Intervalle temporel invalide.")

        if (
            not np.isfinite(self.rtol)
            or not np.isfinite(self.atol)
            or self.rtol <= 0
            or self.atol <= 0
        ):
            raise ValueError("Tolérances invalides.")

        if self.max_step_s is not None:

            if (
                not np.isfinite(self.max_step_s)
                or self.max_step_s <= 0
            ):
                raise ValueError("max_step_s invalide.")


# ============================================================
# RÉSULTATS
# ============================================================

@dataclass
class DynamicsResult:

    times_s: np.ndarray
    states: np.ndarray

    initial_states: np.ndarray
    final_states: np.ndarray

    norm_history: np.ndarray

    solver_success: bool
    solver_message: str

    @property
    def norm_error(self) -> float:

        return float(
            np.max(
                np.abs(self.norm_history - 1.0)
            )
        )


# ============================================================
# MOTEUR DE DYNAMIQUE
# ============================================================

class FluxoniumDynamics:

    def __init__(
        self,
        spectral_result,
        EL_GHz: float,
        phi_ext: float,
    ):

        if not np.isfinite(EL_GHz) or EL_GHz <= 0:
            raise ValueError("EL_GHz doit être positif.")

        if not np.isfinite(phi_ext):
            raise ValueError("phi_ext doit être fini.")

        self.energies = np.asarray(
            spectral_result.energies_GHz,
            dtype=float,
        )

        self.phi_ext = float(phi_ext)
        self.EL_GHz = float(EL_GHz)

        self.dimension = len(self.energies)

        if self.dimension < 2:
            raise ValueError("Au moins deux niveaux sont nécessaires.")

        if not np.all(np.isfinite(self.energies)):
            raise ValueError("Énergies propres non finies.")

        elements = spectral_result.matrix_elements

        self.phi_matrix = np.asarray(
            elements["phi"],
            dtype=complex,
        )

        self.n_matrix = np.asarray(
            elements["n"],
            dtype=complex,
        )

        expected_shape = (
            self.dimension,
            self.dimension,
        )

        if self.phi_matrix.shape != expected_shape:
            raise ValueError("Dimensions incorrectes pour phi_matrix.")

        if self.n_matrix.shape != expected_shape:
            raise ValueError("Dimensions incorrectes pour n_matrix.")

        self.H0 = np.diag(
            self.energies
        ).astype(complex)

        identity = np.eye(
            self.dimension,
            dtype=complex,
        )

        self.dH_dphi_ext = self.EL_GHz * (
            self.phi_ext * identity - self.phi_matrix
        )

        self._validate_hermitian("H0", self.H0)
        self._validate_hermitian(
            "dH_dphi_ext",
            self.dH_dphi_ext,
        )
        self._validate_hermitian(
            "phi_matrix",
            self.phi_matrix,
        )
        self._validate_hermitian(
            "n_matrix",
            self.n_matrix,
        )

    # --------------------------------------------------------
    # VALIDATION HERMITIENNE
    # --------------------------------------------------------

    @staticmethod
    def _validate_hermitian(
        name: str,
        matrix: np.ndarray,
        tolerance: float = 1e-8,
    ):

        error = np.max(
            np.abs(
                matrix - matrix.conj().T
            )
        )

        if error > tolerance:

            raise ValueError(
                f"{name} non hermitien : {error:.3e}"
            )

    # --------------------------------------------------------
    # HAMILTONIEN TEMPOREL
    # --------------------------------------------------------

    def hamiltonian(
        self,
        t: float,
        flux_drive: Optional[Callable] = None,
        charge_drive: Optional[Callable] = None,
    ) -> np.ndarray:

        H = self.H0.copy()

        if flux_drive is not None:

            delta_phi = float(flux_drive(t))

            if not np.isfinite(delta_phi):
                raise ValueError("Commande de flux non finie.")

            H += delta_phi * self.dH_dphi_ext

            H += (
                0.5
                * self.EL_GHz
                * delta_phi**2
                * np.eye(
                    self.dimension,
                    dtype=complex,
                )
            )

        if charge_drive is not None:

            u_n = float(charge_drive(t))

            if not np.isfinite(u_n):
                raise ValueError("Commande de charge non finie.")

            H += u_n * self.n_matrix

        return H

    # --------------------------------------------------------
    # ÉVOLUTION SCHRÖDINGER
    # --------------------------------------------------------

    def evolve(
        self,
        initial_states: np.ndarray,
        config: DynamicsConfig,
        flux_drive: Optional[Callable] = None,
        charge_drive: Optional[Callable] = None,
        output_times: Optional[np.ndarray] = None,
    ) -> DynamicsResult:

        initial = np.asarray(
            initial_states,
            dtype=complex,
        )

        vector_input = initial.ndim == 1

        if vector_input:
            initial = initial[:, None]

        if initial.ndim != 2:
            raise ValueError("initial_states doit être un vecteur ou une matrice.")

        if initial.shape[0] != self.dimension:
            raise ValueError("Dimension initiale incompatible.")

        if not np.all(np.isfinite(initial)):
            raise ValueError("État initial non fini.")

        norms = np.sum(
            np.abs(initial)**2,
            axis=0,
        )

        if np.any(norms <= 0):
            raise ValueError("État initial de norme nulle.")

        if not np.allclose(
            norms,
            1.0,
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError("Les états initiaux doivent être normalisés.")

        if output_times is None:

            output_times = np.linspace(
                config.t_start_s,
                config.t_stop_s,
                501,
            )

        else:

            output_times = np.asarray(
                output_times,
                dtype=float,
            )

        if output_times.ndim != 1 or output_times.size < 2:
            raise ValueError("output_times invalide.")

        if not np.all(np.isfinite(output_times)):
            raise ValueError("Temps de sortie non finis.")

        if np.any(np.diff(output_times) <= 0):
            raise ValueError("output_times doit être croissant.")

        if (
            output_times[0] < config.t_start_s
            or output_times[-1] > config.t_stop_s
        ):
            raise ValueError("Temps de sortie hors intervalle.")

        shape = initial.shape
        y0 = initial.reshape(-1)

        def schrodinger_rhs(t, y):

            states = y.reshape(shape)

            H = self.hamiltonian(
                t,
                flux_drive=flux_drive,
                charge_drive=charge_drive,
            )

            derivative = (
                -1j
                * GHZ_TO_RAD_S
                * (H @ states)
            )

            return derivative.reshape(-1)

        kwargs = {}

        if config.max_step_s is not None:
            kwargs["max_step"] = config.max_step_s

        solution = solve_ivp(
            schrodinger_rhs,
            (
                config.t_start_s,
                config.t_stop_s,
            ),
            y0,
            method=config.method,
            t_eval=output_times,
            rtol=config.rtol,
            atol=config.atol,
            **kwargs,
        )

        if not solution.success:

            raise RuntimeError(
                f"Échec intégration : {solution.message}"
            )

        states = solution.y.T.reshape(
            len(solution.t),
            shape[0],
            shape[1],
        )

        norm_history = np.sum(
            np.abs(states)**2,
            axis=1,
        )

        final_states = states[-1].copy()

        if vector_input:

            states = states[:, :, 0]
            final_states = final_states[:, 0]

        return DynamicsResult(
            times_s=solution.t,
            states=states,
            initial_states=initial.copy(),
            final_states=final_states,
            norm_history=norm_history,
            solver_success=bool(solution.success),
            solver_message=str(solution.message),
        )

    # --------------------------------------------------------
    # PROPAGATEUR COMPLET
    # --------------------------------------------------------

    def propagate(
        self,
        config: DynamicsConfig,
        flux_drive=None,
        charge_drive=None,
        output_times=None,
    ) -> np.ndarray:

        identity = np.eye(
            self.dimension,
            dtype=complex,
        )

        result = self.evolve(
            initial_states=identity,
            config=config,
            flux_drive=flux_drive,
            charge_drive=charge_drive,
            output_times=output_times,
        )

        return result.final_states

    # --------------------------------------------------------
    # MÉTRIQUES DE PORTE
    # --------------------------------------------------------

    @staticmethod
    def gate_metrics(
        propagator: np.ndarray,
        target_gate: np.ndarray,
        computational_levels: tuple = (0, 1),
    ) -> dict:

        U = np.asarray(
            propagator,
            dtype=complex,
        )

        target = np.asarray(
            target_gate,
            dtype=complex,
        )

        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            raise ValueError("propagator doit être carrée.")

        if not np.all(np.isfinite(U)):
            raise ValueError("propagator non fini.")

        levels = np.asarray(
            computational_levels,
            dtype=int,
        )

        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("Sous-espace logique invalide.")

        if np.any(levels < 0) or np.any(levels >= U.shape[0]):
            raise ValueError("Niveau logique hors limites.")

        if len(set(levels.tolist())) != len(levels):
            raise ValueError("Niveaux logiques dupliqués.")

        d = len(levels)

        if target.shape != (d, d):
            raise ValueError("Dimension de porte cible incorrecte.")

        target_unitarity = np.max(
            np.abs(
                target.conj().T @ target
                - np.eye(d)
            )
        )

        if target_unitarity > 1e-8:
            raise ValueError("target_gate doit être unitaire.")

        unitarity_error = float(
            np.max(
                np.abs(
                    U.conj().T @ U
                    - np.eye(U.shape[0])
                )
            )
        )

        K = U[np.ix_(levels, levels)]

        survival = float(
            np.trace(
                K.conj().T @ K
            ).real
        )

        overlap = np.trace(
            target.conj().T @ K
        )

        fidelity = (
            survival + abs(overlap)**2
        ) / (d * (d + 1))

        leakage = 1.0 - survival / d

        return {
            "average_gate_fidelity": float(
                np.clip(fidelity, 0.0, 1.0)
            ),
            "average_leakage": float(
                np.clip(leakage, 0.0, 1.0)
            ),
            "survival_probability_sum": survival,
            "unitarity_error": unitarity_error,
            "projected_propagator": K,
            "target_overlap": complex(overlap),
        }

    # --------------------------------------------------------
    # CONTRÔLE DE LA NORME
    # --------------------------------------------------------

    @staticmethod
    def validate_norm(
        result: DynamicsResult,
        tolerance: float = 1e-7,
    ) -> dict:

        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance doit être positive.")

        error = result.norm_error

        return {
            "norm_error": error,
            "tolerance": float(tolerance),
            "passed": bool(error <= tolerance),
        }


# ============================================================
# PORTES DE RÉFÉRENCE
# ============================================================

def identity_gate():

    return np.eye(2, dtype=complex)


def pauli_x_gate():

    return np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )


def pauli_y_gate():

    return np.array(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=complex,
    )


def pauli_z_gate():

    return np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=complex,
    )


def hadamard_gate():

    return np.array(
        [
            [1, 1],
            [1, -1],
        ],
        dtype=complex,
    ) / np.sqrt(2.0)


# ============================================================
# TESTS ANALYTIQUES
# ============================================================

def run_self_tests() -> dict:
    """
    Tests analytiques indépendants du solveur Fluxonium.

    Vérifie :
        1. évolution identité ;
        2. porte X à phase globale près ;
        3. conservation de la norme ;
        4. fidélité moyenne ;
        5. unitarité du propagateur.
    """

    zero_result = type(
        "SpectralResult",
        (),
        {
            "energies_GHz": np.array([0.0, 0.0]),
            "matrix_elements": {
                "phi": np.zeros((2, 2), dtype=complex),
                "n": np.zeros((2, 2), dtype=complex),
            },
        },
    )()

    engine = FluxoniumDynamics(
        spectral_result=zero_result,
        EL_GHz=1.0,
        phi_ext=0.0,
    )

    # --------------------------------------------------------
    # TEST 1 : IDENTITÉ
    # --------------------------------------------------------

    config_identity = DynamicsConfig(
        t_start_s=0.0,
        t_stop_s=1e-9,
        rtol=1e-10,
        atol=1e-12,
    )

    U_identity = engine.propagate(
        config_identity
    )

    identity_error = float(
        np.max(
            np.abs(
                U_identity - identity_gate()
            )
        )
    )

    identity_metrics = engine.gate_metrics(
        U_identity,
        identity_gate(),
    )

    # --------------------------------------------------------
    # TEST 2 : PORTE X
    # --------------------------------------------------------

    sigma_x = pauli_x_gate()

    engine.H0 = 0.25 * sigma_x

    config_x = DynamicsConfig(
        t_start_s=0.0,
        t_stop_s=1e-9,
        rtol=1e-10,
        atol=1e-12,
    )

    U_x = engine.propagate(
        config_x
    )

    # H = 0.25 GHz * sigma_x
    #
    # U(1 ns) = exp(-i*pi/2*sigma_x)
    #         = -i*sigma_x

    target_with_phase = -1j * sigma_x

    x_error = float(
        np.max(
            np.abs(
                U_x - target_with_phase
            )
        )
    )

    x_metrics = engine.gate_metrics(
        U_x,
        sigma_x,
    )

    # --------------------------------------------------------
    # TEST 3 : CONSERVATION DE LA NORME
    # --------------------------------------------------------

    initial = np.array(
        [1.0, 0.0],
        dtype=complex,
    )

    evolution = engine.evolve(
        initial_states=initial,
        config=config_x,
    )

    norm_metrics = engine.validate_norm(
        evolution,
        tolerance=1e-8,
    )

    # --------------------------------------------------------
    # TEST 4 : UNITARITÉ
    # --------------------------------------------------------

    unitarity_passed = (
        x_metrics["unitarity_error"] < 1e-8
    )

    # --------------------------------------------------------
    # RAPPORT
    # --------------------------------------------------------

    passed = (
        identity_error < 1e-8
        and x_error < 1e-7
        and identity_metrics["average_gate_fidelity"] > 1 - 1e-10
        and x_metrics["average_gate_fidelity"] > 1 - 1e-9
        and norm_metrics["passed"]
        and unitarity_passed
    )

    return {
        "passed": bool(passed),
        "identity_error": identity_error,
        "x_gate_error": x_error,
        "identity_fidelity": identity_metrics[
            "average_gate_fidelity"
        ],
        "x_gate_fidelity": x_metrics[
            "average_gate_fidelity"
        ],
        "norm_error": norm_metrics["norm_error"],
        "norm_test_passed": norm_metrics["passed"],
        "unitarity_error": x_metrics["unitarity_error"],
        "unitarity_test_passed": bool(unitarity_passed),
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "GaussianPulse",
    "SquarePulse",
    "DynamicsConfig",
    "DynamicsResult",
    "FluxoniumDynamics",
    "identity_gate",
    "pauli_x_gate",
    "pauli_y_gate",
    "pauli_z_gate",
    "hadamard_gate",
    "run_self_tests",
]


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":

    import json

    report = run_self_tests()

    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    if not report["passed"]:
        raise SystemExit(1)
