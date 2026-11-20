#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module : pulse_calibration.py
Calibration d'impulsions micro-ondes (Gaussienne & DRAG) avec correction
de décalage de Bloch-Siegert à basse fréquence et intégration de l'équation
maîtresse de Lindblad pour le contrôle haute fidélité du Fluxonium.
"""

import math
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PulseConfig:
    t_gate_ns: float = 20.0
    drag_coeff: float = 0.0
    detuning_GHz: float = 0.0
    bloch_siegert_comp: bool = True
    n_steps: int = 300


class FluxoniumPulseCalibrator:
    """Simulateur dynamique Lindblad et calibrateur de portes quantiques."""

    def __init__(
        self,
        energies_GHz: np.ndarray,
        phi_matrix: np.ndarray,
        T1_us: float = 100.0,
        T_phi_us: float = 100.0,
        dim: int = 4
    ):
        self.dim = dim
        E_rel = (energies_GHz[:dim] - energies_GHz[0]) * 2.0 * np.pi
        self.H0 = np.diag(E_rel).astype(np.complex128)
        self.phi_op = np.real(phi_matrix[:dim, :dim]).astype(np.complex128)

        self.omega_01 = E_rel[1]
        self.alpha_rad_ns = (E_rel[2] - E_rel[1]) - E_rel[1] if dim > 2 else 10.0
        self.phi_01 = abs(self.phi_op[0, 1])
        self.collapse_ops = self._build_lindblad_ops(T1_us, T_phi_us)

    def _build_lindblad_ops(self, T1_us: float, T_phi_us: float) -> List[np.ndarray]:
        """Construit les opérateurs de relaxation et de déphasage en unités ns^-1."""
        ops = []
        g1 = 1.0 / max(T1_us * 1e3, 1.0)
        gphi = 1.0 / max(T_phi_us * 1e3, 1.0)

        L10 = np.zeros((self.dim, self.dim), dtype=np.complex128)
        L10[0, 1] = math.sqrt(g1)
        ops.append(L10)

        if self.dim > 2:
            L21 = np.zeros((self.dim, self.dim), dtype=np.complex128)
            L21[1, 2] = math.sqrt(2.0 * g1)
            ops.append(L21)

        Lphi = np.zeros((self.dim, self.dim), dtype=np.complex128)
        Lphi[1, 1] = math.sqrt(2.0 * gphi)
        ops.append(Lphi)
        return ops

    def _envelopes(self, t: float, cfg: PulseConfig):
        """Retourne les composantes en phase I(t) et en quadrature Q(t) (DRAG)."""
        if t < 0.0 or t > cfg.t_gate_ns:
            return 0.0, 0.0
        sigma = cfg.t_gate_ns / 4.0
        t0 = 0.5 * cfg.t_gate_ns
        g_t = math.exp(-0.5 * ((t - t0) / sigma) ** 2)
        g_0 = math.exp(-0.5 * ((0.0 - t0) / sigma) ** 2)
        env_I = (g_t - g_0) / (1.0 - g_0)

        d_env = -((t - t0) / (sigma ** 2)) * g_t / (1.0 - g_0)
        env_Q = -(cfg.drag_coeff / self.alpha_rad_ns) * d_env
        return env_I, env_Q

    def run_gate(self, cfg: PulseConfig, target_angle: float = np.pi) -> Dict[str, np.ndarray]:
        """Intègre l'équation maîtresse de Lindblad sur la durée t_gate_ns."""
        t_eval = np.linspace(0.0, cfg.t_gate_ns, cfg.n_steps)
        env_I_arr = np.array([self._envelopes(t, cfg)[0] for t in t_eval])
        area = np.trapezoid(env_I_arr, t_eval)

        omega_max = (2.0 * target_angle) / (max(self.phi_01, 1e-4) * area)

        # Compensation dynamique du décalage de Bloch-Siegert (régime basse fréquence Fluxonium)
        bs_shift = 0.0
        if cfg.bloch_siegert_comp and self.omega_01 > 1e-4:
            omega_rabi_eff = 0.5 * omega_max * self.phi_01
            bs_shift = (omega_rabi_eff ** 2) / (4.0 * self.omega_01)

        omega_drive = self.omega_01 + bs_shift + 2.0 * np.pi * cfg.detuning_GHz

        def rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
            rho = rho_vec.reshape((self.dim, self.dim))
            env_I, env_Q = self._envelopes(t, cfg)
            signal = omega_max * (env_I * math.cos(omega_drive * t) + env_Q * math.sin(omega_drive * t))
            H_t = self.H0 + signal * self.phi_op

            drho = -1j * (H_t @ rho - rho @ H_t)
            for L in self.collapse_ops:
                Ld = np.conj(L.T)
                drho += L @ rho @ Ld - 0.5 * ((Ld @ L) @ rho + rho @ (Ld @ L))
            return drho.reshape(-1)

        rho0 = np.zeros((self.dim, self.dim), dtype=np.complex128)
        rho0[0, 0] = 1.0

        sol = solve_ivp(
            rhs,
            (0.0, cfg.t_gate_ns),
            rho0.reshape(-1),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-7,
            atol=1e-9
        )
        pops = np.array([
            np.real(np.diag(sol.y[:, i].reshape((self.dim, self.dim))))
            for i in range(cfg.n_steps)
        ])

        return {
            "time_ns": t_eval,
            "populations": pops,
            "fidelity_X": float(pops[-1, 1]),
            "max_leakage_P2": float(np.max(pops[:, 2])) if self.dim > 2 else 0.0,
            "omega_max_GHz": omega_max / (2.0 * np.pi),
            "bloch_siegert_MHz": (bs_shift / (2.0 * np.pi)) * 1e3
        }

    def optimize_drag(
        self,
        t_gate_ns: float = 16.0,
        drag_grid: np.ndarray = np.linspace(-1.0, 1.0, 11)
    ) -> Dict[str, float]:
        """Balaye le coefficient DRAG pour minimiser la fuite P(|2>)."""
        best_drag = 0.0
        min_leak = 1.0
        best_fid = 0.0

        for d_val in drag_grid:
            res = self.run_gate(
                PulseConfig(t_gate_ns=t_gate_ns, drag_coeff=float(d_val), n_steps=180)
            )
            if res["max_leakage_P2"] < min_leak:
                min_leak = res["max_leakage_P2"]
                best_drag = float(d_val)
                best_fid = res["fidelity_X"]

        return {
            "optimal_drag": best_drag,
            "min_leakage_P2": min_leak,
            "fidelity_X": best_fid
        }


if __name__ == "__main__":
    energies_demo = np.array([0.6928, 0.9300, 4.6359, 6.2000])
    phi_demo = np.zeros((4, 4))
    phi_demo[0, 1] = phi_demo[1, 0] = 2.5851
    phi_demo[1, 2] = phi_demo[2, 1] = 2.9500

    cal = FluxoniumPulseCalibrator(energies_demo, phi_demo, T1_us=150.0, T_phi_us=200.0, dim=4)
    res = cal.run_gate(PulseConfig(t_gate_ns=20.0))
    print(
        f"[Pulse Lindblad OK] Fidélité porte X = {res['fidelity_X']*100:.4f}% | "
        f"Fuite max P(|2>) = {res['max_leakage_P2']*100:.5f}% | "
        f"Shift Bloch-Siegert = +{res['bloch_siegert_MHz']:.2f} MHz"
    )
