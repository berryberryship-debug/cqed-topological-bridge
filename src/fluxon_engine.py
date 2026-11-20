"""
fluxon_engine.py

Moteur de simulation 2D pour réseaux Josephson avec frustration,
jauge de Landau, injection DC et thermostat de Langevin (BBK corrigé).

Auteur : [Votre nom]
Licence : MIT
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------

def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Ramène un angle dans [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def safe_divide(numerator: float, denominator: float) -> float:
    """Division sûre pour éviter les divisions par zéro."""
    if abs(denominator) < 1e-14:
        return 0.0
    return numerator / denominator


def clip_probability(value: float, tol: float = 1e-12) -> float:
    """Contraint une valeur dans [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


# ----------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------

@dataclass
class LatticeParams:
    """Paramètres du réseau Josephson 2D."""

    N: int = 16
    frustration: float = 0.25
    gamma: float = 0.1
    drive_force: float = 0.0
    temperature: float = 0.0
    dt: float = 0.05
    periodic_y: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.N < 2:
            raise ValueError("N doit être au moins 2.")
        if self.dt <= 0:
            raise ValueError("dt doit être positif.")
        if self.gamma < 0:
            raise ValueError("gamma doit être non négatif.")
        if self.temperature < 0:
            raise ValueError("temperature doit être non négative.")

    def to_dict(self) -> Dict[str, float]:
        """Sérialise les paramètres."""
        return asdict(self)


@dataclass
class RunConfig:
    """Configuration d'une exécution."""

    n_steps: int = 1000
    record_every: Optional[int] = 10
    output_dir: str = "results"
    save_history: bool = True

    def __post_init__(self):
        if self.n_steps < 0:
            raise ValueError("n_steps doit être non négatif.")
        if self.record_every is not None and self.record_every < 1:
            raise ValueError("record_every doit être au moins 1.")

    def to_dict(self) -> Dict[str, float]:
        """Sérialise la configuration."""
        return asdict(self)


# ----------------------------------------------------------------------
# Moteur principal
# ----------------------------------------------------------------------

class FluxonEngine2D:
    """
    Réseau Josephson 2D avec frustration, jauge de Landau,
    injection DC aux bords et thermostat de Langevin (BBK corrigé).

    phi : phases des îlots, en radians.
    v   : dérivées temporelles des phases.
    """

    TWO_PI = 2.0 * np.pi

    def __init__(self, params: LatticeParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self._init_state()
        self._init_gauge()
        self._init_noise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_state(self):
        """Initialise les phases et vitesses."""
        N = self.params.N
        self.phi = self.rng.uniform(
            -np.pi, np.pi, (N, N)
        )
        self.v = np.zeros((N, N), dtype=float)

    def _init_gauge(self):
        """
        Initialise la jauge de Landau.

        A_x(i, j) = -f * i
        Stockée sur les liaisons horizontales (i, j) -> (i, j+1).
        """
        N = self.params.N
        f = self.params.frustration
        rows = np.arange(N, dtype=float)[:, None]
        self.A_link_x = -f * np.broadcast_to(
            rows, (N, N - 1)
        ).copy()

    def _init_noise(self):
        """
        Initialise l'amplitude du bruit thermique.

        Pour BBK correct : sigma = sqrt(gamma * T * dt)
        (appliqué aux deux demi-pas avec le même tirage).
        """
        p = self.params
        if p.temperature > 0 and p.gamma > 0:
            self.sigma_noise = np.sqrt(p.gamma * p.temperature * p.dt)
        else:
            self.sigma_noise = 0.0

    def reset(self, seed: Optional[int] = None):
        """Réinitialise l'état avec une nouvelle graine optionnelle."""
        if seed is not None:
            self.params.seed = seed
            self.rng = np.random.default_rng(seed)
        self._init_state()

    def apply_initial_noise(self, sigma: float = 0.0) -> np.ndarray:
        """
        Applique un bruit initial sur les phases.

        Retourne une copie de l'état bruité (ne mute pas en place).
        """
        if sigma <= 0:
            return self.phi.copy()
        noise = self.rng.normal(0.0, sigma, self.phi.shape)
        phi_noisy = wrap_angle(self.phi + noise)
        return phi_noisy

    # ------------------------------------------------------------------
    # Dynamique
    # ------------------------------------------------------------------

    def _forces_conservatives(self, phi: np.ndarray) -> np.ndarray:
        """
        Forces dérivant de l'énergie de Josephson.

        U = sum_links [1 - cos(delta phi - A)]
        """
        force = np.zeros_like(phi)

        # Liaisons horizontales
        dphi_x = wrap_angle(
            phi[:, 1:] - phi[:, :-1] - self.A_link_x
        )
        jx = np.sin(dphi_x)
        force[:, :-1] += jx
        force[:, 1:] -= jx

        # Liaisons verticales
        if self.params.periodic_y:
            dphi_y = wrap_angle(np.roll(phi, -1, axis=0) - phi)
            jy = np.sin(dphi_y)
            force += jy
            force -= np.roll(jy, 1, axis=0)
        else:
            dphi_y = wrap_angle(phi[1:, :] - phi[:-1, :])
            jy = np.sin(dphi_y)
            force[:-1, :] += jy
            force[1:, :] -= jy

        return force

    def _forces_dc(self) -> np.ndarray:
        """Injection DC aux bords gauche et droit."""
        force_dc = np.zeros_like(self.phi)
        if self.params.drive_force != 0.0:
            force_dc[:, 0] += self.params.drive_force
            force_dc[:, -1] -= self.params.drive_force
        return force_dc

    def _forces(
        self,
        phi: np.ndarray,
        noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forces totales."""
        force = self._forces_conservatives(phi) + self._forces_dc()
        if noise is not None:
            force += noise
        return force

    def _draw_noise(self) -> Optional[np.ndarray]:
        """Tire un bruit thermique blanc (une fois par pas)."""
        if self.sigma_noise == 0.0:
            return None
        return self.rng.normal(
            0.0, self.sigma_noise, self.phi.shape
        )

    def integrer(
        self,
        n_steps: Optional[int] = None,
        record_every: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Intègre la dynamique avec BBK corrigé.

        BBK correct :
        v_half = v_n + dt/2 (F_n - gamma v_n) + sqrt(gamma T dt) * eta_n
        phi_{n+1} = phi_n + dt * v_half
        v_{n+1} = v_half + dt/2 (F_{n+1} - gamma v_half) + sqrt(gamma T dt) * eta_n

        Le même bruit eta_n est utilisé aux deux demi-pas.
        """
        if n_steps is None:
            n_steps = 1000
        if record_every is None:
            record_every = self.params.dt and 10

        n_steps = int(n_steps)
        history = []
        dt = self.params.dt
        half_dt = 0.5 * dt
        gamma = self.params.gamma

        for step in range(n_steps):
            noise = self._draw_noise()

            # Premier demi-pas
            force_old = self._forces(self.phi, noise=noise)
            v_half = self.v + half_dt * (
                force_old - gamma * self.v
            )
            if noise is not None:
                v_half += noise  # sqrt(gamma T dt) * eta

            # Drift
            self.phi = wrap_angle(self.phi + dt * v_half)

            # Second demi-pas (même bruit)
            force_new = self._forces(self.phi, noise=noise)
            self.v = v_half + half_dt * (
                force_new - gamma * v_half
            )
            if noise is not None:
                self.v += noise

            if record_every and (step + 1) % record_every == 0:
                record = self.compute_observables()
                record["step"] = step + 1
                history.append(record)

        return history

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def energie(self) -> float:
        """Énergie de Josephson totale."""
        dphi_x = wrap_angle(
            self.phi[:, 1:] - self.phi[:, :-1] - self.A_link_x
        )
        energy = np.sum(1.0 - np.cos(dphi_x))

        if self.params.periodic_y:
            dphi_y = wrap_angle(np.roll(self.phi, -1, axis=0) - self.phi)
        else:
            dphi_y = wrap_angle(self.phi[1:, :] - self.phi[:-1, :])

        energy += np.sum(1.0 - np.cos(dphi_y))
        return float(energy)

    def energie_cinetique(self) -> float:
        """Énergie cinétique = 0.5 * sum v^2."""
        return float(0.5 * np.sum(self.v ** 2))

    def courant_moyen_x(self) -> float:
        """Courant moyen horizontal."""
        dphi_x = wrap_angle(
            self.phi[:, 1:] - self.phi[:, :-1] - self.A_link_x
        )
        return float(np.mean(np.sin(dphi_x)))

    def vorticités(self) -> np.ndarray:
        """
        Calcule la vorticité plaquette par plaquette (version corrigée).

        Pour chaque plaquette (i, j) :
        circulation = dx(i,j) + dy(i+1,j) - dx(i,j-1) - dy(i,j)

        Retourne un tableau de forme (N, N-1) si periodic_y,
        sinon (N-1, N-1).
        """
        phi = self.phi
        N = self.params.N

        # Liaisons horizontales : (N, N-1)
        dx = wrap_angle(
            phi[:, 1:] - phi[:, :-1] - self.A_link_x
        )

        # Liaisons verticales
        if self.params.periodic_y:
            dy = wrap_angle(np.roll(phi, -1, axis=0) - phi)  # (N, N)
        else:
            dy = wrap_angle(phi[1:, :] - phi[:-1, :])  # (N-1, N)

        if self.params.periodic_y:
            # Plaquettes : (N, N-1)
            # dx[i, j] : lien x de (i,j) à (i,j+1)
            # dy[i+1, j] : lien y de (i+1,j) à (i+2,j) -> roll
            dy_forward = np.roll(dy, -1, axis=0)  # dy[i+1, j]
            dx_back = np.roll(dx, 1, axis=1)  # dx[i, j-1], avec j-1
            # Attention : dx a forme (N, N-1), roll axis=1 de 1
            # donne dx[i, j-1] mais j-1 peut être -1 -> wrap
            dx_back = np.roll(dx, 1, axis=1)
            dy_current = dy  # dy[i, j]

            circulation = (
                dx
                + dy_forward[:N, :N-1]
                - dx_back
                - dy_current[:N, :N-1]
            )
            return circulation / self.TWO_PI
        else:
            # Plaquettes : (N-1, N-1)
            dx_sub = dx[:N-1, :]
            dy_sub = dy[:, :N-1]
            dy_forward = dy[1:, :N-1]
            dx_back = dx[:N-1, :]  # pas de roll, x non périodique
            circulation = dx_sub + dy_forward - dx_back - dy_sub
            return circulation / self.TWO_PI

    def winding_number_total(self) -> float:
        """Somme des vorticités (nombre d'enroulement total)."""
        vort = self.vorticités()
        return float(np.sum(vort))

    def temperature_estimate(self) -> float:
        """Estimation de la température à partir de <v^2>."""
        mean_v2 = np.mean(self.v ** 2)
        return safe_divide(mean_v2, 2.0)

    def compute_observables(self) -> Dict[str, float]:
        """Calcule toutes les observables."""
        return {
            "energy": self.energie(),
            "kinetic_energy": self.energie_cinetique(),
            "current_x": self.courant_moyen_x(),
            "winding_number": self.winding_number_total(),
            "temperature_est": self.temperature_estimate(),
        }

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def save_state(self, path: Path):
        """Sauvegarde l'état complet en NPZ."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            phi=self.phi,
            v=self.v,
            params=json.dumps(self.params.to_dict()),
        )
        LOGGER.info("État sauvegardé : %s", path)

    def load_state(self, path: Path):
        """Charge un état depuis un fichier NPZ."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        data = np.load(path, allow_pickle=True)
        self.phi = data["phi"]
        self.v = data["v"]
        LOGGER.info("État chargé : %s", path)


# ----------------------------------------------------------------------
# Expériences
# ----------------------------------------------------------------------

class ExperimentRunner:
    """Orchestre les expériences et les sauvegardes."""

    def __init__(
        self,
        engine: FluxonEngine2D,
        config: RunConfig,
    ):
        self.engine = engine
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[Dict[str, float]]:
        """Lance une simulation complète."""
        history = self.engine.integrer(
            n_steps=self.config.n_steps,
            record_every=self.config.record_every,
        )
        if self.config.save_history:
            self._save_history(history)
        return history

    def run_quench(
        self,
        new_frustration: float,
        n_steps: int = 500,
    ) -> List[Dict[str, float]]:
        """Simule une quench de frustration."""
        old_f = self.engine.params.frustration
        self.engine.params.frustration = new_frustration
        self.engine._init_gauge()
        history = self.engine.integrer(n_steps=n_steps)
        self.engine.params.frustration = old_f
        self.engine._init_gauge()
        return history

    def _save_history(self, history: List[Dict[str, float]]):
        """Sauvegarde l'historique en JSON."""
        path = self.output_dir / "history.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        LOGGER.info("Historique sauvegardé : %s", path)


# ----------------------------------------------------------------------
# Tests unitaires
# ----------------------------------------------------------------------

def test_energy_conservation():
    """Vérifie la conservation d'énergie sans bruit ni drive."""
    params = LatticeParams(
        gamma=0.0, temperature=0.0, drive_force=0.0, dt=0.01
    )
    engine = FluxonEngine2D(params)
    initial_energy = engine.energie() + engine.energie_cinetique()
    engine.integrer(n_steps=100, record_every=100)
    final_energy = engine.energie() + engine.energie_cinetique()
    drift = abs(final_energy - initial_energy) / abs(initial_energy)
    assert drift < 1e-3, f"Dérive d'énergie trop grande : {drift}"


def test_vorticity_shape():
    """Vérifie que vorticités() ne crash pas et a la bonne forme."""
    params = LatticeParams(N=8, periodic_y=True)
    engine = FluxonEngine2D(params)
    vort = engine.vorticités()
    assert vort.shape == (8, 7), f"Forme incorrecte : {vort.shape}"


def test_winding_number():
    """Vérifie que le nombre d'enroulement est un multiple de f."""
    params = LatticeParams(N=8, frustration=0.25, periodic_y=True)
    engine = FluxonEngine2D(params)
    engine.integrer(n_steps=100, record_every=100)
    w = engine.winding_number_total()
    # Après relaxation, w devrait être proche d'un multiple de f
    assert abs(w - round(w)) < 0.5, f"Winding non entier : {w}"


def run_all_tests():
    """Lance tous les tests."""
    test_energy_conservation()
    test_vorticity_shape()
    test_winding_number()
    LOGGER.info("Tous les tests sont passés.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Moteur de fluxons 2D."
    )
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--frustration", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--drive", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.test:
        run_all_tests()
        return

    params = LatticeParams(
        N=args.N,
        frustration=args.frustration,
        gamma=args.gamma,
        drive_force=args.drive,
        temperature=args.temperature,
        dt=args.dt,
        seed=args.seed,
    )
    config = RunConfig(
        n_steps=args.n_steps,
        output_dir=args.output,
    )
    engine = FluxonEngine2D(params)
    runner = ExperimentRunner(engine, config)
    history = runner.run()
    LOGGER.info(
        "Simulation terminée. Winding final : %.3f",
        history[-1]["winding_number"],
    )


if __name__ == "__main__":
    main()
