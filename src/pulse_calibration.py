"""
pulse_calibration.py

Calibration empirique des pulses quantiques (pi, pi/2, etc.).

Rôle
----
Fournir une interface générique pour calibrer l'amplitude d'un pulse
en balayant un facteur multiplicatif autour de l'amplitude RWA théorique,
puis en retenant le facteur qui maximise la fidélité.

Cela corrige le décalage entre RWA stricte et simulation complète
(hors RWA), où les termes contre-tournants introduisent un facteur
correctif (typiquement 0.5 pour un pulse pi).

Utilisation
-----------
Le module est agnostique au modèle physique : il prend en entrée une
fonction `simulate_pulse(amplitude) -> float` (fidélité) et une
amplitude RWA de référence.

Auteur : [Votre nom]
Licence : MIT
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """
    Configuration de la calibration.

    target_angle   : angle de rotation cible (pi pour un pulse pi)
    scan_min       : borne inférieure du facteur de scan
    scan_max       : borne supérieure du facteur de scan
    n_points       : nombre de points du scan
    rwa_correction : facteur correctif par défaut (0.5 pour pulse pi)
    """

    target_angle: float = np.pi
    scan_min: float = 0.3
    scan_max: float = 1.0
    n_points: int = 15
    rwa_correction: float = 0.5

    def __post_init__(self):
        if self.target_angle <= 0:
            raise ValueError("target_angle doit être positif.")
        if self.scan_min <= 0 or self.scan_max <= self.scan_min:
            raise ValueError("scan_min doit être positif et < scan_max.")
        if self.n_points < 2:
            raise ValueError("n_points doit être au moins 2.")
        if not 0 < self.rwa_correction <= 2.0:
            raise ValueError("rwa_correction doit être dans (0, 2].")

    def to_dict(self) -> Dict[str, float]:
        """Sérialise la configuration."""
        return asdict(self)


@dataclass
class CalibrationResult:
    """
    Résultat d'une calibration.

    best_factor    : facteur multiplicatif optimal
    best_fidelity  : fidélité maximale obtenue
    best_amplitude : amplitude calibrée
    scan_factors   : liste des facteurs testés
    scan_fidelities: liste des fidélités correspondantes
    """

    best_factor: float
    best_fidelity: float
    best_amplitude: float
    scan_factors: List[float]
    scan_fidelities: List[float]

    def to_dict(self) -> Dict[str, float]:
        """Sérialise le résultat."""
        return asdict(self)

    def save(self, path: Path):
        """Sauvegarde le résultat en JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info("Résultat de calibration sauvegardé : %s", path)


# ----------------------------------------------------------------------
# Calibration générique
# ----------------------------------------------------------------------

def calibrate_pulse(
    simulate_pulse: Callable[[float], float],
    amplitude_rwa: float,
    config: Optional[CalibrationConfig] = None,
) -> CalibrationResult:
    """
    Calibre l'amplitude d'un pulse par balayage.

    Paramètres
    ----------
    simulate_pulse : fonction amplitude -> fidélité (float entre 0 et 1)
    amplitude_rwa  : amplitude théorique RWA de référence
    config         : configuration de la calibration

    Retourne
    --------
    CalibrationResult avec le meilleur facteur, la meilleure fidélité
    et l'amplitude calibrée.
    """
    if config is None:
        config = CalibrationConfig()

    scan_factors = list(
        np.linspace(config.scan_min, config.scan_max, config.n_points)
    )
    scan_fidelities: List[float] = []

    best_factor = None
    best_fidelity = -1.0
    best_amplitude = None

    for factor in scan_factors:
        amplitude = amplitude_rwa * factor
        fidelity = simulate_pulse(amplitude)
        scan_fidelities.append(fidelity)

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_factor = factor
            best_amplitude = amplitude

    LOGGER.info(
        "Calibration : facteur=%.3f, fidélité=%.4f, amplitude=%.3e",
        best_factor, best_fidelity, best_amplitude,
    )

    return CalibrationResult(
        best_factor=best_factor,
        best_fidelity=best_fidelity,
        best_amplitude=best_amplitude,
        scan_factors=scan_factors,
        scan_fidelities=scan_fidelities,
    )


def calibrate_pi_pulse(
    simulate_pulse: Callable[[float], float],
    amplitude_rwa_pi: float,
    config: Optional[CalibrationConfig] = None,
) -> CalibrationResult:
    """
    Calibre spécifiquement un pulse pi.

    Wrapper autour de calibrate_pulse avec target_angle = pi.
    """
    if config is None:
        config = CalibrationConfig()
    config.target_angle = np.pi
    return calibrate_pulse(simulate_pulse, amplitude_rwa_pi, config)


def calibrate_half_pi_pulse(
    simulate_pulse: Callable[[float], float],
    amplitude_rwa_half_pi: float,
    config: Optional[CalibrationConfig] = None,
) -> CalibrationResult:
    """
    Calibre spécifiquement un pulse pi/2.

    Wrapper autour de calibrate_pulse avec target_angle = pi/2.
    """
    if config is None:
        config = CalibrationConfig()
    config.target_angle = np.pi / 2.0
    return calibrate_pulse(simulate_pulse, amplitude_rwa_half_pi, config)


# ----------------------------------------------------------------------
# Calibration adaptative (raffinement)
# ----------------------------------------------------------------------

def calibrate_pulse_adaptive(
    simulate_pulse: Callable[[float], float],
    amplitude_rwa: float,
    config: Optional[CalibrationConfig] = None,
    n_refinements: int = 3,
) -> CalibrationResult:
    """
    Calibration adaptative : scan grossier puis raffinements successifs.

    À chaque raffinement, la plage de scan est resserrée autour du
    meilleur facteur trouvé précédemment.

    n_refinements : nombre de raffinements (0 = scan simple)
    """
    if config is None:
        config = CalibrationConfig()

    # Scan initial
    result = calibrate_pulse(simulate_pulse, amplitude_rwa, config)
    best_factor = result.best_factor

    # Raffinements successifs
    for _ in range(n_refinements):
        half_width = (config.scan_max - config.scan_min) / (2.0 * config.n_points)
        new_min = max(1e-6, best_factor - half_width)
        new_max = min(2.0, best_factor + half_width)

        refined_config = CalibrationConfig(
            target_angle=config.target_angle,
            scan_min=new_min,
            scan_max=new_max,
            n_points=config.n_points,
            rwa_correction=config.rwa_correction,
        )
        result = calibrate_pulse(simulate_pulse, amplitude_rwa, refined_config)
        best_factor = result.best_factor

    LOGGER.info(
        "Calibration adaptative terminée. Facteur final : %.4f",
        best_factor,
    )
    return result


# ----------------------------------------------------------------------
# Intégration avec les modèles
# ----------------------------------------------------------------------

def calibrate_fluxonium_pi_pulse(
    fluxonium_model,
    config: Optional[CalibrationConfig] = None,
    adaptive: bool = False,
    n_refinements: int = 3,
) -> CalibrationResult:
    """
    Calibre un pulse pi pour un modèle fluxonium.

    fluxonium_model : instance de FluxoniumModel
    """
    amplitude_rwa = fluxonium_model.pulse_amplitude_rwa(np.pi)

    def simulate(amplitude: float) -> float:
        result = fluxonium_model.simulate_x_gate(amplitude)
        return result["fidelity"]

    if adaptive:
        return calibrate_pulse_adaptive(
            simulate, amplitude_rwa, config, n_refinements
        )
    return calibrate_pulse(simulate, amplitude_rwa, config)


def calibrate_kerr_cat_pi_pulse(
    kerr_cat_simulator,
    config: Optional[CalibrationConfig] = None,
    adaptive: bool = False,
    n_refinements: int = 3,
) -> CalibrationResult:
    """
    Calibre un pulse pi pour un simulateur Kerr-cat.

    kerr_cat_simulator : instance de KerrCatSimulator
    """
    # Pour Kerr-cat, l'amplitude RWA dépend du modèle spécifique.
    # Ici on utilise une amplitude de référence générique.
    amplitude_rwa = 1.0  # à adapter selon le modèle

    def simulate(amplitude: float) -> float:
        # À adapter : simulation d'un pulse sur le Kerr-cat
        # Pour l'instant, on retourne une fidélité proxy.
        # Cette fonction devra être connectée au simulateur réel.
        raise NotImplementedError(
            "La simulation de pulse Kerr-cat n'est pas encore connectée."
        )

    if adaptive:
        return calibrate_pulse_adaptive(
            simulate, amplitude_rwa, config, n_refinements
        )
    return calibrate_pulse(simulate, amplitude_rwa, config)


# ----------------------------------------------------------------------
# Tests unitaires
# ----------------------------------------------------------------------

def test_calibration_finds_optimum():
    """Vérifie que la calibration trouve le facteur optimal."""

    # Fonction de test : fidélité maximale à facteur = 0.5
    def simulate(amplitude: float) -> float:
        # amplitude_rwa = 1.0, donc facteur = amplitude
        factor = amplitude
        # Fidélité gaussienne centrée sur 0.5
        return float(np.exp(-((factor - 0.5) ** 2) / 0.01))

    config = CalibrationConfig(
        scan_min=0.3, scan_max=1.0, n_points=15
    )
    result = calibrate_pulse(simulate, 1.0, config)
    assert abs(result.best_factor - 0.5) < 0.05, (
        f"Facteur optimal incorrect : {result.best_factor}"
    )
    assert result.best_fidelity > 0.95, "Fidélité trop faible."


def test_adaptive_refines():
    """Vérifie que la calibration adaptative raffine le résultat."""

    def simulate(amplitude: float) -> float:
        factor = amplitude
        return float(np.exp(-((factor - 0.5) ** 2) / 0.001))

    config = CalibrationConfig(
        scan_min=0.3, scan_max=1.0, n_points=10
    )
    result = calibrate_pulse_adaptive(
        simulate, 1.0, config, n_refinements=2
    )
    assert abs(result.best_factor - 0.5) < 0.02, (
        f"Raffinement insuffisant : {result.best_factor}"
    )


def run_all_tests():
    """Lance tous les tests."""
    test_calibration_finds_optimum()
    test_adaptive_refines()
    LOGGER.info("Tous les tests sont passés.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Calibration de pulses quantiques."
    )
    parser.add_argument(
        "--target_angle", type=float, default=np.pi,
        help="Angle de rotation cible."
    )
    parser.add_argument(
        "--scan_min", type=float, default=0.3,
        help="Borne inférieure du scan."
    )
    parser.add_argument(
        "--scan_max", type=float, default=1.0,
        help="Borne supérieure du scan."
    )
    parser.add_argument(
        "--n_points", type=int, default=15,
        help="Nombre de points du scan."
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Active la calibration adaptative."
    )
    parser.add_argument(
        "--n_refinements", type=int, default=3,
        help="Nombre de raffinements adaptatifs."
    )
    parser.add_argument(
        "--output", type=str, default="results/calibration.json",
        help="Fichier de sortie."
    )
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.test:
        run_all_tests()
        return

    # Exemple avec une fonction de test
    def simulate_demo(amplitude: float) -> float:
        factor = amplitude
        return float(np.exp(-((factor - 0.5) ** 2) / 0.01))

    config = CalibrationConfig(
        target_angle=args.target_angle,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        n_points=args.n_points,
    )
    if args.adaptive:
        result = calibrate_pulse_adaptive(
            simulate_demo, 1.0, config, args.n_refinements
        )
    else:
        result = calibrate_pulse(simulate_demo, 1.0, config)

    result.save(Path(args.output))
    LOGGER.info(
        "Calibration terminée. Meilleur facteur : %.4f, fidélité : %.4f",
        result.best_factor, result.best_fidelity,
    )


if __name__ == "__main__":
    main()
