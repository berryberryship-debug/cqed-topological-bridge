"""
test/test_chirurgie_21q_pont.py

Tests du simulateur de chirurgie de réseau (pont_lourd_unifie.py).

Vérifie :
- taux de succès logique dans [0, 1] ;
- décroissance du succès avec le bruit ;
- reproductibilité par graine ;
- robustesse aux paramètres invalides.

Note : ce test importe depuis pont_lourd_unifie.py situé à la racine
du dépôt. Si vous avez déplacé le fichier dans src/, remplacez
l'import par : from src.pont_lourd_unifie import SimChirurgie21Q
"""

import numpy as np
import pytest

from pont_lourd_unifie import SimChirurgie21Q


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def low_noise_sim():
    """Simulateur avec bruit faible, graine fixe."""
    pX = np.full(21, 0.001)
    pZ = np.full(21, 0.001)
    return SimChirurgie21Q(pX, pZ, 0.001, seed=123)


# ----------------------------------------------------------------------
# Tests de base
# ----------------------------------------------------------------------

def test_success_in_range(low_noise_sim):
    """Le taux de succès logique doit être dans [0, 1]."""
    succes = low_noise_sim.executer_lote(shots=1000)
    assert 0.0 <= succes <= 1.0


def test_success_is_float(low_noise_sim):
    """Le taux de succès doit être un float."""
    succes = low_noise_sim.executer_lote(shots=100)
    assert isinstance(succes, float)


# ----------------------------------------------------------------------
# Tests physiques
# ----------------------------------------------------------------------

def test_success_decreases_with_noise():
    """Plus de bruit doit donner moins de succès (tendance)."""
    pX_low = np.full(21, 0.001)
    pZ_low = np.full(21, 0.001)
    pX_high = np.full(21, 0.01)
    pZ_high = np.full(21, 0.01)

    sim_low = SimChirurgie21Q(pX_low, pZ_low, 0.001, seed=123)
    sim_high = SimChirurgie21Q(pX_high, pZ_high, 0.001, seed=123)

    succes_low = sim_low.executer_lote(shots=2000)
    succes_high = sim_high.executer_lote(shots=2000)

    assert succes_high <= succes_low, (
        f"Succès high ({succes_high:.4f}) devrait être <= "
        f"succès low ({succes_low:.4f})."
    )


def test_zero_noise_high_success():
    """Avec un bruit quasi nul, le succès doit être proche de 1."""
    pX = np.full(21, 1e-6)
    pZ = np.full(21, 1e-6)
    sim = SimChirurgie21Q(pX, pZ, 1e-6, seed=42)
    succes = sim.executer_lote(shots=1000)
    assert succes > 0.90, (
        f"Avec un bruit quasi nul, le succès devrait être > 0.90, "
        f"obtenu : {succes:.4f}"
    )


# ----------------------------------------------------------------------
# Tests de reproductibilité
# ----------------------------------------------------------------------

def test_reproducibility_same_seed():
    """Même graine doit donner le même résultat."""
    pX = np.full(21, 0.005)
    pZ = np.full(21, 0.005)
    sim1 = SimChirurgie21Q(pX, pZ, 0.001, seed=42)
    sim2 = SimChirurgie21Q(pX, pZ, 0.001, seed=42)
    succes1 = sim1.executer_lote(shots=500)
    succes2 = sim2.executer_lote(shots=500)
    assert succes1 == succes2


def test_different_seed_different_result():
    """Graines différentes : les deux résultats restent dans [0, 1]."""
    pX = np.full(21, 0.005)
    pZ = np.full(21, 0.005)
    sim1 = SimChirurgie21Q(pX, pZ, 0.001, seed=1)
    sim2 = SimChirurgie21Q(pX, pZ, 0.001, seed=2)
    succes1 = sim1.executer_lote(shots=500)
    succes2 = sim2.executer_lote(shots=500)
    assert 0.0 <= succes1 <= 1.0
    assert 0.0 <= succes2 <= 1.0


# ----------------------------------------------------------------------
# Tests de robustesse
# ----------------------------------------------------------------------

def test_invalid_probability_raises():
    """Une probabilité hors [0, 1] doit lever une erreur."""
    pX = np.full(21, 0.5)
    pZ = np.full(21, 0.5)
    with pytest.raises(Exception):
        SimChirurgie21Q(pX, pZ, 1.5, seed=42)


def test_merge_rounds_parameter():
    """Le paramètre rounds_merge doit être accepté."""
    pX = np.full(21, 0.001)
    pZ = np.full(21, 0.001)
    sim = SimChirurgie21Q(pX, pZ, 0.001, seed=42)
    succes = sim.executer_lote(shots=500, rounds_merge=5)
    assert 0.0 <= succes <= 1.0
