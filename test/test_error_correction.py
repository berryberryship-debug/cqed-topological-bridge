import numpy as np
from src.error_correction import BiasedNoiseChannel, PhaseFlipRepetitionCode


def test_repetition_code_suppresses_biased_errors():
    """Vérifie que l'augmentation de la distance d réduit l'erreur logique sous fort biais."""
    channel = BiasedNoiseChannel(p_total=0.03, bias_eta=500.0)
    code_d3 = PhaseFlipRepetitionCode(distance=3)
    code_d7 = PhaseFlipRepetitionCode(distance=7)

    p_L3 = code_d3.analytical_logical_error(channel)
    p_L7 = code_d7.analytical_logical_error(channel)

    assert p_L3 < channel.p_total
    assert p_L7 < p_L3


def test_monte_carlo_matches_analytical():
    """Vérifie la cohérence entre simulation Monte Carlo et formule combinatoire."""
    channel = BiasedNoiseChannel(p_total=0.04, bias_eta=1000.0)
    code = PhaseFlipRepetitionCode(distance=5)

    mc = code.run_monte_carlo(channel, n_shots=40_000, seed=123)
    exact = code.analytical_logical_error(channel)

    assert np.isclose(mc["p_logical_total"], exact, rtol=0.15)
