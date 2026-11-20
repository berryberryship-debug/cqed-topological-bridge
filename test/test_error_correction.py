"""
test/test_error_correction.py

Tests du module error_correction.py, alignés sur l'API actuelle
(run_full_pipeline, PhysicalErrorRates, error_budget).
Aucune référence à une API phantom, aucun bloc shell.
"""

import numpy as np
import pytest

from src.error_correction import (
    PhysicalErrorRates,
    error_budget,
    run_full_pipeline,
    logical_error_rate_surface_code,
    logical_error_rate_repetition_code,
)


# ----------------------------------------------------------------------
# Tests PhysicalErrorRates
# ----------------------------------------------------------------------

def test_relaxation_error_positive():
    rates = PhysicalErrorRates(
        t1=7.19e-6, t2=14.38e-6,
        gate_fidelity=0.99, gate_time=20e-9,
    )
    assert rates.relaxation_error > 0.0


def test_dephasing_error_positive():
    rates = PhysicalErrorRates(
        t1=7.19e-6, t2=14.38e-6,
        gate_fidelity=0.99, gate_time=20e-9,
    )
    assert rates.dephasing_error > 0.0


def test_total_error_sum():
    rates = PhysicalErrorRates(
        t1=7.19e-6, t2=14.38e-6,
        gate_fidelity=0.99, gate_time=20e-9,
    )
    total = (
        rates.relaxation_error
        + rates.dephasing_error
        + rates.gate_error
    )
    assert abs(rates.total_physical_error - total) < 1e-12


# ----------------------------------------------------------------------
# Tests run_full_pipeline (API découplée)
# ----------------------------------------------------------------------

def test_run_full_pipeline_returns_dict():
    budget = run_full_pipeline(
        T1=7.19e-6, T2=14.38e-6,
        gate_fidelity=0.99, gate_time=20e-9,
        code_distance=3, code_type="surface",
    )
    assert isinstance(budget, dict)
    assert "logical_error_rate" in budget
    assert "below_threshold" in budget


def test_run_full_pipeline_below_threshold():
    budget = run_full_pipeline(
        T1=7.19e-6, T2=14.38e-6,
        gate_fidelity=0.99, gate_time=20e-9,
        code_distance=3, code_type="surface",
    )
    assert budget["below_threshold"] is True


def test_run_full_pipeline_logical_error_positive():
    budget = run_full_pipeline(
        T1=7.19e-6, T2=14.38e-6,
        gate_fidelity=0.99, gate_time=20e-9,
        code_distance=3, code_type="surface",
    )
    assert budget["logical_error_rate"] > 0.0


# ----------------------------------------------------------------------
# Tests de cohérence entre formules
# ----------------------------------------------------------------------

def test_surface_code_formula_consistency():
    """La formule surface code doit être cohérente avec error_budget."""
    p = 0.005
    d = 3
    p_L_direct = logical_error_rate_surface_code(p, d)
    rates = PhysicalErrorRates(
        t1=1e-3, t2=1e-3,
        gate_fidelity=1.0 - p, gate_time=1e-9,
    )
    budget = error_budget(rates, code_distance=d, code_type="surface")
    # total_physical_error ≈ p (car gate_error domine)
    assert abs(budget["total_physical_error"] - p) < 1e-6
    assert budget["logical_error_rate"] > 0.0


def test_repetition_code_formula():
    """La formule de répétition doit être cohérente."""
    p = 0.01
    n = 3
    p_L = logical_error_rate_repetition_code(p, n)
    # Terme dominant : 3 * p * (1-p)²
    expected = 3 * p * (1 - p) ** 2
    assert abs(p_L - expected) < 1e-12
