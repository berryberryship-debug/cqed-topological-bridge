import numpy as np
from src.fluxonium_model import (
    FluxoniumHardwareParams,
    FluxoniumHamiltonianSolver,
    DecoherenceAnalyzer
)
from src.pulse_calibration import FluxoniumPulseCalibrator, PulseConfig


def test_sweet_spot_first_derivative_vanishes():
    """Vérifie que df01/dPhi_ext = 0 au point doux Phi_ext = pi."""
    params = FluxoniumHardwareParams(E_J_GHz=4.0, E_C_GHz=1.0, E_L_GHz=0.5)
    solver = FluxoniumHamiltonianSolver(params, n_grid=400, n_states=4)
    sweep = solver.sweep_flux(n_points=101)
    mid = len(sweep["flux"]) // 2

    assert np.isclose(sweep["flux"][mid], np.pi)
    assert abs(sweep["df01_dphi"][mid]) < 1e-2


def test_giant_positive_anharmonicity():
    """Vérifie que l'anharmonicité alpha = f12 - f01 est positive et élevée à Phi_ext = pi."""
    params = FluxoniumHardwareParams(E_J_GHz=4.0, E_C_GHz=1.0, E_L_GHz=0.5)
    solver = FluxoniumHamiltonianSolver(params, n_grid=400, n_states=4)
    res = solver.solve(phi_ext=np.pi)

    f01 = res["energies"][1] - res["energies"][0]
    f12 = res["energies"][2] - res["energies"][1]
    alpha = f12 - f01

    assert f01 < 0.5
    assert alpha > 2.5


def test_lindblad_trace_preservation_and_gate_fidelity():
    """Vérifie la conservation de la probabilité totale et la fidélité de la porte X."""
    params = FluxoniumHardwareParams(E_J_GHz=4.0, E_C_GHz=1.0, E_L_GHz=0.5)
    solver = FluxoniumHamiltonianSolver(params, n_grid=350, n_states=4)
    res = solver.solve(phi_ext=np.pi)
    mats = solver.compute_matrix_elements(res["states"])

    cal = FluxoniumPulseCalibrator(res["energies"], mats["phi"], T1_us=200.0, T_phi_us=200.0, dim=4)
    sim = cal.run_gate(PulseConfig(t_gate_ns=20.0, n_steps=200))

    total_probs = np.sum(sim["populations"], axis=1)
    assert np.allclose(total_probs, 1.0, atol=1e-4)
    assert sim["fidelity_X"] > 0.99
    assert sim["max_leakage_P2"] < 1e-3
