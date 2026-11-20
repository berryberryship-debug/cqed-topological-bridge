import numpy as np
from src.kerr_cat_simulator import KerrCatParams, KerrCatSimulator


def test_kerr_cat_degeneracy_and_parities():
    """Vérifie la quasi-dégénérescence et les parités opposées (+1 / -1) des états chats."""
    params = KerrCatParams(K_MHz=6.5, eps2_MHz=26.0, n_fock=45)
    sim = KerrCatSimulator(params)
    spec = sim.solve_spectrum(n_states=4)

    assert spec["tunnel_splitting_MHz"] < 1e-2
    assert spec["excited_gap_MHz"] > 20.0
    parities_sorted = np.sort(spec["parities"][:2])
    assert np.allclose(parities_sorted, [-1.0, 1.0], atol=1e-2)


def test_kerr_cat_exponential_noise_bias():
    """Vérifie que le ratio T_X / T_Z dépasse 100 pour |alpha|^2 = 4."""
    params = KerrCatParams(K_MHz=6.5, eps2_MHz=26.0, kappa_1_kHz=2.5, n_fock=40)
    sim = KerrCatSimulator(params)
    bias_res = sim.simulate_noise_bias(t_max_us=10.0, n_steps=30)

    assert bias_res["noise_bias_ratio"] > 100.0
    assert bias_res["T_X_bit_us"] > bias_res["T_Z_phase_us"]
