cat << 'EOF' > test_chirurgie_21q_pont.py
import math
import numpy as np

def simuler_pont_21q(n_qubits=21, tau_cycle_ns=800.0, gradient_flux=0.04, seed=42):
    rng = np.random.default_rng(seed)
    # Coordonnées normalisées des 21 noeuds (2 patchs de 9 + 3 ancillaires d'interface)
    x_coords = np.linspace(-1.0, 1.0, n_qubits)
    delta_phi = gradient_flux * np.exp(-0.5 * (x_coords / 0.6)**2) * np.sin(np.pi * x_coords)

    # Taux locaux T1_i et T_phi_i (en microsecondes) modulés par l'écart au point doux pi
    T1_i = 160.0 / (1.0 + 1.5 * (delta_phi**2))
    T_phi_i = 250.0 / (1.0 + 850.0 * (delta_phi**2))

    tau_us = tau_cycle_ns * 1e-3
    # Conversion en probabilités d'erreur Pauli par cycle (p_X, p_Z)
    p_X_i = 0.25 * (1.0 - np.exp(-tau_us / T1_i))
    p_Z_i = 0.5 * (1.0 - np.exp(-tau_us / T_phi_i)) + 0.25 * (1.0 - np.exp(-tau_us / T1_i))

    # Biais moyen de phase sur l'ensemble du réseau de 21 qubits
    biais_moyen = np.mean(p_Z_i / np.maximum(2.0 * p_X_i, 1e-12))

    # Simulation rapide de 20 000 cycles de chirurgie d=3 sur le sous-réseau d'interface
    n_shots = 20_000
    p_Z_interface = np.mean(p_Z_i[9:12])
    erreurs_interface = rng.random((n_shots, 3)) < p_Z_interface
    # Échec logique si >= 2 des 3 ancillaires d'interface basculent simultanément
    echec_logique = np.mean(np.sum(erreurs_interface, axis=1) >= 2)

    return {
        "T1_min_us": float(np.min(T1_i)),
        "T_phi_min_us": float(np.min(T_phi_i)),
        "p_X_moy": float(np.mean(p_X_i)),
        "p_Z_moy": float(np.mean(p_Z_i)),
        "biais_eta": float(biais_moyen),
        "p_echec_chirurgie": float(echec_logique)
    }

if __name__ == "__main__":
    r = simuler_pont_21q()
    print("--- Pont Microscopique -> Chirurgie 21 Qubits (d=3) ---")
    print(f"  Cohérence min réseau : T1_min = {r['T1_min_us']:.1f} us | T_phi_min = {r['T_phi_min_us']:.1f} us")
    print(f"  Erreurs Pauli / cycle: p_X = {r['p_X_moy']*100:.4f}% | p_Z = {r['p_Z_moy']*100:.4f}% (Biais eta = {r['biais_eta']:.1f})")
    print(f"  Échec logique d=3    : {r['p_echec_chirurgie']*100:.5f}%")
EOF
python test_chirurgie_21q_pont.py
