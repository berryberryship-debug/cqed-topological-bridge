
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intégration Fluxonium -> Noise Channel.

Ce script suppose que les API appelées existent
dans les modules correspondants.
"""

import numpy as np

from fluxonium_hamiltonian import (
    FluxoniumHamiltonian,
    FluxoniumParams,
)

from noise_channel import (
    ChannelConfig,
    build_pauli_channel,
    effective_error_rate,
)


def main():

    # --------------------------------------------------------
    # 1. Paramètres physiques
    # --------------------------------------------------------

    params = FluxoniumParams(
        EJ=4.0,
        EC=1.0,
        EL=0.5,
        phi_ext=np.pi,
    )

    # --------------------------------------------------------
    # 2. Modèle Fluxonium
    # --------------------------------------------------------

    model = FluxoniumHamiltonian(params)

    # --------------------------------------------------------
    # 3. Taux de cohérence
    # --------------------------------------------------------

    rates = model.coherence_rates()

    required_rates = ("T1_s", "T_phi_s")

    for key in required_rates:
        if key not in rates:
            raise KeyError(
                f"Taux de cohérence manquant : {key}"
            )

        if not np.isfinite(rates[key]) or rates[key] <= 0:
            raise ValueError(
                f"Valeur invalide pour {key}: {rates[key]}"
            )

    # --------------------------------------------------------
    # 4. Sensibilité au flux
    # --------------------------------------------------------

    df01_dphi = model.df01_dphi()

    if not np.isfinite(df01_dphi):
        raise ValueError(
            "df01_dphi n'est pas fini."
        )

    # --------------------------------------------------------
    # 5. Configuration du canal
    # --------------------------------------------------------

    config = ChannelConfig(
        T1=rates["T1_s"],
        T_phi=rates["T_phi_s"],
        tau=20e-9,
        df01_dphi=df01_dphi,
        S_phi=2e-6,
    )

    # --------------------------------------------------------
    # 6. Construction du canal de Pauli
    # --------------------------------------------------------

    channel = build_pauli_channel(config)

    pX = float(channel["X"])
    pY = float(channel["Y"])
    pZ = float(channel["Z"])

    # --------------------------------------------------------
    # 7. Vérifications
    # --------------------------------------------------------

    for name, probability in (
        ("pX", pX),
        ("pY", pY),
        ("pZ", pZ),
    ):
        if not np.isfinite(probability):
            raise ValueError(
                f"{name} n'est pas fini."
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{name} hors de [0,1] : {probability}"
            )

    print("=== FLUXONIUM / NOISE CHANNEL ===")
    print(f"T1       : {rates['T1_s']:.6e} s")
    print(f"T_phi    : {rates['T_phi_s']:.6e} s")
    print(f"df01/dphi: {df01_dphi:.6e}")
    print(f"pX       : {pX:.8e}")
    print(f"pY       : {pY:.8e}")
    print(f"pZ       : {pZ:.8e}")

    print(
        "Erreur effective :",
        effective_error_rate(channel),
    )


if __name__ == "__main__":
    main()
