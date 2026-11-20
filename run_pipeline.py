#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construction du canal de Pauli effectif pour un qubit fluxonium."""

from __future__ import annotations

import math
from fluxonium_hamiltonian import FluxoniumHamiltonian, FluxoniumParams
from noise_channel import ChannelConfig, build_pauli_channel, effective_error_rate

# --- Constantes du protocole ---------------------------------------------
GATE_TIME_S: float = 20e-9          # durée de porte (s)
FLUX_NOISE_S_PHI: float = 2e-6      # densité spectrale de bruit de flux (φ₀²/Hz)


def build_default_params() -> FluxoniumParams:
    """Construit les paramètres Fluxonium au point doux Φ_ext = π."""
    try:
        return FluxoniumParams(
            EC=1.0,
            EL=0.5,
            EJ=5.0,
            phi_ext=math.pi,
        )
    except TypeError:
        # Repli si FluxoniumParams possède déjà des valeurs par défaut complètes
        return FluxoniumParams()


def run() -> dict[str, float]:
    """Exécute la chaîne Fluxonium -> PTM Pauli Twirl -> Budget d'erreur."""
    # --- 1. Modèle physique ----------------------------------------------
    params = build_default_params()
    model = FluxoniumHamiltonian(params)

    # --- 2. Taux de cohérence --------------------------------------------
    rates = model.coherence_rates()
    T1_s = float(rates["T1_s"])
    T_phi_s = float(rates["T_phi_s"])

    # --- 3. Sensibilité df01/dφ & géométrie (si exposée) -----------------
    df01_dphi = float(model.df01_dphi())
    g_fs = float(model.fubini_study_metric()) if hasattr(model, "fubini_study_metric") else None

    # --- 4. Configuration du canal ---------------------------------------
    config = ChannelConfig(
        T1=T1_s,
        T_phi=T_phi_s,
        tau=GATE_TIME_S,
        df01_dphi=df01_dphi,
        S_phi=FLUX_NOISE_S_PHI,
    )

    # --- 5. Canal de Pauli exact (PTM) -----------------------------------
    channel = build_pauli_channel(config)
    pX, pY, pZ = float(channel["X"]), float(channel["Y"]), float(channel["Z"])

    # --- 6. Budget d'erreur ----------------------------------------------
    eps = float(effective_error_rate(config))

    print(f"T1 = {T1_s * 1e6:.2f} µs | T_phi = {T_phi_s * 1e6:.2f} µs")
    if g_fs is not None:
        print(f"df01/dφ = {df01_dphi:.3e} Hz/φ₀ | g_FS = {g_fs:.4f} | erreur/porte ≈ {eps:.2e}")
    else:
        print(f"df01/dφ = {df01_dphi:.3e} Hz/φ₀ | erreur/porte ≈ {eps:.2e}")
    print(f"pX = {pX:.2e} | pY = {pY:.2e} | pZ = {pZ:.2e}")

    return {
        "T1_s": T1_s,
        "T_phi_s": T_phi_s,
        "df01_dphi": df01_dphi,
        "pX": pX,
        "pY": pY,
        "pZ": pZ,
        "eps": eps,
    }


if __name__ == "__main__":
    run()
