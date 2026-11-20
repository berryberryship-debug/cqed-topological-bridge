"""Construction du canal de Pauli effectif pour un qubit fluxonium."""

from __future__ import annotations

from fluxonium_hamiltonian import FluxoniumHamiltonian, FluxoniumParams
from noise_channel import ChannelConfig, build_pauli_channel, effective_error_rate

# --- Constantes du protocole ---------------------------------------------
GATE_TIME_S: float = 20e-9          # durée de porte (s)
FLUX_NOISE_S_PHI: float = 2e-6      # densité spectrale de bruit de flux (φ₀²/Hz)

# --- 1. Modèle physique --------------------------------------------------
params = FluxoniumParams(...)       # à compléter avec les paramètres réels
model  = FluxoniumHamiltonian(params)

# --- 2. Taux de cohérence -----------------------------------------------
rates = model.coherence_rates()
T1_s    = float(rates["T1_s"])
T_phi_s = float(rates["T_phi_s"])

# --- 3. Sensibilité df01/dφ (API publique) ------------------------------
#    À exposer côté FluxoniumHamiltonian :
#        def df01_dphi(self) -> float:
#            return self._df01_dphi()
df01_dphi = model.df01_dphi()

# --- 4. Configuration du canal ------------------------------------------
config = ChannelConfig(
    T1=T1_s,
    T_phi=T_phi_s,
    tau=GATE_TIME_S,
    df01_dphi=df01_dphi,
    S_phi=FLUX_NOISE_S_PHI,
)

# --- 5. Canal de Pauli ---------------------------------------------------
channel = build_pauli_channel(config)
pX, pY, pZ = channel["X"], channel["Y"], channel["Z"]

# --- 6. Budget d'erreur (facultatif mais utile) -------------------------
eps = effective_error_rate(config)
print(f"T1 = {T1_s*1e6:.2f} µs | T_phi = {T_phi_s*1e6:.2f} µs")
print(f"df01/dφ = {df01_dphi:.3e} Hz/φ₀ | erreur/porte ≈ {eps:.2e}")
print(f"pX = {pX:.2e} | pY = {pY:.2e} | pZ = {pZ:.2e}")
