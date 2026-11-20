from fluxonium_hamiltonian import FluxoniumHamiltonian, FluxoniumParams
from noise_channel import ChannelConfig, build_pauli_channel

# 1. Modèle physique
params  = FluxoniumParams(...)
model   = FluxoniumHamiltonian(params)

# 2. Taux de cohérence (T1, T_phi)
rates = model.coherence_rates()
T1_s    = rates["T1_s"]
T_phi_s = rates["T_phi_s"]

# 3. Sensibilité df01/dφ — via une méthode publique
#    (à exposer côté FluxoniumHamiltonian : def df01_dphi(self) -> float)
df01_dphi = model.df01_dphi()

# 4. Configuration du canal
config = ChannelConfig(
    T1=T1_s,
    T_phi=T_phi_s,
    tau=20e-9,
    df01_dphi=df01_dphi,
    S_phi=2e-6,
)

# 5. Canal de Pauli
channel = build_pauli_channel(config)
pX, pY, pZ = channel["X"], channel["Y"], channel["Z"]
