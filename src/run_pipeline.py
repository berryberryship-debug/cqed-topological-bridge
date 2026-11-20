from fluxonium_hamiltonian import FluxoniumHamiltonian, FluxoniumParams
from noise_channel import ChannelConfig, build_pauli_channel, effective_error_rate

model = FluxoniumHamiltonian(FluxoniumParams(...))
rates = model.coherence_rates()

config = ChannelConfig(
    T1=rates["T1_s"],
    T_phi=rates["T_phi_s"],
    tau=20e-9,
    df01_dphi=model._df01_dphi(),  # ou exposé via une méthode publique
    S_phi=2e-6,
)
channel = build_pauli_channel(config)
pX, pY, pZ = channel["X"], channel["Y"], channel["Z"]
