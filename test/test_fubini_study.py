cat << 'EOF' > test_fubini_study.py
import numpy as np
from scipy.linalg import eigh

def calculer_fubini_study_fluxonium(E_J=4.0, E_C=1.0, E_L=0.5, dphi_ext=1e-4):
    N = 350
    phi = np.linspace(-5 * np.pi, 5 * np.pi, N)
    h = phi[1] - phi[0]

    d2 = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        d2[i, i] = -30.0
        if i - 1 >= 0: d2[i, i - 1] = 16.0
        if i + 1 < N: d2[i, i + 1] = 16.0
        if i - 2 >= 0: d2[i, i - 2] = -1.0
        if i + 2 < N: d2[i, i + 2] = -1.0
    T = -4.0 * E_C * (d2 / (12.0 * h**2))

    def etat_fondamental(phi_ext):
        V = 0.5 * E_L * (phi**2) - E_J * np.cos(phi - phi_ext)
        _, vecs = eigh(T + np.diag(V), subset_by_index=[0, 1])
        psi0 = vecs[:, 0]
        psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * h)
        return psi0

    points_flux = [0.0, 0.5 * np.pi, 0.9 * np.pi, np.pi]
    resultats = []
    for f_ext in points_flux:
        psi_a = etat_fondamental(f_ext)
        psi_b = etat_fondamental(f_ext + dphi_ext)
        overlap = abs(np.sum(np.conj(psi_a) * psi_b) * h)
        g_pp = max(0.0, 1.0 - overlap**2) / (dphi_ext**2)
        resultats.append((f_ext / np.pi, g_pp))
    return resultats

if __name__ == "__main__":
    res = calculer_fubini_study_fluxonium()
    print("--- Métrique quantique de Fubini-Study g_{Phi,Phi} ---")
    for ratio_pi, g_val in res:
        print(f"  Phi_ext = {ratio_pi:4.2f} * pi  ->  g_PhiPhi = {g_val:.6f} rad^-2")
EOF
python test_fubini_study.py
