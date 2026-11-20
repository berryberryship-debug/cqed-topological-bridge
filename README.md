# Quantum-Lab — Fluxonium–Lattice Surgery Bridge ($d=3$, 21 Qubits) & Bosonic QEC

Coupled physical-to-topological simulation suite connecting superconducting circuit quantum electrodynamics (**cQED — Fluxonium & Kerr-Cat**) to **fault-tolerant rotated surface code lattice surgery** ($d=3$) and biased-noise quantum error correction.

---

## 1. Architecture du Pipeline

Le modèle relie le niveau matériel microscopique aux protocoles de correction d'erreurs topologiques sur un réseau replié en zigzag de **21 qubits supraconducteurs** (2 patchs logiques $L_1, L_2$ de 9 qubits de données + 3 qubits ancillaires d'interface), tout en intégrant la dynamique de fluxons, le contrôle DRAG/Lindblad et les architectures bosoniques Kerr-Cat :

1. **Modulation spatiale du substrat (`coordonnees_reseau_21q` & `modulation_moire_substrat`) :**
   Modélise un gradient spatial de flux $\delta\Phi_i$ pondéré par une enveloppe gaussienne autour du point doux $\Phi_{\text{ext}} = \pi$ sur la géométrie millimétrique du processeur.
2. **Noyau dynamique conservatif (`evoluer_noyau_conservatif` — `fluxon_engine.py`) :**
   Maintient un bilan de masse strictement fermé ($r \le 10^{-15}$) entre les impulsions de contrôle, la propagation de fluxons et la relaxation pour fixer la densité effective de quasiparticules hors équilibre ($x_{\text{qp}}$).
3. **Solveur Hamiltonien Fluxonium microscopique (`SolveurFluxonium` / `fluxonium_model.py`) :**
   Diagonalise l'Hamiltonien individuel des nœuds supraconducteurs (base de Fock $N_{\text{basis}} = 35$ ou différences finies d'ordre 4 en phase $\hat{\phi}$) :
   $$H_i = 4E_C(\hat{n} - n_g)^2 + \frac{1}{2}E_L\hat{\phi}^2 - E_{J,i}\cos(\hat{\phi} - \Phi_i)$$
   Extrait les temps de relaxation diélectrique, inductive et par quasiparticules ($T_{1,i}$), le déphasage pur de flux $1/f$ ($T_{\phi,i}$), la suppression de dispersion de charge $n_g$, ainsi que la métrique locale de Fubini–Study :
   $$g_{\Phi\Phi,i} = \frac{1 - \vert{}\langle \psi_0(\Phi_i) \vert{} \psi_0(\Phi_i + d\Phi) \rangle\vert{}^2}{d\Phi^2}$$
4. **Contrôle micro-onde & Équation maîtresse de Lindblad (`pulse_calibration.py`) :**
   Calibre les portes quantiques ($X$, $\sqrt{X}$) par enveloppes gaussiennes et correction **DRAG**, exploitant l'anharmonicité positive géante ($\alpha = f_{12} - f_{01} > 0$) au point doux $\Phi_{\text{ext}} = \pi$ pour supprimer la fuite vers $\vert{}2\rangle$.
5. **Qubit bosonique Kerr-Cat (`kerr_cat_simulator.py`) :**
   Modélise la stabilisation d'états chats de Schrödinger $\vert{}\mathcal{C}_\alpha^\pm\rangle \propto \vert{}\alpha\rangle \pm \vert{}-\alpha\rangle$ sous pompage à deux photons ($\hat{H} = -K\hat{a}^{\dagger 2}\hat{a}^2 + \varepsilon_2(\hat{a}^{\dagger 2} + \hat{a}^2)$), la fonction de Wigner $W(x, p)$ et le biais exponentiel du bruit ($T_X \gg T_Z$).
6. **Chirurgie de réseau hétérogène & QEC à bruit biaisé (`SimChirurgie21Q` & `error_correction.py`) :**
   Convertit les taux locaux $(T_{1,i}, T_{\phi,i})$ en canal de Pauli asymétrique $(p_{X,i}, p_{Z,i})$ sur la durée de cycle $\tau_{\text{cycle}}$ et exécute :
   * Le protocole complet de chirurgie de réseau *Init $\to$ Merge ($d$ rounds) $\to$ Split $\to$ Measure $Z_{L1}$ $\to$ Pauli Frame* avec décodage par syndrome de poids minimal ($d=3$, 21 qubits).
   * Les codes de répétition de phase à distance impaire $d$ optimisés pour les architectures à fort biais $\eta = p_Z / (p_X + p_Y)$.

---

## 2. Structure du Dépôt

```text
Quantum-lab/
├── src/
│   ├── __init__.py               # Exports unifiés du package
│   ├── fluxon_engine.py          # Noyau conservatif (r <= 1e-15) & dynamique des fluxons
│   ├── fluxonium_model.py        # Solveur Hamiltonien, métrique Fubini-Study & T1/T_phi/T2
│   ├── pulse_calibration.py      # Équation maîtresse de Lindblad & optimisation DRAG
│   ├── kerr_cat_simulator.py     # Oscillateur Kerr-Cat, fonction de Wigner & biais de bruit
│   └── error_correction.py       # Chirurgie de réseau 21Q & QEC Pauli biaisé
├── tests/
│   ├── test_fluxon.py
│   ├── test_fluxonium.py
│   ├── test_kerr_cat.py
│   └── test_error_correction.py
├── notebooks/
│   └── fluxonium_dashboard.ipynb # Tableau de bord interactif multi-panneaux
├── results/                      # Graphiques et rapports générés
├── README.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
