# Fluxonium–Lattice Surgery Bridge ($d=3$, 21 Qubits)

Coupled physical-to-topological simulation scaffold connecting superconducting circuit quantum electrodynamics (**cQED — Fluxonium**) to **fault-tolerant rotated surface code lattice surgery** ($d=3$).

---

## 1. Architecture du Pipeline

Le modèle relie quatre couches physiques et mathématiques sur un réseau replié en zigzag de **21 qubits supraconducteurs** (2 patchs logiques $L_1, L_2$ de 9 qubits de données + 3 qubits ancillaires d'interface) :

1. **Modulation spatiale du substrat (`coordonnees_reseau_21q` & `modulation_moire_substrat`) :**
   Modélise un gradient spatial de flux $\delta\Phi_i$ pondéré par une enveloppe gaussienne autour du point doux $\Phi_{\text{ext}} = \pi$ sur la géométrie millimétrique du processeur.
2. **Noyau dynamique conservatif (`evoluer_noyau_conservatif`) :**
   Maintient un bilan de masse strictement fermé ($r \le 10^{-15}$) entre les impulsions de contrôle et la relaxation pour fixer la densité effective de quasiparticules hors équilibre ($x_{\text{qp}}$).
3. **Solveur Hamiltonien Fluxonium microscopique (`SolveurFluxonium`) :**
   Diagonalise exactement ($N_{\text{basis}} = 35$) l'Hamiltonien individuel de chacun des 21 nœuds :
   $$H_i = 4E_C\hat{n}^2 + \frac{1}{2}E_L\hat{\phi}^2 - E_{J,i}\cos(\hat{\phi} - \Phi_i)$$
   Extrait les temps de relaxation diélectrique et quasiparticules ($T_{1,i}$), le déphasage pur de flux $1/f$ ($T_{\phi,i}$), ainsi que la métrique locale de Fubini–Study :
   $$g_{\Phi\Phi,i} = \frac{1 - |\langle \psi_0(\Phi_i) | \psi_0(\Phi_i + d\Phi) \rangle|^2}{d\Phi^2}$$
4. **Chirurgie de réseau hétérogène (`SimChirurgie21Q`) :**
   Convertit les taux locaux $(T_{1,i}, T_{\phi,i})$ en canal de Pauli asymétrique $(p_{X,i}, p_{Z,i})$ sur la durée de cycle $\tau_{\text{cycle}}$ et exécute le protocole complet *Init $\to$ Merge ($d$ rounds) $\to$ Split $\to$ Measure $Z_{L1}$ $\to$ Pauli Frame* avec décodage par syndrome de poids minimal.

---

## 2. Installation & Exécution

### Dépendances
Uniquement **Python 3.10+** et **NumPy** :

```bash
pip install numpy
