# Quantum-Lab — Fluxonium–Lattice Surgery Bridge (d=3, 21 Qubits) & Bosonic QEC

Pipeline multi-échelle paramétré par la physique cQED, connectant
les qubits supraconducteurs (**Fluxonium & Kerr-Cat**) à la
**chirurgie de réseau sur code de surface tourné (d=3, 21 qubits)**
et à la **correction d'erreurs quantique à bruit biaisé**.

## Statut scientifique (honnête)

Ce projet est un **pipeline multi-échelle**, pas encore une
dérivation physique de bout en bout validée expérimentalement.
La chaîne est :

Fluxonium (T1, T_phi, d f01/d Phi, g_FS)
    → noise_channel (canal AD + déphasage, Pauli twirl exact)
    → p_X, p_Y, p_Z
    → error_correction (budget d'erreur logique)
    → chirurgie de réseau (Monte-Carlo, code de surface d=3)

L'interface géométrie → p_Z est fermée par un **Pauli twirling
exact** (via la matrice de transfert de Pauli), pas par une
approximation postulée. Le couplage géométrique
(`fs_coupling_gamma`) reste **phénoménologique** et est
clairement étiqueté comme tel.

**Ce qui n'est PAS implémenté (et ne doit pas être annoncé) :**
- Contrôle DRAG (non implémenté)
- Fonction de Wigner (non implémentée)
- Lattice surgery stabilizer-level complet (prototype seulement)
- Validation expérimentale

## 1. Architecture du pipeline

1. **Modulation spatiale du substrat** : gradient spatial de flux
   delta_Phi_i pondéré par une enveloppe gaussienne autour du
   point doux Phi_ext = pi.

2. **Noyau dynamique conservatif** : bilan de masse fermé
   (résidu <= 1e-10) régulant la densité effective de
   quasiparticules hors équilibre (x_qp).

3. **Solveur Hamiltonien Fluxonium microscopique** :
   diagonalise l'Hamiltonien en base de phase (Dirichlet,
   stencil central) :
   H = 4 EC n² + 0.5 EL (phi - phi_ext)² - EJ cos(phi)
   Extrait T1, T_phi, QGT, métrique de Fubini-Study
   (différence symétrique, convergence validée).

4. **Canal de bruit & Pauli twirling exact** :
   convertit (T1, T_phi, tau) en canal de Pauli exact via la
   matrice de transfert de Pauli (PTM).
   Le twirl de l'amplitude damping donne p_X = p_Y = gamma/4,
   p_Z = gamma/2 ; le déphasage pur donne p_Z = lambda_z.
   La composition est exacte (produit des diagonales du PTM).

5. **Qubit bosonique Kerr-Cat** : stabilisation d'états chats
   sous pompage à deux photons, équation maîtresse de Lindblad,
   intégration RK4, fidélité logique et leakage.

6. **Chirurgie de réseau & QEC à bruit biaisé** :
   convertit les taux locaux (T1_i, T_phi_i) en canal de Pauli
   asymétrique (p_X_i, p_Z_i) sur la durée de cycle tau_cycle,
   et exécute le protocole de chirurgie de réseau
   Init → Merge (d rounds) → Split → Measure Z_L1
   avec décodage par syndrome de poids 1 (d=3, 21 qubits),
   ainsi que les codes de répétition de phase à distance impaire d.

## 2. Structure du dépôt (réelle)
