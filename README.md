Quantum-Lab — Fluxonium–Lattice Surgery Bridge
Pipeline multi-échelle : Fluxonium, bruit, QEC biaisée, Kerr-cat et chirurgie de réseau d=3
Ce dépôt implémente un pipeline numérique multi-échelle reliant la modélisation de qubits supraconducteurs Fluxonium et Kerr-cat à des canaux de bruit, à la correction d’erreurs quantique (QEC) à bruit biaisé, et à un prototype de chirurgie de réseau sur code de surface tourné de distance d=3, utilisant 21 qubits.
Le projet connecte plusieurs niveaux de description :
Substrat et quasiparticules
│
▼
Fluxonium : T1, T_phi, df01/dPhi, g_FS
│
▼
Canal de bruit : amplitude damping + déphasage
│
▼
Pauli twirling : pX, pY, pZ
│
▼
Budget d’erreur et QEC biaisée
│
▼
Chirurgie de réseau : code de surface tourné d=3
Ce dépôt est un pipeline de modélisation numérique multi-échelle. Il n’est pas une démonstration expérimentale, ni une dérivation physique complète de bout en bout.
1. Statut scientifique et périmètre épistémique
Le pipeline sépare explicitement :
les quantités calculées à partir d’un modèle physique ;
les paramètres phénoménologiques ;
les approximations de réduction ;
les prototypes de correction d’erreurs ;
les éléments non validés expérimentalement.
Il ne faut pas confondre :
un modèle numérique ;
une prédiction expérimentale ;
une calibration matérielle ;
une démonstration de seuil de QEC ;
une implémentation complète de correction d’erreurs ;
une architecture physique entièrement dérivée.
Le pipeline est conçu pour étudier la cohérence d’une chaîne de modélisation, pas pour affirmer qu’une architecture matérielle fonctionnera telle quelle.
1.1 Ce qui est implémenté
Hamiltonien Fluxonium en base de phase.
Diagonalisation numérique du spectre.
Observables T1, T_phi, df01/dPhi.
Quantum geometric tensor (QGT).
Métrique de Fubini–Study g_FS.
Modèle de quasiparticules froides pour la petite jonction.
Canal d’amplitude damping et déphasage pur.
Pauli twirling via matrice de transfert de Pauli.
Canal de Pauli effectif pX, pY, pZ.
Qubit bosonique Kerr-cat.
Équation maîtresse de Lindblad et intégration RK4.
Codes de répétition de phase.
Budget d’erreur logique.
Prototype de chirurgie de réseau sur code de surface tourné d=3.
Décodeur MWPM pondéré par la géométrie et le biais local.
1.2 Ce qui n’est pas implémenté
Contrôle DRAG.
Fonction de Wigner.
Dynamique microscopique complète des quasiparticules.
Quasiparticules dans le réseau inductif.
Asymétrie de gap supraconducteur.
Couplage physique complet entre quatre Fluxonium.
Chirurgie de réseau complète au niveau des stabilisateurs.
Validation expérimentale.
1.3 Ce qui reste phénoménologique
Le couplage géométrique fs_coupling_gamma.
La réduction des qubits Fluxonium à des qubits à deux niveaux.
La conversion des taux continus en probabilités de Pauli par cycle.
Le traitement du bruit quasi statique comme un canal effectif.
Le couplage XX entre qubits, lorsqu’il est utilisé comme entrée.
Les paramètres de modulation spatiale du substrat.
2. Architecture du pipeline
2.1 Vue d’ensemble
Le pipeline suit la chaîne suivante :
Substrat et quasiparticules
│
▼
Fluxonium : T1, T_phi, df01/dPhi, g_FS
│
▼
Canal de bruit : amplitude damping + déphasage
│
▼
Pauli twirling : pX, pY, pZ
│
▼
Budget d’erreur et QEC biaisée
│
▼
Chirurgie de réseau : code de surface tourné d=3
Chaque étape produit des paramètres ou des observables utilisés par l’étape suivante.
2.2 Modulation spatiale du substrat
Le module fluxon_engine.py décrit une modulation spatiale du substrat.
Chaque qubit i reçoit un décalage de flux :
ΔΦ_i.
Ce décalage est pondéré par une enveloppe gaussienne centrée autour du point doux :
Φ_ext = π.
La modulation spatiale permet de générer des paramètres locaux différents pour chaque qubit :
ΔΦ_i
→ Φ_ext,i
→ f01,i
→ T1_i
→ T_phi_i
→ pX_i, pY_i, pZ_i
Cette étape ne prétend pas décrire une fabrication réelle. Elle fournit un cadre paramétrique pour étudier l’effet d’une inhomogénéité spatiale sur les taux de bruit locaux.
2.3 Noyau dynamique conservatif
Le module fluxon_engine.py maintient un bilan effectif de densité de quasiparticules hors équilibre :
x_qp.
Le bilan est conçu comme conservatif :
x_qp(t + dt) = x_qp(t) + production - pertes
avec une contrainte numérique :
|résidu de bilan| <= 1e-10
Le bilan permet de réguler la densité effective de quasiparticules utilisée par la couche Fluxonium.
Important : ce bilan ne décrit pas une dynamique microscopique complète des quasiparticules dans un circuit supraconducteur réel. Il ne modélise notamment pas :
la diffusion des quasiparticules ;
le piégeage dans les défauts ;
la recombinaison ;
les effets de bord ;
les quasiparticules dans le réseau inductif ;
les asymétries de gap supraconducteur.
3. Solveur Hamiltonien Fluxonium
3.1 Hamiltonien
Le module fluxonium_hamiltonian.py construit et diagonalise l’Hamiltonien du Fluxonium en base de phase.
Les conditions aux limites sont de type Dirichlet. La discrétisation utilise un stencil central.
L’Hamiltonien est :
H = 4 EC n² + 0.5 EL (phi - phi_ext)² - EJ cos(phi).
Les énergies sont exprimées en unités de h, c’est-à-dire en GHz.
3.2 Quantités extraites
Le solveur extrait notamment :
les niveaux d’énergie ;
les états propres ;
les fréquences de transition ;
les éléments de matrice ;
les dipôles de charge et de phase ;
les taux de relaxation ;
les taux de déphasage ;
le quantum geometric tensor ;
la métrique de Fubini–Study.
3.3 Convergence
La convergence est vérifiée par raffinement de grille et par différences symétriques.
Les tests comparent notamment :
les fréquences de transition ;
les éléments de matrice ;
les dérivées de fréquence ;
la métrique de Fubini–Study ;
la stabilité des états propres.
La convergence locale ne garantit pas une convergence globale sur tout l’espace des paramètres.
4. Observables du Fluxonium
4.1 Observables principales
Le module fluxonium_model.py calcule les observables physiques utilisées par les couches supérieures du pipeline.
4.2 Sensibilité au flux
La dérivée de fréquence est utilisée pour relier une fluctuation de flux à une fluctuation de fréquence :
δf01 = (df01/dΦ) δΦ.
Au premier ordre, une fluctuation de flux produit donc une fluctuation de fréquence.
Au second ordre, la courbure de la fréquence contribue également :
δf01 = (df01/dΦ) δΦ + 0.5 (d²f01/dΦ²) δΦ².
4.3 Métrique de Fubini–Study
La métrique de Fubini–Study mesure la sensibilité géométrique de l’état fondamental aux variations du paramètre de flux. Elle est calculée à partir du quantum geometric tensor.
Important : la métrique de Fubini–Study ne détermine pas à elle seule un couplage physique entre qubits. Le couplage géométrique fs_coupling_gamma reste un paramètre phénoménologique.
5. Quasiparticules froides
5.1 Modèle implémenté
Le modèle de relaxation par quasiparticules implémenté concerne la petite jonction Josephson, dans l’approximation des quasiparticules froides.
Le taux descendant est de la forme :
Gamma_qp_down = (8/pi) (EJ/hbar) x_qp sqrt(2 Delta/(h f01)) |<0|sin((phi - phi_ext)/2)|1>|².
5.2 Hypothèses
Les hypothèses sont :
quasiparticules froides ;
énergie typique des quasiparticules très inférieure à h f01 ;
h f01 << Delta ;
petite jonction uniquement ;
aucun taux montant QP ;
aucune contribution QP du réseau inductif ;
aucune asymétrie de gap supraconducteur ;
aucun piègeage ou recombinaison microscopique.
5.3 Portée du modèle
Le taux QP calculé est donc un taux de la petite jonction, et non le taux QP total d’un Fluxonium réel. Il ne doit pas être interprété comme une description complète des quasiparticules dans un circuit Fluxonium.
6. Canal de bruit et Pauli twirling
6.1 Canal de bruit
Le module noise_channel.py convertit les taux locaux en un canal de Pauli effectif sur une durée tau.
Il combine :
l’amplitude damping ;
le déphasage pur ;
une composition via la matrice de transfert de Pauli (PTM).
6.2 Amplitude damping
Pour un canal d’amplitude damping de paramètre gamma, le twirling de Pauli donne :
pX = pY = gamma/4,
pZ = gamma/2.
6.3 Déphasage pur
Pour un déphasage pur de paramètre lambda_Z, le twirling donne :
pZ = lambda_Z.
6.4 Composition des canaux
La composition des canaux est réalisée par produit des diagonales de leurs matrices de transfert de Pauli.
Le canal de Pauli effectif est donc :
(pX, pY, pZ).
6.5 Réserve importante
Le canal de Pauli est une représentation effective des erreurs sur la durée tau. Il ne reproduit pas entièrement :
la dynamique temporelle du bruit quasi statique 1/f ;
les corrélations temporelles entre cycles ;
les effets non markoviens ;
la non-exponentialité de la cohérence de Ramsey ;
l’effet de la courbure d²f01/dΦ² ;
les sauts d’amplitude et de phase simultanés.
Le bruit quasi statique ne doit pas être converti automatiquement en un taux de déphasing markovien.
7. Qubit bosonique Kerr-cat
7.1 Dynamique Lindblad
Le module kerr_cat_simulator.py implémente une dynamique Lindblad pour un qubit bosonique Kerr-cat.
Il inclut :
pompage à deux photons ;
intégration RK4 ;
fidélité logique ;
estimation du leakage.
7.2 Indépendance du modèle
Ce module est indépendant du modèle Fluxonium. Il ne doit pas être présenté comme une simulation couplée Fluxonium–Kerr-cat complète.
8. Correction d’erreurs et chirurgie de réseau
8.1 Conversion des taux
Le module error_correction.py convertit les taux locaux en probabilités d’erreur de Pauli sur la durée d’un cycle tau_cycle.
Il implémente :
des codes de répétition de phase pour d impair ;
un budget d’erreur logique ;
un prototype de chirurgie de réseau sur code de surface tourné d=3 ;
une séquence :
Init
→ Merge (d rounds)
→ Split
→ Measure Z_L1
8.2 Décodage
Le décodage utilise un syndrome de poids 1. Le module geometric_mwpm_decoder.py permet une pondération du décodage MWPM par la géométrie et le biais local.
8.3 Réserve importante
La chirurgie de réseau est un prototype. Elle ne constitue pas :
une implémentation complète au niveau des stabilisateurs ;
une démonstration de seuil de correction d’erreurs ;
une validation d’une architecture matérielle ;
une preuve de correction d’erreurs universelle.
9. Structure du dépôt
.
├── init.py
├── error_correction.py
├── fluxon_engine.py
├── fluxonium_hamiltonian.py
├── fluxonium_model.py
├── fubini_study_manifold.py
├── geometric_mwpm_decoder.py
├── kerr_cat_simulator.py
├── noise_channel.py
├── pulse_calibration.py
│
├── test/
│   ├── test_chirurgie_21q_pont.py
│   ├── test_error_correction.py
│   ├── test_fluxonium.py
│   ├── test_fubini_study.py
│   └── test_kerr_cat.py
│
├── LICENSE
├── README.md
├── edifice.py
├── fluxonium_core.py
├── pont_lourd_unifie.py
├── pyproject.toml
├── requirements.txt
└── run_pipeline.py
10. Rôle des modules
11. Installation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
12. Exécution
Pipeline standard
python run_pipeline.py
Orchestrateur complet
python pont_lourd_unifie.py
13. Tests
pytest test/
Les tests valident des propriétés structurelles et numériques :
spectre et convergence du Fluxonium ;
convergence de la métrique de Fubini–Study ;
cohérence du solveur Lindblad Kerr-cat ;
codes de répétition et budget d’erreur ;
intégration du pont sur 21 qubits.
Ils ne remplacent pas une validation expérimentale.
14. Limites scientifiques
Ce pipeline ne doit pas être interprété comme une prédiction expérimentale directe.
Les principales limites sont :
le couplage géométrique reste phénoménologique ;
le canal de Pauli est une réduction effective ;
le bruit quasi statique n’est pas un canal de Lindblad markovien ;
les QP du réseau inductif ne sont pas modélisées ;
l’asymétrie de gap n’est pas modélisée ;
la chirurgie de réseau est un prototype ;
aucun contrôle DRAG n’est implémenté ;
aucune validation expérimentale n’est incluse.
15. Bonnes pratiques
Ne pas affirmer que le pipeline :
dérive un seuil de QEC ;
valide une architecture matérielle ;
prédit une fidélité expérimentale ;
implémente une correction d’erreurs complète ;
modélise toutes les sources de quasiparticules ;
remplace une calibration expérimentale.
Le pipeline est un outil de modélisation et de prototypage multi-échelle.
