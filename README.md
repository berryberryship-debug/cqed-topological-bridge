# Quantum-Lab — Fluxonium–Lattice Surgery Bridge

## Pipeline multi-échelle : Fluxonium, bruit quantique, QEC biaisée, Kerr-Cat et chirurgie de réseau — code de surface tourné (d=3), 21 qubits

## Quantum-Lab est un environnement de modélisation numérique consacré à l'étude des interfaces entre les circuits supraconducteurs, la géométrie quantique des états, les canaux de bruit et la correction d'erreurs quantique.

## Le projet relie plusieurs niveaux de description :

Substrat et quasiparticules
          │
          ▼
Fluxonium
T1 · T_phi · df01/dPhi · QGT · g_FS
          │
          ▼
Canaux de bruit
Amplitude damping + déphasage pur
          │
          ▼
Pauli twirling / PTM
pX · pY · pZ
          │
          ▼
Correction d'erreurs quantique
Bruit biaisé · budget logique · décodage
          │
          ▼
Chirurgie de réseau
Code de surface tourné d=3 · 21 qubits

## Le simulateur bosonique Kerr-Cat constitue une branche complémentaire du projet. Il possède sa propre dynamique et n'est pas encore intégré à une simulation couplée Fluxonium–Kerr-Cat.

---

# 1. Statut scientifique et périmètre épistémique

Ce dépôt implémente un pipeline de modélisation numérique multi-échelle. Il ne constitue ni une démonstration expérimentale, ni une dérivation physique complète de bout en bout.

Le projet distingue explicitement :

- les quantités calculées à partir d'un modèle physique ;
- les paramètres phénoménologiques ;
- les approximations de réduction ;
- les modèles numériques effectifs ;
- les prototypes de correction d'erreurs ;
- les éléments qui ne sont pas validés expérimentalement.

Il faut distinguer cinq niveaux de preuve :

Niveau| Signification
Modèle analytique| Relation définie à partir d'hypothèses physiques ou mathématiques
Calcul numérique| Résultat produit par une méthode de calcul déterminée
Validation numérique| Vérification d'invariants, de convergence ou comparaison à une référence
Prédiction expérimentale| Résultat quantitatif confronté à des mesures matérielles
Validation expérimentale| Reproduction indépendante et vérification sur dispositif réel

Une validation numérique locale ne constitue pas automatiquement une validation de toute la chaîne.

Le pipeline est conçu pour étudier la cohérence d'une architecture de modélisation, pas pour affirmer qu'une architecture matérielle fonctionnera telle quelle.

## 1.1 Fonctionnalités présentes dans le périmètre du projet

Les modules du dépôt couvrent les domaines suivants :

Fluxonium

- Construction de l'Hamiltonien en base de phase.
- Diagonalisation numérique du spectre.
- Calcul des niveaux d'énergie et des états propres.
- Calcul des fréquences de transition.
- Éléments de matrice de charge et de phase.
- Estimations des temps de relaxation et de déphasage.
- Sensibilité fréquentielle au flux.

Géométrie quantique

- Quantum geometric tensor (QGT).
- Métrique de Fubini–Study (g_{FS}).
- Calcul de sensibilité géométrique des états propres.

Substrat et quasiparticules

- Modulation spatiale paramétrique du flux.
- Modèle effectif de densité de quasiparticules (x_{qp}).
- Bilan numérique de production et de pertes.
- Contribution des quasiparticules froides de la petite jonction Josephson.

Canaux de bruit

- Amplitude damping.
- Déphasage pur.
- Matrice de transfert de Pauli (PTM).
- Pauli twirling.
- Probabilités effectives (p_X,p_Y,p_Z).

Correction d'erreurs quantique

- Codes de répétition de phase.
- Modèle de bruit biaisé.
- Budget d'erreur logique.
- Décodage pondéré par des paramètres locaux.
- Prototype de chirurgie de réseau sur code de surface tourné (d=3).

Qubit bosonique

- Modèle Kerr-Cat.
- Pompage à deux photons.
- Équation maîtresse de Lindblad.
- Intégration numérique RK4.
- Estimations de fidélité logique et de leakage.

La présence d'une fonctionnalité dans cette liste ne signifie pas qu'elle a été validée expérimentalement.

## 1.2 Éléments non implémentés ou non établis

Les éléments suivants ne doivent pas être présentés comme des fonctionnalités achevées :

- Contrôle DRAG.
- Fonction de Wigner.
- Dynamique microscopique complète des quasiparticules.
- Diffusion et piégeage microscopiques des quasiparticules.
- Quasiparticules dans le réseau inductif.
- Asymétrie de gap supraconducteur.
- Couplage physique complet entre quatre Fluxonium.
- Chirurgie de réseau complète au niveau des stabilisateurs et des circuits.
- Validation expérimentale intégrée.
- Démonstration d'un seuil de correction d'erreurs.

## 1.3 Paramètres et réductions phénoménologiques

Les éléments suivants nécessitent une qualification particulière :

- Le couplage géométrique "fs_coupling_gamma".
- La réduction du Fluxonium à un sous-espace de deux niveaux.
- La conversion des taux continus en probabilités de Pauli par cycle.
- Le traitement du bruit quasi statique par un canal effectif.
- Le couplage XX entre qubits lorsqu'il est fourni comme paramètre d'entrée.
- Les paramètres de modulation spatiale du substrat.
- Toute relation postulée entre une métrique géométrique et une probabilité d'erreur.

Ces paramètres doivent être documentés avec leurs unités, leur domaine de validité et leur méthode de calibration.
## 1.3 ## Audit : corrections et points ouverts

*Section à insérer après « Statut scientifique et périmètre épistémique ». Rédigée à partir de la page GitHub du dépôt et du README long. Le code, les tests et la CI n'ont pas été lus : tout ce qui touche à leur contenu est marqué « à vérifier ».*

### A. Cohérence documentaire (à trancher)

Deux versions du README coexistent et se contredisent. La version prudente (périmètre épistémique) est retenue ici comme référence.

| Point | Version courte | Version prudente (référence) |
|---|---|---|
| DRAG / pulse_calibration | Implémenté | Non implémenté |
| Fonction de Wigner | Présente | Non implémentée |
| Tolérance du bilan | r ≤ 1e-15 | 1e-10 |
| Chirurgie de réseau | « Protocole complet » | Prototype Monte-Carlo |
| Kerr-Cat | « Couplé » | Branche indépendante |
| Dérivées en phase | Ordre 4 ou Fock N=35 | Stencil central, Dirichlet |

Actions : supprimer la version courte, ou la mettre à jour après vérification dans le code.

### B. Correction mathématique : twirling de l'amplitude damping

Remplace les probabilités du §6.2. Avec γ = 1 − e^(−τ/T1), la PTM de l'amplitude damping a pour diagonale (1, √(1−γ), √(1−γ), 1−γ). Les probabilités de Pauli sont p_P = (1 + Σ ± λ_i)/4, ce qui donne :

- p_X = p_Y = γ/4
- p_Z = (2 − γ − 2√(1−γ)) / 4 ≈ γ²/16 pour γ petit
- p_I = 1 − p_X − p_Y − p_Z

L'ancienne valeur p_Z = γ/2 est erronée. Référence : Ghosh, Fowler, Geller (2012), twirling de l'amplitude damping pour le code de surface (référence complète à vérifier).

Formules vérifiées et conservées :

- Déphasage pur : p_Z = (1 − e^(−τ/Tφ))/2
- 1/T2 = 1/(2T1) + 1/Tφ

**Test à ajouter :** pour γ ∈ {1e-4, 1e-2, 0.3, 0.9}, comparer les p calculés à la formule fermée ci-dessus, et vérifier Σp = 1.

### C. Bilan de quasiparticules : portée du test

Si le résidu R_qp = x(t+Δt) − x(t) − P + L est calculé avec la formule qui met à jour x, il est nul par construction. Il vérifie alors l'arithmétique, pas la physique.

- Documenter comment x(t+Δt) est calculé.
- Ajouter un test indépendant (solution analytique d'un cas simple, ou comparaison avec un intégrateur de référence).
- Pour 1e-15 : l'epsilon machine en float64 vaut environ 2,2e-16 ; une tolérance de cet ordre est irréaliste sur de longues intégrations. Justifier ou relâcher.

### D. Comptage des qubits (chirurgie 21Q)

« 2 patchs × 9 qubits de données + 3 ancillas d'interface = 21 » ne comprend pas les ancillas de stabilisateurs. À ma connaissance, un patch tourné d=3 compte 9 données + 8 ancillas = 17 qubits (Horsman et al., *New J. Phys.* 14, 123011, 2012). Préciser ce que représentent les 21 qubits, ou renommer le prototype.

### E. Conventions à unifier

- Position du flux : (φ − φ_ext)² dans une version, cos(φ − Φ) dans l'autre ; terme n_g présent dans une seule. Fixer une convention et son signe.
- La métrique de Fubini–Study par différence finie, g ≈ (1 − |⟨ψ|ψ(Φ+dΦ)⟩|²)/dΦ², est correcte à l'ordre dominant ; préciser l'état utilisé (fondamental ou état de qubit) et le pas dΦ.
- Préfacteur du taux quasiparticules (8/π) : à confronter à Pop et al., *Nature* 508, 369 (2014) ; non vérifié ici.

### F. Hygiène du dépôt

- Renommer `NOYAU QUANTIQUE V52.py` (espaces, majuscules) ; retirer les numéros de version des noms de fichiers (git suffit).
- Aligner l'arborescence du README sur le dépôt : `test/` ou `tests/`, `notebooks/`, `results/`, présence de `run_pipeline.py`.
- Ajouter une CI (GitHub Actions) exécutant `python -m pytest -q`, avec badge.

### G. Sources à vérifier avant publication

| Sujet | Référence | Statut |
|---|---|---|
| Twirling de l'amplitude damping | Ghosh, Fowler, Geller (2012) | Titre et lien à confirmer |
| Chirurgie de réseau | Horsman et al., NJP 14, 123011 (2012) | À confirmer |
| Quasiparticules, fluxonium | Pop et al., Nature 508, 369 (2014) | À confirmer |
| Kerr-Cat | Puri, Boutin, Blais, npj Quantum Inf. 3, 18 (2017) | À confirmer |

Aucun lien n'a été ouvert ni vérifié à ce stade. Les références sont données de mémoire : en cas de doute, « Je ne sais pas ».

### H. Interface de recherche

Le « ? » entre géométrie quantique (g_FS) et canal de bruit reste l'interface ouverte du projet. Tant qu'une relation physique testable n'est pas établie, `fs_coupling_gamma` reste un paramètre phénoménologique, et toute analogie entre géométrie et erreur relève de l'hypothèse de travail, pas du résultat.

---

# 2. Architecture du pipeline

## 2.1 Vue d'ensemble

L'architecture générale suit une succession de transformations entre niveaux de description :

SUBSTRAT
   │
   ├── Modulation spatiale du flux
   └── Densité effective de quasiparticules
              │
              ▼
FLUXONIUM
   │
   ├── Hamiltonien H
   ├── Spectre et états propres
   ├── T1 et T_phi
   ├── df01/dPhi
   └── QGT et g_FS
              │
       ┌──────┴────────┐
       ▼               ▼
GÉOMÉTRIE          BRUIT
QUANTIQUE           │
       │            ├── Amplitude damping
       │            ├── Déphasage pur
       │            └── PTM
       │                  │
       │                  ▼
       │             PAULI TWIRLING
       │             pX · pY · pZ
       │                  │
       └────── ? ─────────┘
                          │
                          ▼
                   MODÈLES QEC
                          │
                          ├── Répétition de phase
                          ├── Budget logique
                          ├── Décodage
                          └── Chirurgie 21Q

Le point d'interrogation désigne une interface de recherche : la géométrie quantique et le modèle de bruit peuvent être calculés séparément, mais leur couplage doit être défini par une relation physique ou phénoménologique explicite.

Le calcul de (g_{FS}) ne suffit pas à déduire une probabilité d'erreur.

## 2.2 Modulation spatiale du substrat

Le module "fluxon_engine.py" décrit une modulation spatiale paramétrique du substrat.

Chaque qubit (i) reçoit un décalage de flux :

[
\Delta\Phi_i.
]

Ce décalage peut être pondéré par une enveloppe gaussienne centrée autour du point doux :

[
\Phi_{\mathrm{ext}}=\pi.
]

La chaîne de calcul locale est :

[
\Delta\Phi_i
\longrightarrow
\Phi_{\mathrm{ext},i}
\longrightarrow
f_{01,i}
\longrightarrow
T_{1,i},T_{\phi,i}
\longrightarrow
p_{X,i},p_{Y,i},p_{Z,i}.
]

Cette modulation permet d'étudier l'effet d'une inhomogénéité paramétrique sur les propriétés locales des qubits.

Elle ne constitue pas un modèle de fabrication ni une description complète des variations matérielles d'un dispositif supraconducteur réel.

## 2.3 Noyau dynamique conservatif

Le module "fluxon_engine.py" maintient un bilan effectif de densité de quasiparticules hors équilibre :

[
x_{qp}.
]

Le bilan numérique est formulé sous la forme :

[
x_{qp}(t+\Delta t)

x_{qp}(t)
+
P(t,\Delta t)

L(t,\Delta t),
]

où :

- (P) représente la production effective ;
- (L) représente les pertes effectives.

Le résidu du bilan est défini par :

[
R_{qp}

x_{qp}(t+\Delta t)

x_{qp}(t)

P
+
L.
]

Une tolérance numérique annoncée dans le modèle est :

[
|R_{qp}|\leq 10^{-10}.
]

Cette tolérance concerne la fermeture numérique du bilan défini par le code. Elle ne démontre pas que la dynamique représente quantitativement les quasiparticules d'un circuit réel.

Le modèle ne comprend notamment pas :

- la diffusion microscopique ;
- le piégeage dans les défauts ;
- la recombinaison microscopique ;
- les effets de bord ;
- les quasiparticules dans le réseau inductif ;
- les asymétries de gap supraconducteur.

---

# 3. Solveur Hamiltonien Fluxonium

## 3.1 Hamiltonien

Le module "fluxonium_hamiltonian.py" construit et diagonalise l'Hamiltonien du Fluxonium en base de phase.

L'Hamiltonien est :

[
H=
4E_C\hat n^2
+
\frac{1}{2}E_L(\hat\phi-\phi_{\mathrm{ext}})^2

E_J\cos(\hat\phi).
]

Les opérateurs satisfont la relation canonique :

[
[\hat\phi,\hat n]=i.
]

Les conditions aux limites numériques sont de type Dirichlet. La discrétisation en phase utilise un stencil central.

Les paramètres énergétiques peuvent être exprimés en unités de fréquence, par exemple :

[
E/h \quad \text{en GHz}.
]

La convention de conversion entre énergie et fréquence doit rester cohérente dans l'ensemble du solveur.

## 3.2 Quantités extraites

Le solveur permet d'obtenir notamment :

- les niveaux d'énergie ;
- les états propres ;
- les fréquences de transition ;
- les éléments de matrice ;
- les dipôles de charge ;
- les dipôles de phase ;
- les observables nécessaires aux estimations de relaxation ;
- les sensibilités fréquentielles ;
- les quantités nécessaires au calcul géométrique.

Les taux de relaxation et de déphasage dépendent des hypothèses de bruit et des paramètres physiques introduits. Ils ne sont pas des constantes intrinsèques déterminées par le seul spectre.

## 3.3 Convergence numérique

La convergence doit être examinée selon plusieurs dimensions :

- taille de la grille ;
- étendue du domaine en phase ;
- conditions aux limites ;
- précision des dérivées numériques ;
- stabilité des états propres ;
- stabilité des éléments de matrice ;
- stabilité des fréquences de transition.

Les comparaisons peuvent porter sur :

[
f_{01},
\qquad
\langle 0|\hat n|1\rangle,
\qquad
\langle 0|\hat\phi|1\rangle,
\qquad
\frac{\partial f_{01}}{\partial\Phi},
\qquad
g_{FS}.
]

La convergence locale ne garantit pas la convergence globale dans tout l'espace des paramètres.

Une vérification de convergence doit aussi contrôler l'hermiticité numérique :

[
|H-H^\dagger|.
]

---

# 4. Observables du Fluxonium

## 4.1 Observables principales

Le module "fluxonium_model.py" regroupe les observables physiques utilisées par les couches supérieures du pipeline.

Les principales grandeurs concernent :

- les fréquences de transition ;
- les éléments de matrice ;
- les taux de relaxation ;
- les taux de déphasage ;
- la sensibilité au flux ;
- les quantités géométriques associées aux états propres.

## 4.2 Sensibilité au flux

La dérivée de fréquence relie une fluctuation de flux à une fluctuation de fréquence.

Au premier ordre :

[
\delta f_{01}
\simeq
\frac{\partial f_{01}}{\partial\Phi}\delta\Phi.
]

Au second ordre :

[
\delta f_{01}
\simeq
\frac{\partial f_{01}}{\partial\Phi}\delta\Phi
+
\frac{1}{2}
\frac{\partial^2 f_{01}}{\partial\Phi^2}
(\delta\Phi)^2.
]

Le premier terme décrit la sensibilité linéaire. Le second introduit la contribution de la courbure fréquentielle.

Cette distinction devient importante au voisinage d'un point doux, où la dérivée première peut être faible alors que la dérivée seconde reste non nulle.

4.3 Quantum geometric tensor et métrique de Fubini–Study

Le quantum geometric tensor (QGT) décrit la réponse géométrique d'un état quantique à une variation de paramètres.

Pour un état pur normalisé (|\psi(\lambda)\rangle), sa forme projectée peut s'écrire :

[
Q_{\mu\nu}

\langle\partial_\mu\psi|
(1-|\psi\rangle\langle\psi|)
|\partial_\nu\psi\rangle.
]

La métrique de Fubini–Study correspond à sa partie réelle :

[
g_{\mu\nu}^{FS}

\operatorname{Re}(Q_{\mu\nu}).
]

Dans le cas d'un paramètre de flux :

[
g_{\Phi\Phi}^{FS}

\operatorname{Re}(Q_{\Phi\Phi}).
]

Cette métrique mesure la sensibilité géométrique de l'état quantique aux variations du paramètre considéré.

Elle ne détermine pas, à elle seule :

- un taux de relaxation ;
- un taux de déphasage ;
- une probabilité d'erreur ;
- un couplage physique entre deux qubits.

Le paramètre "fs_coupling_gamma" demeure phénoménologique tant qu'une relation physique indépendante et testable ne le justifie pas.

---

# 5. Quasiparticules froides

## 5.1 Modèle de la petite jonction

Le modèle de relaxation par quasiparticules concerne la petite jonction Josephson du Fluxonium, dans l'approximation des quasiparticules froides.

Le taux descendant est de la forme :

[
\Gamma_{\mathrm{qp},\downarrow}

\frac{8}{\pi}
\frac{E_J}{\hbar}
x_{qp}
\sqrt{\frac{2\Delta}{h f_{01}}}
\left|
\left\langle 0\left|
\sin\left(
\frac{\hat\phi-\phi_{\mathrm{ext}}}{2}
\right)
\right|1\right\rangle
\right|^2.
]

Cette expression doit être interprétée avec les conventions de normalisation de (x_{qp}), de (E_J), de (\Delta) et des opérateurs utilisées par l'implémentation.

La vérification dimensionnelle et la comparaison à la formulation de référence sont nécessaires avant toute interprétation quantitative.

## 5.2 Hypothèses

Le modèle repose sur les hypothèses suivantes :

- quasiparticules froides ;
- énergie typique des quasiparticules très inférieure à (hf_{01}) ;
- (hf_{01}\ll\Delta) ;
- contribution de la petite jonction uniquement ;
- absence de taux montant QP dans le modèle considéré ;
- absence de contribution QP du réseau inductif ;
- absence d'asymétrie de gap supraconducteur ;
- absence de modélisation microscopique du piégeage et de la recombinaison.

## 5.3 Portée

Le taux calculé est une contribution de relaxation associée à la petite jonction dans les hypothèses du modèle.

Il ne représente pas le taux total de relaxation par quasiparticules d'un Fluxonium réel.

---

# 6. Canal de bruit et Pauli twirling

## 6.1 Construction du canal

Le module "noise_channel.py" construit un canal de bruit effectif sur une durée (\tau).

Les mécanismes considérés comprennent :

- l'amplitude damping ;
- le déphasage pur ;
- la représentation par matrice de transfert de Pauli ;
- le Pauli twirling ;
- la composition des canaux dans la représentation PTM.

Les taux physiques sont convertis en paramètres de canal selon les conventions et les approximations explicitement utilisées.

## 6.2 Amplitude damping

Pour un canal d'amplitude damping de paramètre :

[
\gamma=1-e^{-\tau/T_1},
]

le Pauli twirling donne les probabilités :

[
p_X=p_Y=\frac{\gamma}{4},
]

[
p_Z=\frac{\gamma}{2},
]

[
p_I=1-\gamma.
]

Ces probabilités décrivent le canal de Pauli obtenu après twirling du canal d'amplitude damping.

Elles ne signifient pas que l'amplitude damping initial est lui-même un canal de Pauli.

6.3 Déphasage pur

Pour un déphasage pur de paramètre :

[
\lambda_Z

\frac{1-e^{-\tau/T_\phi}}{2},
]

la représentation de Pauli est :

[
p_Z=\lambda_Z,
\qquad
p_I=1-\lambda_Z.
]

Cette expression suppose la convention de déphasage pur retenue par le module.

Il faut distinguer (T_\phi) de (T_2). Lorsque les mécanismes considérés sont ceux d'une relaxation d'amplitude et d'un déphasage pur markovien :

[
\frac{1}{T_2}

\frac{1}{2T_1}
+
\frac{1}{T_\phi}.
]

6.4 Composition PTM

La matrice de transfert de Pauli représente l'action du canal sur les opérateurs de Pauli.

Pour les canaux de Pauli, la composition correspond à la multiplication de leurs matrices de transfert :

[
R_{\mathcal E_2\circ\mathcal E_1}

R_{\mathcal E_2}R_{\mathcal E_1}.
]

Lorsque les matrices considérées sont diagonales dans la base de Pauli, la composition se réduit au produit de leurs composantes diagonales.

Le canal de Pauli effectif est ensuite représenté par :

[
\mathcal E_{\mathrm{Pauli}}

{p_I,p_X,p_Y,p_Z}.
]

Précision épistémique : le twirling est une opération de réduction du canal. Il supprime certaines composantes cohérentes et non diagonales. L'exactitude algébrique de la composition PTM ne signifie donc pas que le canal twirlé conserve toute l'information du canal quantique initial.

## 6.5 Bruit quasi statique et bruit coloré

Le canal de Pauli effectif ne reproduit pas intégralement :

- les corrélations temporelles entre cycles ;
- la dynamique du bruit quasi statique (1/f) ;
- les effets non markoviens ;
- la non-exponentialité de la cohérence de Ramsey ;
- les contributions de la courbure (\partial^2 f_{01}/\partial\Phi^2) ;
- les corrélations entre erreurs d'amplitude et de phase.

Le bruit quasi statique ne doit pas être automatiquement assimilé à un taux de déphasage markovien constant.

Une réduction en canal effectif doit préciser la durée considérée, la distribution du bruit, les hypothèses temporelles et la méthode de twirling.

---

# 7. Qubit bosonique Kerr-Cat

## 7.1 Modèle dynamique

Le module "kerr_cat_simulator.py" étudie une dynamique bosonique de type Kerr-Cat.

Le modèle comprend notamment :

- le pompage à deux photons ;
- un Hamiltonien effectif de Kerr ;
- une équation maîtresse de Lindblad ;
- une intégration numérique RK4 ;
- des estimations de fidélité logique ;
- une estimation du leakage.

La dynamique est décrite sous la forme générale :

[
\frac{d\rho}{dt}

-\frac{i}{\hbar}[H,\rho]
+
\sum_j
\mathcal D"L_j" (\rho),
]

où :

[
\mathcal D"L" (\rho)

L\rho L^\dagger

\frac{1}{2}
\left{
L^\dagger L,\rho
\right}.
]

La signification physique des opérateurs (L_j), leurs unités et leurs taux doivent être définis par les paramètres du modèle.

## 7.2 Indépendance du modèle

Le simulateur Kerr-Cat est une branche bosonique indépendante du modèle Fluxonium.

Il ne constitue pas une simulation couplée complète entre un Fluxonium et un Kerr-Cat.

Les résultats de fidélité et de leakage sont conditionnels au Hamiltonien, aux dissipateurs, à la troncature de Fock et à la méthode d'intégration employés.

Toute fonctionnalité de calibration qui lève "NotImplementedError" doit être considérée comme non opérationnelle.

---

# 8. Correction d'erreurs quantique et chirurgie de réseau

## 8.1 Conversion des taux en probabilités locales

Le module "error_correction.py" utilise des probabilités d'erreur locales associées à une durée de cycle :

[
\tau_{\mathrm{cycle}}.
]

Les paramètres de bruit peuvent être convertis en :

[
p_{X,i},\quad p_{Y,i},\quad p_{Z,i}.
]

Cette conversion dépend du canal retenu et des approximations de réduction.

Elle ne constitue pas une description complète des erreurs corrélées, cohérentes ou temporellement persistantes.

## 8.2 Code de répétition de phase

Le modèle de répétition de phase considère une distance impaire (d).

Les états logiques sont :

[
|0_L\rangle=|+\rangle^{\otimes d},
]

[
|1_L\rangle=|-\rangle^{\otimes d}.
]

Pour un modèle indépendant d'erreurs de phase de probabilité (p_Z), la probabilité d'une majorité d'erreurs est :

[
P_L^{(Z)}

\sum_{k=(d+1)/2}^{d}
\binom{d}{k}
p_Z^k(1-p_Z)^{d-k}.
]

Cette expression suppose des erreurs indépendantes et identiquement distribuées dans le modèle considéré.

Elle ne décrit pas automatiquement les erreurs corrélées ou les erreurs de mesure.

## 8.3 Budget d'erreur logique

Le budget logique agrège les contributions d'erreurs selon les hypothèses du modèle.

Lorsque plusieurs contributions sont combinées par une formule d'indépendance, cette hypothèse doit être explicitée.

Une combinaison de probabilités marginales ne doit pas être interprétée comme une distribution conjointe exacte en l'absence de vérification de l'indépendance.

## 8.4 Prototype de chirurgie de réseau

Le prototype de chirurgie de réseau étudie une séquence de type :

Init
  │
  ▼
Merge
  │
  ▼
d rounds
  │
  ▼
Split
  │
  ▼
Measure Z_L1

La configuration annoncée est un code de surface tourné de distance :

[
d=3
]

avec :

[
21\ \text{qubits}.
]

La chirurgie est un prototype numérique Monte-Carlo simplifié.

Elle ne doit pas être présentée comme :

- une simulation complète au niveau des stabilisateurs ;
- une simulation complète des circuits de portes ;
- une modélisation exhaustive des erreurs de lecture ;
- une démonstration de seuil QEC ;
- une validation d'une architecture matérielle ;
- une preuve de correction d'erreurs universelle.

## 8.5 Décodage géométrique

Le module "geometric_mwpm_decoder.py" explore une pondération du décodage par des informations géométriques et par le biais local.

Cette approche constitue un prototype de décodage pondéré.

Pour revendiquer un MWPM complet de code de surface, il faudrait notamment spécifier et valider :

- le graphe des événements de syndrome ;
- les erreurs de données ;
- les erreurs de mesure ;
- les frontières logiques ;
- les contraintes des stabilisateurs X et Z ;
- l'historique espace-temps ;
- les règles de merge et de split ;
- le traitement des chemins logiques.

Une pondération géométrique ne suffit pas, à elle seule, à établir ces propriétés.

---

# 9. Structure du dépôt

L'organisation du dépôt sépare les modules de calcul, les orchestrateurs et les tests.

cqed-topological-bridge/
├── __init__.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── README.md
│
├── run_pipeline.py
├── pont_lourd_unifie.py
├── edifice.py
├── fluxonium_core.py
│
├── src/
│   ├── __init__.py
│   ├── error_correction.py
│   ├── fluxon_engine.py
│   ├── fluxonium_hamiltonian.py
│   ├── fluxonium_model.py
│   ├── fubini_study_manifold.py
│   ├── geometric_mwpm_decoder.py
│   ├── kerr_cat_simulator.py
│   ├── noise_channel.py
│   └── pulse_calibration.py
│
└── test/
    ├── test_chirurgie_21q_pont.py
    ├── test_error_correction.py
    ├── test_fluxonium.py
    ├── test_fubini_study.py
    └── test_kerr_cat.py

Cette arborescence décrit l'organisation documentée du projet. Les noms de modules, les exports et les points d'entrée doivent rester synchronisés avec les fichiers effectivement suivis dans le dépôt.

---

# 10. Rôle des modules

Module| Responsabilité
"fluxonium_core.py"| Structures et constantes liées au modèle Fluxonium
"fluxonium_hamiltonian.py"| Construction et diagonalisation de l'Hamiltonien
"fluxonium_model.py"| Observables, éléments de matrice et estimations de cohérence
"fubini_study_manifold.py"| QGT et géométrie de Fubini–Study
"fluxon_engine.py"| Modulation du substrat et bilan effectif de quasiparticules
"noise_channel.py"| Canaux de bruit, PTM et Pauli twirling
"kerr_cat_simulator.py"| Dynamique bosonique Kerr-Cat
"pulse_calibration.py"| Outils de calibration d'impulsions
"error_correction.py"| Codes de répétition et budget d'erreur logique
"geometric_mwpm_decoder.py"| Prototype de décodage pondéré
"pont_lourd_unifie.py"| Orchestration du pont et prototype de chirurgie 21Q
"edifice.py"| Architecture globale et intégration des couches
"run_pipeline.py"| Point d'entrée du pipeline standard

---

# 11. Installation

Créer un environnement virtuel :

python -m venv .venv

Activer l'environnement.

Linux / macOS

source .venv/bin/activate

Windows PowerShell

.\.venv\Scripts\Activate.ps1

Installer les dépendances :

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

Les dépendances doivent être conformes aux déclarations de "requirements.txt" et "pyproject.toml".

---

# 12. Exécution

Les points d'entrée présents dans le dépôt sont :

Pipeline standard

python run_pipeline.py

Orchestrateur du pont

python pont_lourd_unifie.py

Ces commandes correspondent aux scripts présents dans le dépôt. Leur exécution effective dépend de la cohérence des imports, des paramètres requis et des API des modules.

Un point d'entrée ne doit être annoncé comme opérationnel qu'après une exécution réussie dans un environnement propre.

---

# 13. Tests

Lancer la suite :

python -m pytest test/ -v

Ou l'ensemble des tests détectés par pytest :

python -m pytest -q

Les tests couvrent les domaines suivants :

Fichier| Domaine
"test_fluxonium.py"| Spectre, états propres et propriétés du Fluxonium
"test_fubini_study.py"| QGT et métrique de Fubini–Study
"test_kerr_cat.py"| Dynamique Lindblad, fidélité et leakage
"test_error_correction.py"| Codes de répétition et budget d'erreur
"test_chirurgie_21q_pont.py"| Intégration du prototype de chirurgie 21 qubits

Les tests peuvent vérifier :

- des invariants algébriques ;
- la cohérence de calculs numériques ;
- la convergence d'une méthode ;
- la stabilité d'une simulation ;
- le comportement d'une abstraction logicielle.

Ils ne remplacent pas une validation analytique indépendante ni une validation expérimentale.

Aucun résultat de test ne doit être annoncé comme réussi sans exécution effective sur le commit concerné.

---

# 14. Limites scientifiques

Les principales limites du pipeline sont les suivantes :

Domaine| Limite actuelle
Fluxonium| Résultats dépendants des paramètres, de la discrétisation et des hypothèses de bruit
Géométrie quantique| (g_{FS}) ne détermine pas directement les probabilités d'erreur
Quasiparticules| Modèle limité à la contribution froide de la petite jonction
Substrat| Bilan effectif, sans dynamique microscopique complète
Bruit| Canal de Pauli réduit, avec perte d'information liée au twirling
Bruit quasi statique| Corrélations temporelles non entièrement représentées par un canal markovien
Kerr-Cat| Branche indépendante du Fluxonium
Calibration| Pas de contrôle DRAG implémenté
QEC| Modèles effectifs et prototypes
Décodage| Pas de démonstration d'un MWPM espace-temps complet
Chirurgie| Prototype Monte-Carlo, non simulation stabilizer-level complète
Expérience| Aucune validation expérimentale intégrée

---

# 15. Feuille de route de validation

Les étapes suivantes sont des objectifs de recherche et de validation.

P0 — Cohérence logicielle

- Vérifier les imports de tous les modules.
- Aligner les exports de "src/__init__.py" sur les symboles réellement définis.
- Vérifier la cohérence entre les API et les tests.
- Éliminer toute commande shell présente accidentellement dans les fichiers Python de test.
- Exécuter les tests dans un environnement propre.
- Documenter les résultats d'exécution.

P1 — Solveur Fluxonium

- Vérifier l'hermiticité de l'Hamiltonien.
- Vérifier les opérateurs discrets.
- Étudier la convergence en taille de grille.
- Étudier la convergence en étendue de domaine.
- Comparer les spectres à une méthode indépendante.
- Vérifier les éléments de matrice.

P2 — Géométrie quantique

- Vérifier la normalisation des états.
- Contrôler l'invariance de jauge.
- Étudier la convergence en pas de différence finie.
- Comparer les méthodes de calcul du QGT.
- Distinguer les métriques locales des grandeurs physiques de bruit.

P3 — Bruit et canaux

- Documenter les conventions spectrales.
- Distinguer bruit blanc, bruit coloré et bruit quasi statique.
- Comparer les canaux à une évolution de référence.
- Quantifier l'information perdue par le twirling.
- Vérifier la conversion des taux en probabilités par cycle.

P4 — QEC et décodage

- Vérifier les probabilités logiques par énumération exhaustive pour les petits codes.
- Séparer erreurs de données et erreurs de mesure.
- Définir le graphe des événements de syndrome.
- Construire un graphe espace-temps.
- Comparer le décodeur à une implémentation de référence.

P5 — Chirurgie de réseau

- Définir les qubits de données et ancillas.
- Définir les stabilisateurs initiaux et finaux.
- Décrire les circuits de préparation, merge et split.
- Modéliser les erreurs de portes et de lecture.
- Vérifier le suivi du cadre de Pauli.
- Vérifier les observables logiques.
- Étudier plusieurs distances avant toute affirmation de seuil.

P6 — Intégration physique

- Définir les hypothèses de couplage entre qubits.
- Relier les paramètres matériels aux canaux locaux.
- Établir une interface physique testable entre géométrie et bruit, si elle est retenue.
- Comparer les prédictions aux données expérimentales.
- Publier les incertitudes et les domaines de validité.

---

# 16. Bonnes pratiques d'interprétation

Les résultats du pipeline doivent être présentés avec leur niveau de preuve.

Ne pas affirmer que le projet :

- dérive un seuil de QEC ;
- valide une architecture matérielle ;
- prédit une fidélité expérimentale ;
- implémente une correction d'erreurs complète ;
- modélise toutes les sources de quasiparticules ;
- établit un couplage géométrique physique démontré ;
- remplace une calibration expérimentale ;
- reproduit intégralement un bruit non markovien à partir d'un canal de Pauli indépendant.

Pour chaque résultat quantitatif, documenter autant que possible :

1. les paramètres matériels ;
2. les unités ;
3. les hypothèses physiques ;
4. les conditions aux limites ;
5. la taille de grille ou de base ;
6. les tolérances numériques ;
7. la durée d'évolution ou de cycle ;
8. le modèle de bruit ;
9. les conventions de twirling ;
10. la graine aléatoire et le nombre de tirs ;
11. les incertitudes numériques et statistiques ;
12. le commit du code utilisé.

---

# 17. Positionnement du projet

Quantum-Lab étudie une question centrale :

Comment relier de manière traçable les propriétés microscopiques d'un circuit supraconducteur, la géométrie de ses états quantiques, ses canaux de bruit effectifs et les performances d'une architecture de correction d'erreurs ?

L'intérêt du projet réside dans la construction d'interfaces explicites entre ces niveaux.

La géométrie quantique, les modèles de dissipation, les canaux de Pauli et les mécanismes de correction d'erreurs sont des descriptions distinctes. Leur articulation exige des hypothèses identifiables, des transformations mathématiques explicites et des validations indépendantes.

Le pipeline constitue ainsi un environnement de recherche numérique pour explorer ces relations, tester leur cohérence et identifier les conditions nécessaires à une intégration physique plus complète.

Quantum-Lab — Modéliser les interfaces, contrôler les réductions, éprouver les correspondances.
