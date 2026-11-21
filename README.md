# Quantum-Lab — Fluxonium–Lattice Surgery Bridge

## Pipeline multi-échelle : Fluxonium, bruit quantique, QEC biaisée, Kerr-Cat et chirurgie de réseau — prototype de code de surface tourné (d=3), 21 qubits

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
Prototype de code de surface tourné d=3 · 21 qubits

Le simulateur bosonique Kerr-Cat constitue une branche complémentaire du projet. Il possède sa propre dynamique et n'est pas encore intégré à une simulation couplée Fluxonium–Kerr-Cat.

---

1. Statut scientifique et périmètre épistémique

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

## 1.1 Fonctionnalités présentes dans le périmètre documentaire

Les modules du projet sont documentés comme couvrant les domaines suivants :

Fluxonium

- construction de l'Hamiltonien en base de phase ;
- diagonalisation numérique du spectre ;
- calcul des niveaux d'énergie et des états propres ;
- calcul des fréquences de transition ;
- éléments de matrice de charge et de phase ;
- estimations des temps de relaxation et de déphasage ;
- sensibilité fréquentielle au flux.

Géométrie quantique

- quantum geometric tensor (QGT) ;
- métrique de Fubini–Study (g_{FS}) ;
- calcul de sensibilité géométrique des états propres.

Substrat et quasiparticules

- modulation spatiale paramétrique du flux ;
- modèle effectif de densité de quasiparticules (x_{qp}) ;
- bilan numérique de production et de pertes ;
- contribution des quasiparticules froides de la petite jonction Josephson.

Canaux de bruit

- amplitude damping ;
- déphasage pur ;
- matrice de transfert de Pauli (PTM) ;
- Pauli twirling ;
- probabilités effectives (p_X,p_Y,p_Z).

Correction d'erreurs quantique

- codes de répétition de phase ;
- modèle de bruit biaisé ;
- budget d'erreur logique ;
- décodage pondéré par des paramètres locaux ;
- prototype de chirurgie de réseau sur code de surface tourné, distance (d=3).

Qubit bosonique

- modèle Kerr-Cat ;
- pompage à deux photons ;
- équation maîtresse de Lindblad ;
- intégration numérique ;
- estimations de fidélité logique et de leakage.

La présence d'une fonctionnalité dans cette liste ne signifie pas qu'elle a été validée expérimentalement.

---

## 1.2 Éléments non implémentés ou non établis

Les éléments suivants ne doivent pas être présentés comme des fonctionnalités achevées sans vérification du code correspondant :

- contrôle DRAG ;
- fonction de Wigner ;
- dynamique microscopique complète des quasiparticules ;
- diffusion et piégeage microscopiques des quasiparticules ;
- quasiparticules dans le réseau inductif ;
- asymétrie de gap supraconducteur ;
- couplage physique complet entre quatre Fluxonium ;
- chirurgie de réseau complète au niveau des stabilisateurs et des circuits ;
- validation expérimentale intégrée ;
- démonstration d'un seuil de correction d'erreurs.

Les éléments ci-dessus constituent soit des extensions prévues, soit des limites actuelles du périmètre scientifique.

---

## 1.3 Paramètres et réductions phénoménologiques

Les éléments suivants nécessitent une qualification particulière :

- le couplage géométrique "fs_coupling_gamma" ;
- la réduction du Fluxonium à un sous-espace de deux niveaux ;
- la conversion des taux continus en probabilités de Pauli par cycle ;
- le traitement du bruit quasi statique par un canal effectif ;
- le couplage (XX) entre qubits lorsqu'il est fourni comme paramètre d'entrée ;
- les paramètres de modulation spatiale du substrat ;
- toute relation postulée entre une métrique géométrique et une probabilité d'erreur.

Ces paramètres doivent être documentés avec :

- leurs unités ;
- leur domaine de validité ;
- leur convention ;
- leur méthode de calibration ;
- leur statut théorique, numérique ou phénoménologique.

En particulier, aucune relation directe

[
g_{FS}\rightarrow p_{\mathrm{erreur}}
]

ne doit être considérée comme établie sans modèle physique intermédiaire et validation indépendante.

---

## 1.4 Audit scientifique : corrections et points ouverts

Cette section consolide les incohérences documentaires identifiées entre les différentes descriptions du projet.

Le principe retenu est de privilégier la formulation prudente tant que l'implémentation réelle n'a pas été vérifiée.

## A. Cohérence documentaire

Les affirmations suivantes doivent être considérées comme des points à vérifier dans le code avant publication définitive :

Point| Affirmation à éviter sans vérification| Formulation retenue
DRAG / "pulse_calibration"| « Implémenté »| Non établi sans vérification
Fonction de Wigner| « Présente »| Non établie sans vérification
Tolérance du bilan| (r\leq10^{-15})| (r\leq10^{-10}) dans le périmètre documentaire
Chirurgie de réseau| « Protocole complet »| Prototype numérique / Monte-Carlo
Kerr-Cat| « Couplé »| Branche indépendante
Dérivées en phase| Ordre 4 ou Fock (N=35)| Stencil central avec conditions de Dirichlet, selon le solveur documenté

Les versions documentaires contradictoires doivent être supprimées ou harmonisées après vérification du dépôt.

---

## B. Correction du Pauli twirling de l'amplitude damping

Pour un canal d'amplitude damping de paramètre

[
\gamma=1-e^{-\tau/T_1},
]

la représentation PTM possède les valeurs propres :

[
\lambda_X=\lambda_Y=\sqrt{1-\gamma},
]

[
\lambda_Z=1-\gamma.
]

Après Pauli twirling, les probabilités sont :

[
p_X=p_Y=\frac{\gamma}{4},
]

[
p_Z=
\frac{2-\gamma-2\sqrt{1-\gamma}}{4},
]

et

[
p_I=
\frac{2-\gamma+2\sqrt{1-\gamma}}{4}.
]

Elles vérifient :

[
p_I+p_X+p_Y+p_Z=1.
]

Pour (\gamma\ll1) :

[
p_X=p_Y\simeq\frac{\gamma}{4},
]

[
p_Z\simeq\frac{\gamma^2}{16}.
]

Ainsi, la relation

[
p_Z=\frac{\gamma}{2}
]

est incorrecte pour ce twirling, tout comme

[
p_I=1-\gamma.
]

L'amplitude damping initial n'est pas lui-même un canal de Pauli. Le twirling constitue une réduction du canal à une représentation diagonale de Pauli.

Référence principale :

- J. Ghosh, A. G. Fowler, M. R. Geller, 2012, Surface code with decoherence: An analysis of three superconducting architectures, Physical Review A 86, 062318. DOI : 10.1103/PhysRevA.86.062318.

Test numérique requis

Pour

[
\gamma\in
{10^{-4},10^{-2},0.3,0.9},
]

le test doit comparer les probabilités calculées aux expressions analytiques ci-dessus et vérifier :

[
\left|
p_I+p_X+p_Y+p_Z-1
\right|
]

à une tolérance numérique explicitement documentée.

---

C. Bilan de quasiparticules : portée du test

Si le résidu

[
R_{qp}

x_{qp}(t+\Delta t)
-x_{qp}(t)
-P
+L
]

est calculé après avoir défini (x_{qp}(t+\Delta t)) par exactement la même équation de bilan, alors :

[
R_{qp}=0
]

est essentiellement une conséquence de la construction numérique.

Ce test vérifie alors principalement l'arithmétique du bilan et non la validité physique de la dynamique.

Il faut donc distinguer :

1. fermeture algébrique du bilan ;
2. exactitude de l'intégrateur ;
3. validité du modèle physique.

Une validation indépendante devrait comparer le modèle à :

- une solution analytique d'un cas simple ;
- ou un intégrateur de référence indépendant ;
- ou une solution numérique obtenue avec une méthode et un pas différents.

La tolérance (10^{-15}) ne doit pas être présentée comme une précision physique. En précision flottante IEEE 754 double, l'epsilon machine est d'environ

[
2.22\times10^{-16},
]

et les erreurs d'arrondi, de conditionnement et d'accumulation rendent une tolérance de l'ordre de (10^{-15}) potentiellement excessive sur une intégration longue.

La tolérance actuellement retenue dans le périmètre documentaire est donc :

[
|R_{qp}|\leq10^{-10},
]

sauf justification numérique spécifique.

---

D. Comptage des qubits dans la chirurgie 21Q

Le terme « 21 qubits » doit être défini explicitement.

Un patch de code de surface tourné de distance (d=3) est généralement décrit avec :

[
9\ \text{qubits de données}
+
8\ \text{qubits ancillas}

17\ \text{qubits}.
]

Le comptage de 21 qubits ne peut donc pas être présenté simplement comme :

[
2\times9+3=21
]

sans préciser que les 8 ancillas de syndrome de chaque patch ne sont pas représentées dans ce comptage.

Le prototype doit donc préciser ce que représentent les 21 qubits :

- qubits physiques effectivement simulés ;
- qubits de données ;
- ancillas d'interface ;
- ancillas sélectionnées ;
- ou abstraction réduite de la chirurgie.

La référence de Horsman et al. est pertinente pour le protocole de lattice surgery, mais ne doit pas être utilisée comme justification directe du comptage spécifique de 21 qubits.

Référence :

- D. Horsman, A. G. Fowler, S. J. Devitt, R. Van Meter, 2012, Surface code quantum computing by lattice surgery, New Journal of Physics 14, 123011. DOI : 10.1088/1367-2630/14/12/123011.

---

E. Conventions à unifier

Hamiltonien

La convention retenue est :

[
H=
4E_C\hat n^2
+
\frac{E_L}{2}
(\hat\phi-\phi_{\mathrm{ext}})^2

E_J\cos(\hat\phi).
]

avec :

[
[\hat\phi,\hat n]=i.
]

et

[
\hat n=-i\frac{\partial}{\partial\phi}.
]

Le signe du terme Josephson doit rester identique dans l'ensemble du projet.

Toute convention alternative doit être accompagnée d'une transformation explicite des variables.

Métrique de Fubini–Study

Pour un état pur normalisé :

[
|\psi(\Phi)\rangle,
]

une estimation par différence finie peut utiliser :

[
g_{\Phi\Phi}^{FS}
\simeq
\frac{
1-
|\langle\psi(\Phi)|\psi(\Phi+\delta\Phi)\rangle|^2
}
{\delta\Phi^2}.
]

Le README doit préciser :

- l'état utilisé ;
- le paramètre varié ;
- le pas (\delta\Phi) ;
- la convention d'unité de flux ;
- la méthode de convergence.

Quasiparticules

Le préfacteur du taux de quasiparticules doit être confronté à la formulation exacte de la littérature et aux conventions utilisées pour (x_{qp}), (E_J), (\Delta), (f_{01}) et les éléments de matrice.

Référence principale :

- I. M. Pop et al., 2014, Coherent suppression of electromagnetic dissipation due to superconducting quasiparticles, Nature 508, 369–372.

Kerr-Cat

Référence principale pour le contexte Kerr + pompage à deux photons :

- S. Puri, S. Boutin, A. Blais, 2017, Engineering the quantum states of light in a Kerr-nonlinear resonator by two-photon driving, npj Quantum Information 3, 18.

---

F. Hygiène du dépôt

Les noms de fichiers devraient suivre une convention Python stable :

- minuscules ;
- underscores ;
- pas d'espaces ;
- pas de versions dans les noms de fichiers.

Par exemple :

NOYAU QUANTIQUE V52.py

devrait devenir un nom du type :

noyau_quantique.py

Les versions doivent être gérées par Git, les tags, les releases ou les métadonnées du projet plutôt que par les noms des modules.

L'arborescence documentée doit également rester synchronisée avec le dépôt réel :

test/
tests/
notebooks/
results/
run_pipeline.py

ne doivent être documentés que s'ils existent effectivement dans le commit concerné.

Une CI GitHub Actions devrait exécuter au minimum :

python -m pytest -q

dans un environnement propre.

---

G. Références scientifiques principales

Sujet| Référence| Statut
Amplitude damping / Pauli twirling| Ghosh, Fowler, Geller (2012), PRA 86, 062318| Référence identifiée
Lattice surgery| Horsman et al. (2012), NJP 14, 123011| Référence identifiée
Quasiparticules / dissipation| Pop et al. (2014), Nature 508, 369–372| Référence identifiée
Kerr-Cat| Puri, Boutin, Blais (2017), npj QI 3, 18| Référence identifiée

Ces références soutiennent les cadres théoriques correspondants. Elles ne valident pas automatiquement les implémentations numériques particulières de Quantum-Lab.

---

H. Interface de recherche

L'interface :

[
g_{FS}
\quad ? \quad
\text{canal de bruit}
]

reste une question de recherche.

Le calcul de (g_{FS}) fournit une information géométrique sur la variation d'un état quantique par rapport à un paramètre.

Il ne fournit pas directement :

[
T_1,
\qquad
T_\phi,
\qquad
\Gamma,
\qquad
p_X,
\qquad
p_Y,
\qquad
p_Z.
]

Une chaîne physique plus complète devrait introduire explicitement les intermédiaires pertinents, par exemple :

[
g_{FS}
\rightarrow
\text{susceptibilité ou observable couplée}
\rightarrow
S_\Phi(\omega)
\rightarrow
\Gamma_\phi
\rightarrow
p_Z.
]

Tant qu'une telle relation n'est pas définie, dérivée et confrontée à une référence indépendante, "fs_coupling_gamma" reste un paramètre phénoménologique.

Toute relation entre géométrie quantique et erreur doit donc être présentée comme une hypothèse de travail lorsqu'elle n'est pas établie par le modèle.

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

Le point d'interrogation désigne une interface de recherche.

La géométrie quantique et le modèle de bruit peuvent être calculés séparément. Leur couplage nécessite une relation physique ou phénoménologique explicite.

Le calcul de (g_{FS}) ne suffit pas à déduire une probabilité d'erreur.

---

## 2.2 Modulation spatiale du substrat

Le module "fluxon_engine.py" décrit une modulation spatiale paramétrique du substrat.

Chaque qubit (i) reçoit un décalage de flux :

[
\Delta\Phi_i.
]

Ce décalage peut être pondéré par une enveloppe spatiale, par exemple gaussienne, centrée autour d'un point doux défini dans la convention du modèle.

La chaîne de calcul locale peut être représentée comme :

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

---

2.3 Noyau dynamique conservatif

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

Le résidu du bilan est :

[
R_{qp}

x_{qp}(t+\Delta t)

x_{qp}(t)

P
+
L.
]

Une tolérance numérique documentaire est :

[
|R_{qp}|\leq10^{-10}.
]

Cette tolérance concerne la fermeture numérique du bilan défini par le modèle. Elle ne démontre pas que la dynamique représente quantitativement les quasiparticules d'un circuit réel.

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

Le modèle utilise la convention :

[
H=
4E_C\hat n^2
+
\frac{E_L}{2}
(\hat\phi-\phi_{\mathrm{ext}})^2

E_J\cos(\hat\phi).
]

Les opérateurs satisfont :

[
[\hat\phi,\hat n]=i.
]

Dans la représentation de phase :

[
\hat n=-i\frac{\partial}{\partial\phi}.
]

Les conditions aux limites numériques sont de type Dirichlet.

La discrétisation en phase utilise un stencil central.

Les paramètres énergétiques peuvent être exprimés en unités de fréquence, par exemple :

[
E/h
\quad\text{en GHz}.
]

La convention de conversion entre énergie et fréquence doit rester cohérente dans l'ensemble du solveur.

---

## 3.2 Quantités extraites

Le solveur permet d'obtenir, selon les fonctionnalités effectivement présentes dans l'implémentation :

- les niveaux d'énergie ;
- les états propres ;
- les fréquences de transition ;
- les éléments de matrice ;
- les dipôles de charge ;
- les dipôles de phase ;
- les observables nécessaires aux estimations de relaxation ;
- les sensibilités fréquentielles ;
- les quantités nécessaires au calcul géométrique.

Les taux de relaxation et de déphasage dépendent des hypothèses de bruit et des paramètres physiques introduits.

Ils ne sont pas des constantes intrinsèques déterminées par le seul spectre.

---

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
]

[
\langle0|\hat n|1\rangle,
]

[
\langle0|\hat\phi|1\rangle,
]

[
\frac{\partial f_{01}}{\partial\Phi},
]

et

[
g_{FS}.
]

La convergence locale ne garantit pas la convergence globale dans tout l'espace des paramètres.

Une vérification de convergence doit également contrôler l'hermiticité numérique :

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

---

## 4.2 Sensibilité au flux

La dérivée de fréquence relie une fluctuation de flux à une fluctuation de fréquence.

Au premier ordre :

[
\delta f_{01}
\simeq
\frac{\partial f_{01}}{\partial\Phi}
\delta\Phi.
]

Au second ordre :

[
\delta f_{01}
\simeq
\frac{\partial f_{01}}{\partial\Phi}
\delta\Phi
+
\frac12
\frac{\partial^2f_{01}}{\partial\Phi^2}
(\delta\Phi)^2.
]

Le premier terme décrit la sensibilité linéaire.

Le second introduit la contribution de la courbure fréquentielle.

Cette distinction devient importante au voisinage d'un point doux, où la dérivée première peut être faible alors que la dérivée seconde reste non nulle.

---

## 4.3 Quantum geometric tensor et métrique de Fubini–Study

Le quantum geometric tensor (QGT) décrit la réponse géométrique d'un état quantique à une variation de paramètres.

Pour un état pur normalisé

[
|\psi(\lambda)\rangle,
]

sa forme projetée peut s'écrire :

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

Le paramètre "fs_coupling_gamma" demeure phénoménologique tant qu'une relation physique indépendante et testable ne le justifie.

---

# 5. Quasiparticules froides

## 5.1 Modèle de la petite jonction

Le modèle de relaxation par quasiparticules concerne la petite jonction Josephson du Fluxonium, dans l'approximation des quasiparticules froides.

Le taux descendant est représenté sous la forme :

[
\Gamma_{\mathrm{qp},\downarrow}

\frac{8}{\pi}
\frac{E_J}{\hbar}
x_{qp}
\sqrt{\frac{2\Delta}{h f_{01}}}
\left|
\left\langle0\left|
\sin\left(
\frac{\hat\phi-\phi_{\mathrm{ext}}}{2}
\right)
\right|1\right\rangle
\right|^2.
]

Cette expression doit être interprétée avec les conventions de normalisation de (x_{qp}), de (E_J), de (\Delta) et des opérateurs utilisées par l'implémentation.

La vérification dimensionnelle et la comparaison à la formulation de référence sont nécessaires avant toute interprétation quantitative.

Référence principale :

Pop et al., Nature 508, 369–372 (2014).

---

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

---

## 5.3 Portée

Le taux calculé est une contribution de relaxation associée à la petite jonction dans les hypothèses du modèle.

Il ne représente pas automatiquement le taux total de relaxation par quasiparticules d'un Fluxonium réel.

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

---

## 6.2 Amplitude damping

Pour un canal d'amplitude damping de paramètre :

[
\gamma

1-e^{-\tau/T_1},
]

les valeurs propres de la partie dissipative de la PTM sont :

[
\lambda_X=\lambda_Y=\sqrt{1-\gamma},
]

[
\lambda_Z=1-\gamma.
]

Après Pauli twirling, les probabilités sont :

[
p_X=p_Y=\frac{\gamma}{4},
]

[
p_Z=
\frac{
2-\gamma-2\sqrt{1-\gamma}
}{4},
]

[
p_I=
\frac{
2-\gamma+2\sqrt{1-\gamma}
}{4}.
]

Elles vérifient :

[
p_I+p_X+p_Y+p_Z=1.
]

Pour (\gamma\ll1) :

[
p_X=p_Y\simeq\frac{\gamma}{4},
]

[
p_Z\simeq\frac{\gamma^2}{16}.
]

Ces probabilités décrivent le canal de Pauli obtenu après twirling.

Elles ne signifient pas que l'amplitude damping initial est lui-même un canal de Pauli.

Le twirling constitue une réduction du canal quantique original.

---

## 6.3 Déphasage pur

Pour un déphasage pur de paramètre :

[
\lambda_Z

\frac{1-e^{-\tau/T_\phi}}{2},
]

la représentation de Pauli est :

[
p_Z=\lambda_Z,
]

[
p_I=1-\lambda_Z.
]

Cette expression suppose la convention de déphasage pur retenue par le module.

Il faut distinguer (T_\phi) de (T_2).

Lorsque les mécanismes considérés sont ceux d'une relaxation d'amplitude et d'un déphasage pur markovien :

[
\frac{1}{T_2}

\frac{1}{2T_1}
+
\frac{1}{T_\phi}.
]

---

## 6.4 Composition PTM

La matrice de transfert de Pauli représente l'action du canal sur les opérateurs de Pauli.

Pour les canaux de Pauli :

[
R_{\mathcal E_2\circ\mathcal E_1}

R_{\mathcal E_2}
R_{\mathcal E_1}.
]

Lorsque les matrices considérées sont diagonales dans la base de Pauli, la composition se réduit au produit de leurs composantes diagonales.

Le canal de Pauli effectif est ensuite représenté par :

[
\mathcal E_{\mathrm{Pauli}}

{p_I,p_X,p_Y,p_Z}.
]

Précision épistémique :

le twirling est une opération de réduction du canal.

Il supprime certaines composantes cohérentes et non diagonales.

L'exactitude algébrique de la composition PTM ne signifie donc pas que le canal twirlé conserve toute l'information du canal quantique initial.

---

## 6.5 Bruit quasi statique et bruit coloré

Le canal de Pauli effectif ne reproduit pas intégralement :

- les corrélations temporelles entre cycles ;
- la dynamique du bruit quasi statique (1/f) ;
- les effets non markoviens ;
- la non-exponentialité de certaines enveloppes de cohérence ;
- les contributions de la courbure (\partial^2f_{01}/\partial\Phi^2) ;
- les corrélations entre erreurs d'amplitude et de phase.

Le bruit quasi statique ne doit donc pas être automatiquement assimilé à un taux de déphasage markovien constant.

Une réduction en canal effectif doit préciser :

- la durée considérée ;
- la distribution du bruit ;
- les hypothèses temporelles ;
- la convention spectrale ;
- la méthode de twirling ;
- les corrélations éventuellement négligées.

---

# 7. Qubit bosonique Kerr-Cat

## 7.1 Modèle dynamique

Le module "kerr_cat_simulator.py" étudie une dynamique bosonique de type Kerr-Cat.

Le modèle comprend notamment :

- le pompage à deux photons ;
- un Hamiltonien effectif de Kerr ;
- une équation maîtresse de Lindblad ;
- une intégration numérique ;
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

\frac12
\left{
L^\dagger L,\rho
\right}.
]

La signification physique des opérateurs (L_j), leurs unités et leurs taux doivent être définis par les paramètres du modèle.

---

## 7.2 Indépendance du modèle

Le simulateur Kerr-Cat est une branche bosonique indépendante du modèle Fluxonium.

Il ne constitue pas une simulation couplée complète entre un Fluxonium et un Kerr-Cat.

Les résultats de fidélité et de leakage sont conditionnels :

- au Hamiltonien ;
- aux dissipateurs ;
- à la troncature de Fock ;
- à la méthode d'intégration ;
- aux paramètres choisis.

Toute fonctionnalité de calibration qui n'est pas effectivement opérationnelle doit être documentée comme telle.

Référence de cadre théorique :

S. Puri, S. Boutin, A. Blais (2017), Engineering the quantum states of light in a Kerr-nonlinear resonator by two-photon driving, npj Quantum Information 3, 18.

---

# 8. Correction d'erreurs quantique et chirurgie de réseau

## 8.1 Conversion des taux en probabilités locales

Le module "error_correction.py" utilise des probabilités d'erreur locales associées à une durée de cycle :

[
\tau_{\mathrm{cycle}}.
]

Les paramètres de bruit peuvent être convertis en :

[
p_{X,i},
\qquad
p_{Y,i},
\qquad
p_{Z,i}.
]

Cette conversion dépend du canal retenu et des approximations de réduction.

Elle ne constitue pas une description complète :

- des erreurs corrélées ;
- des erreurs cohérentes ;
- des erreurs temporellement persistantes ;
- des erreurs de mesure ;
- des corrélations entre qubits.

---

## 8.2 Code de répétition de phase

Le modèle de répétition de phase considère une distance impaire (d).

Les états logiques sont :

[
|0_L\rangle

|+\rangle^{\otimes d},
]

[
|1_L\rangle

|-\rangle^{\otimes d}.
]

Pour un modèle indépendant d'erreurs de phase de probabilité (p_Z), la probabilité d'une majorité d'erreurs est :

[
P_L^{(Z)}

\sum_{k=(d+1)/2}^{d}
\binom{d}{k}
p_Z^k
(1-p_Z)^{d-k}.
]

Cette expression suppose des erreurs indépendantes et identiquement distribuées.

Elle ne décrit pas automatiquement :

- les erreurs corrélées ;
- les erreurs de mesure ;
- les corrélations espace-temps ;
- les processus cohérents.

---

## 8.3 Budget d'erreur logique

Le budget logique agrège les contributions d'erreurs selon les hypothèses du modèle.

Lorsque plusieurs contributions sont combinées par une formule d'indépendance, cette hypothèse doit être explicitement indiquée.

Une combinaison de probabilités marginales ne doit pas être interprétée comme une distribution conjointe exacte en l'absence de vérification de l'indépendance.

---

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

La configuration annoncée utilise une distance :

[
d=3
]

et un total documentaire de :

[
21\ \text{qubits}.
]

Ce nombre doit être interprété selon la définition exacte du prototype.

Un patch de code de surface tourné (d=3) standard est généralement décrit avec :

[
9\ \text{qubits de données}
+
8\ \text{ancillas}

17\ \text{qubits}.
]

Par conséquent, le nombre 21 doit être documenté explicitement : il peut correspondre à une architecture réduite, à des qubits d'interface supplémentaires ou à une autre convention de comptage.

La chirurgie est un prototype numérique.

Elle ne doit pas être présentée comme :

- une simulation complète au niveau des stabilisateurs ;
- une simulation complète des circuits de portes ;
- une modélisation exhaustive des erreurs de lecture ;
- une démonstration de seuil QEC ;
- une validation d'une architecture matérielle ;
- une preuve de correction d'erreurs universelle.

---

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

L'organisation documentée sépare les modules de calcul, les orchestrateurs et les tests.

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

Cette arborescence doit être considérée comme une structure documentaire à maintenir synchronisée avec les fichiers effectivement suivis dans le dépôt.

Les noms de modules, les exports et les points d'entrée doivent être vérifiés contre le commit concerné.

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

La présence d'un module dans cette table ne constitue pas, à elle seule, une certification de son niveau d'implémentation.

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

Les dépendances doivent rester conformes aux déclarations de "requirements.txt" et "pyproject.toml".

---

# 12. Exécution

Les points d'entrée documentés sont :

Pipeline standard

python run_pipeline.py

Orchestrateur du pont

python pont_lourd_unifie.py

Ces commandes correspondent aux points d'entrée documentés.

Leur opérationnalité doit être établie par une exécution effective dans un environnement propre.

La simple présence d'un fichier ou d'une fonction "main()" ne constitue pas une preuve d'exécution réussie.

---

# 13. Tests

Lancer la suite :

python -m pytest test/ -v

ou :

python -m pytest -q

Les tests documentés couvrent notamment :

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

Ils ne remplacent pas :

- une validation analytique indépendante ;
- une comparaison à une référence numérique ;
- une calibration expérimentale ;
- une validation expérimentale.

Aucun résultat de test ne doit être annoncé comme réussi sans exécution effective sur le commit concerné.

---

# 14. Limites scientifiques

Domaine| Limite actuelle
Fluxonium| Résultats dépendants des paramètres, de la discrétisation et des hypothèses de bruit
Géométrie quantique| (g_{FS}) ne détermine pas directement les probabilités d'erreur
Quasiparticules| Modèle limité à la contribution froide de la petite jonction
Substrat| Bilan effectif, sans dynamique microscopique complète
Bruit| Canal de Pauli réduit, avec perte d'information liée au twirling
Bruit quasi statique| Corrélations temporelles non entièrement représentées par un canal markovien
Kerr-Cat| Branche indépendante du Fluxonium
Calibration| Contrôle DRAG non établi comme fonctionnalité opérationnelle sans vérification
QEC| Modèles effectifs et prototypes
Décodage| Pas de démonstration d'un MWPM espace-temps complet
Chirurgie| Prototype numérique, non simulation stabilizer-level complète
Expérience| Aucune validation expérimentale intégrée

---

# 15. Feuille de route de validation

Les étapes suivantes sont des objectifs de recherche et de validation.

P0 — Cohérence logicielle

- vérifier les imports de tous les modules ;
- aligner les exports de "src/__init__.py" sur les symboles réellement définis ;
- vérifier la cohérence entre les API et les tests ;
- éliminer toute commande shell présente accidentellement dans les fichiers Python de test ;
- exécuter les tests dans un environnement propre ;
- documenter les résultats d'exécution.

---

P1 — Solveur Fluxonium

- vérifier l'hermiticité de l'Hamiltonien ;
- vérifier les opérateurs discrets ;
- étudier la convergence en taille de grille ;
- étudier la convergence en étendue de domaine ;
- comparer les spectres à une méthode indépendante ;
- vérifier les éléments de matrice ;
- vérifier la stabilité des dérivées de fréquence.

---

P2 — Géométrie quantique

- vérifier la normalisation des états ;
- contrôler l'invariance de jauge ;
- étudier la convergence en pas de différence finie ;
- comparer les méthodes de calcul du QGT ;
- distinguer les métriques locales des grandeurs physiques de bruit ;
- identifier précisément l'état et le paramètre utilisés pour chaque métrique.

---

P3 — Bruit et canaux

- documenter les conventions spectrales ;
- distinguer bruit blanc, bruit coloré et bruit quasi statique ;
- comparer les canaux à une évolution de référence ;
- quantifier l'information perdue par le twirling ;
- vérifier la conversion des taux en probabilités par cycle ;
- tester analytiquement le Pauli twirling de l'amplitude damping ;
- vérifier les conventions de (T_1), (T_\phi) et (T_2).

---

P4 — QEC et décodage

- vérifier les probabilités logiques par énumération exhaustive pour les petits codes ;
- séparer erreurs de données et erreurs de mesure ;
- définir le graphe des événements de syndrome ;
- construire un graphe espace-temps ;
- comparer le décodeur à une implémentation de référence ;
- documenter les frontières et les observables logiques.

---

P5 — Chirurgie de réseau

- définir précisément les qubits de données et ancillas ;
- définir les stabilisateurs initiaux et finaux ;
- expliciter le comptage des 21 qubits ;
- décrire les circuits de préparation, merge et split ;
- modéliser les erreurs de portes et de lecture ;
- vérifier le suivi du cadre de Pauli ;
- vérifier les observables logiques ;
- étudier plusieurs distances avant toute affirmation de seuil.

---

P6 — Intégration physique

- définir les hypothèses de couplage entre qubits ;
- relier les paramètres matériels aux canaux locaux ;
- établir une interface physique testable entre géométrie et bruit, si elle est retenue ;
- comparer les prédictions aux données expérimentales ;
- publier les incertitudes ;
- publier les domaines de validité ;
- distinguer les paramètres calibrés des paramètres libres.

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

# 17. Visualisation de trajectoire

Une visualisation telle que :

« Trajectoire — Édifice quantique »

peut être utilisée comme représentation graphique d'une trajectoire dynamique simulée.

Elle peut présenter simultanément :

- une quantité cumulée ;
- une concurrence ;
- une fidélité ;
- une pureté ;
- les populations des états logiques.

Cette visualisation doit être interprétée comme une représentation d'un sous-modèle dynamique.

Elle ne constitue pas, à elle seule, une démonstration du pipeline complet :

[
\text{Fluxonium}
\rightarrow
\text{géométrie}
\rightarrow
\text{bruit}
\rightarrow
\text{QEC}.
]

Elle ne constitue pas non plus une validation expérimentale.

---

# 18. Positionnement du projet

Quantum-Lab étudie une question centrale :

«Comment relier de manière traçable les propriétés microscopiques d'un circuit supraconducteur, la géométrie de ses états quantiques, ses canaux de bruit effectifs et les performances d'une architecture de correction d'erreurs ?»

L'intérêt du projet réside dans la construction d'interfaces explicites entre ces niveaux.

La géométrie quantique, les modèles de dissipation, les canaux de Pauli et les mécanismes de correction d'erreurs sont des descriptions distinctes.

Leur articulation exige :

- des hypothèses identifiables ;
- des transformations mathématiques explicites ;
- des unités cohérentes ;
- des domaines de validité ;
- des validations indépendantes.

Le point central du projet n'est donc pas de supposer que ces descriptions sont déjà équivalentes.

Il consiste à déterminer quelles transformations permettent éventuellement de les relier sans perdre leur statut épistémique propre.

Dans cette perspective :

[
\boxed{
\text{modèle}
\rightarrow
\text{réduction}
\rightarrow
\text{canal effectif}
\rightarrow
\text{QEC}
\rightarrow
\text{validation}
}
]

constitue une chaîne de transformations à contrôler, et non une identité physique présupposée.

La géométrie quantique peut fournir une information supplémentaire sur la structure locale de l'espace des états. Le canal de bruit décrit une autre couche de phénomènes. La relation entre ces deux descriptions reste une question de recherche tant qu'elle n'est pas établie par une relation physique explicite.

Quantum-Lab constitue ainsi un environnement de recherche numérique pour :

- construire ces interfaces ;
- tester leur cohérence ;
- mesurer les effets des réductions ;
- identifier les hypothèses nécessaires ;
- distinguer les résultats démontrés des paramètres phénoménologiques ;
- préparer, lorsque cela devient possible, une confrontation avec des données expérimentales.

Quantum-Lab — Modéliser les interfaces, contrôler les réductions, éprouver les correspondances.
