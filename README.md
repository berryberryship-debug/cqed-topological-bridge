# Quantum-Lab — Fluxonium–Lattice Surgery Bridge

## Pipeline multi-échelle : Fluxonium, bruit quantique, QEC biaisée, Kerr-Cat et chirurgie de réseau

## Quantum-Lab est un environnement de modélisation numérique consacré à l’étude des interfaces entre les circuits supraconducteurs, la géométrie quantique des états, les canaux de bruit et la correction d’erreurs quantique.

## Le projet relie plusieurs niveaux de description :

SUBSTRAT
   │
   ├── modulation paramétrique du flux
   └── densité effective de quasiparticules
             │
             ▼
FLUXONIUM
   │
   ├── Hamiltonien
   ├── spectre et états propres
   ├── T1 · Tφ
   ├── df01/dΦ
   └── QGT · gFS
             │
       ┌─────┴─────┐
       ▼           ▼
   GÉOMÉTRIE     BRUIT
   QUANTIQUE       │
       │            ├── amplitude damping
       │            ├── déphasage pur
       │            └── PTM
       │                 │
       │                 ▼
       │            PAULI TWIRLING
       │            pX · pY · pZ
       │                 │
       └────── ? ───────┘
                     │
                     ▼
                 QEC
                     │
                     ├── répétition de phase
                     ├── budget d’erreur
                     ├── décodage pondéré
                     └── chirurgie de réseau

Le simulateur bosonique Kerr-Cat constitue une branche complémentaire. Il possède sa propre dynamique et n’est pas actuellement intégré à une simulation couplée Fluxonium–Kerr-Cat.

---

# 1. Statut scientifique et périmètre épistémique

Quantum-Lab implémente un pipeline de modélisation numérique multi-échelle.

Le dépôt ne constitue ni une démonstration expérimentale, ni une dérivation physique complète de bout en bout, ni une validation d’une architecture matérielle réelle.

Le projet distingue explicitement :

- les quantités calculées à partir d’un modèle physique ;
- les paramètres phénoménologiques ;
- les approximations de réduction ;
- les modèles numériques effectifs ;
- les prototypes de correction d’erreurs ;
- les hypothèses encore ouvertes ;
- les éléments non validés expérimentalement.

## 1.1 Niveaux de preuve

Niveau| Signification
Modèle analytique| Relation définie à partir d’hypothèses physiques ou mathématiques
Calcul numérique| Résultat produit par une méthode de calcul déterminée
Validation numérique| Vérification d’invariants, de convergence ou comparaison à une référence
Prédiction expérimentale| Résultat quantitatif confronté à des mesures matérielles
Validation expérimentale| Reproduction indépendante et vérification sur dispositif réel

Une validation numérique locale ne constitue pas automatiquement une validation de l’ensemble de la chaîne.

Le pipeline est conçu pour étudier la cohérence d’une architecture de modélisation et les interfaces entre niveaux de description.

---

# 2. Périmètre fonctionnel

## 2.1 Fluxonium

Le projet comprend les éléments suivants :

- construction de l’Hamiltonien Fluxonium en base de phase ;
- diagonalisation numérique du spectre ;
- calcul des niveaux d’énergie ;
- calcul des états propres ;
- calcul des fréquences de transition ;
- calcul d’éléments de matrice de charge et de phase ;
- estimation de quantités de cohérence à partir de paramètres fournis ;
- calcul de la sensibilité fréquentielle au flux ;
- calcul de quantités géométriques associées aux états propres.

## 2.2 Géométrie quantique

Le projet explore :

- le quantum geometric tensor (QGT) ;
- la métrique de Fubini–Study ;
- la sensibilité géométrique des états propres ;
- les variations de cette métrique avec le paramètre de flux.

La géométrie quantique est ici utilisée comme niveau descriptif supplémentaire.

Elle ne constitue pas, en elle-même, un modèle de dissipation.

## 2.3 Substrat et quasiparticules

Le modèle comprend :

- une modulation spatiale paramétrique du flux ;
- un modèle effectif de densité de quasiparticules x_{qp} ;
- un bilan numérique de production et de pertes ;
- une contribution effective des quasiparticules froides de la petite jonction Josephson.

Ce modèle ne constitue pas une simulation microscopique complète de la dynamique des quasiparticules.

## 2.4 Canaux de bruit

Le projet comprend :

- amplitude damping ;
- déphasage pur ;
- représentation en matrice de transfert de Pauli (PTM) ;
- Pauli twirling ;
- conversion en probabilités effectives p_X,p_Y,p_Z ;
- composition de canaux diagonaux dans la représentation PTM.

## 2.5 Correction d’erreurs quantique

Le périmètre comprend :

- code de répétition de phase ;
- modèle de bruit biaisé ;
- budget d’erreur logique ;
- probabilités locales d’erreur ;
- prototype de décodage pondéré ;
- prototype de chirurgie de réseau.

La chirurgie de réseau est traitée comme un prototype numérique, et non comme une implémentation complète d’un protocole de surface code au niveau des stabilisateurs et des circuits physiques.

## 2.6 Kerr-Cat

Le projet contient une branche indépendante consacrée au qubit bosonique Kerr-Cat :

- Hamiltonien effectif de Kerr ;
- pompage à deux photons ;
- équation maîtresse de Lindblad ;
- intégration numérique ;
- estimation de fidélité ;
- estimation du leakage.

Cette branche n’est pas actuellement couplée au moteur Fluxonium dans une simulation physique commune.

---

# 3. Éléments non implémentés ou non établis

Les éléments suivants ne doivent pas être présentés comme des fonctionnalités achevées sans vérification indépendante du code correspondant :

- contrôle DRAG ;
- calibration physique complète d’impulsions ;
- fonction de Wigner ;
- dynamique microscopique complète des quasiparticules ;
- diffusion microscopique des quasiparticules ;
- piégeage microscopique ;
- recombinaison microscopique complète ;
- quasiparticules dans le réseau inductif ;
- asymétrie de gap supraconducteur ;
- couplage physique complet entre plusieurs Fluxonium ;
- simulation complète de chirurgie de réseau au niveau stabilisateur/circuit ;
- validation expérimentale intégrée ;
- démonstration d’un seuil de correction d’erreurs.

La présence d’un fichier ou d’une API ne suffit pas à établir qu’une fonctionnalité est opérationnelle.

Une fonctionnalité est considérée comme opérationnelle uniquement après vérification de son implémentation et, lorsque pertinent, exécution des tests correspondants.

---

# 4. Paramètres phénoménologiques et interfaces ouvertes

Plusieurs paramètres constituent des réductions entre niveaux physiques.

Ils doivent être documentés avec :

- leur définition ;
- leurs unités ;
- leur domaine de validité ;
- leur origine ;
- leur méthode de calibration ;
- leur statut physique ou phénoménologique.

Les principaux paramètres concernés sont :

- "fs_coupling_gamma" ;
- les paramètres de réduction à deux niveaux ;
- la conversion des taux continus en probabilités de Pauli par cycle ;
- les modèles effectifs de bruit quasi statique ;
- les paramètres de couplage XX ;
- les paramètres de modulation spatiale du substrat ;
- les paramètres de densité effective de quasiparticules ;
- toute relation postulée entre géométrie quantique et probabilité d’erreur.

L’existence d’une relation numérique entre deux quantités ne suffit pas à établir qu’elle constitue une relation physique.

---

# 5. Architecture du pipeline

## 5.1 Vue générale

La chaîne principale suit une succession de transformations :

SUBSTRAT
   │
   ├── ΔΦ spatial
   └── xqp effectif
          │
          ▼
FLUXONIUM
   │
   ├── H(Φext)
   ├── spectre
   ├── états propres
   ├── T1 · Tφ
   ├── df01/dΦ
   └── QGT · gFS
          │
      ┌───┴────┐
      ▼        ▼
 GÉOMÉTRIE   BRUIT
 QUANTIQUE     │
                ├── amplitude damping
                ├── déphasage
                └── PTM
                     │
                     ▼
                PAULI TWIRLING
                     │
                     ├── pX
                     ├── pY
                     └── pZ
                     │
                     ▼
                    QEC
                     │
                     ├── code de répétition
                     ├── budget logique
                     ├── décodage
                     └── chirurgie

L’interface :

gFS  ─────────── ? ─────────── bruit

constitue une question de recherche.

Le calcul de g_{FS} ne fournit pas automatiquement une probabilité d’erreur.

Tant qu’aucune relation physique indépendante et testable n’est établie, "fs_coupling_gamma" doit rester qualifié de paramètre phénoménologique.

---

# 6. Modulation spatiale du substrat

Le module "fluxon_engine.py" décrit une modulation spatiale paramétrique du substrat.

Chaque qubit i reçoit un décalage de flux :

[
\Delta\Phi_i.
]

Ce décalage peut être pondéré spatialement par une enveloppe paramétrique autour d’un point de fonctionnement choisi.

La chaîne locale est :

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

Cette chaîne permet d’étudier l’effet d’une inhomogénéité paramétrique sur les propriétés locales des qubits.

Elle ne constitue pas :

- un modèle de fabrication ;
- une description complète des variations matérielles ;
- une simulation électromagnétique du dispositif ;
- une reconstruction microscopique du substrat.

---

# 7. Noyau de quasiparticules

Le module "fluxon_engine.py" maintient un modèle effectif de densité de quasiparticules hors équilibre :

[
x_{qp}.
]

Le bilan est formulé sous la forme :

[
x_{qp}(t+\Delta t)

x_{qp}(t)
+
P(t,\Delta t)

L(t,\Delta t),
]

où :

- P représente la production effective ;
- L représente les pertes effectives.

Le résidu est :

[
R_{qp}

x_{qp}(t+\Delta t)

x_{qp}(t)

P
+
L.
]

Une tolérance numérique de référence peut être utilisée :

[
|R_{qp}|\leq10^{-10}.
]

Cette tolérance doit être interprétée comme une tolérance de fermeture numérique du bilan défini par le modèle.

Elle ne démontre pas l’exactitude physique de la dynamique des quasiparticules.

7.1 Portée du test de bilan

Si x_{qp}(t+\Delta t) est calculé directement à partir de :

[
x_{qp}(t+\Delta t)

x_{qp}(t)+P-L,
]

alors le résidu est nul par construction, à l’erreur numérique près.

Le test vérifie alors principalement :

- la cohérence arithmétique ;
- les signes ;
- la conservation de la convention de bilan ;
- l’implémentation de l’expression utilisée.

Il ne constitue pas une validation indépendante de la dynamique.

Une validation plus forte doit comparer le modèle à :

- une solution analytique d’un cas simple ;
- ou un intégrateur indépendant ;
- ou une formulation numérique de référence.

La tolérance doit également être adaptée à la précision flottante et à la longueur de l’intégration.

---

# 8. Solveur Hamiltonien Fluxonium

## 8.1 Hamiltonien

Le Hamiltonien utilisé dans la convention du projet est :

[
H

4E_C\hat n^2
+
\frac{E_L}{2}
(\hat\phi-\phi_{\mathrm{ext}})^2

E_J\cos(\hat\phi).
]

Avec :

[
[\hat\phi,\hat n]=i.
]

Le paramètre de flux réduit est exprimé en radians :

[
\phi_{\mathrm{ext}}

2\pi\frac{\Phi_{\mathrm{ext}}}{\Phi_0}.
]

Les énergies peuvent être exprimées sous forme de fréquences :

[
E/h
]

notamment en GHz.

La convention énergétique doit rester identique dans les différents modules du projet.

## 8.2 Discrétisation

Le solveur travaille en base de phase sur une grille numérique finie.

Les conditions aux limites utilisées par l’implémentation sont de type Dirichlet.

La dérivation numérique repose sur un stencil central.

Cette discrétisation constitue une approximation du problème continu.

En particulier, sur une grille finie :

[
[\hat\phi,\hat n]=i
]

ne doit pas être interprété comme une identité numérique exacte à toutes les positions de grille.

La convergence doit donc être contrôlée lorsque :

- la taille de grille varie ;
- l’étendue du domaine varie ;
- le nombre d’états retenus varie ;
- les dérivées numériques varient.

---

# 9. Quantités extraites du solveur

Le solveur permet notamment d’obtenir :

- les niveaux d’énergie ;
- les états propres ;
- les fréquences de transition ;
- les éléments de matrice ;
- les éléments de matrice de charge ;
- les éléments de matrice de phase ;
- la sensibilité fréquentielle au flux ;
- les quantités nécessaires au calcul géométrique.

Les temps T_1 et T_\phi ne sont pas déduits automatiquement du seul spectre.

Lorsqu’ils sont fournis comme paramètres, le code les traite comme des paramètres de cohérence du modèle.

---

# 10. Convergence numérique du Fluxonium

La convergence doit être examinée selon plusieurs axes :

- taille de grille ;
- étendue du domaine en phase ;
- conditions aux limites ;
- précision des dérivées ;
- stabilité des états propres ;
- stabilité des éléments de matrice ;
- stabilité des fréquences de transition.

Les observables candidates comprennent :

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

[
g_{\Phi\Phi}^{FS}.
]

L’hermiticité numérique doit également être contrôlée :

[
|H-H^\dagger|.
]

La convergence d’une observable ne garantit pas nécessairement la convergence de toutes les autres.

---

# 11. Sensibilité au flux

La variation de la fréquence fondamentale avec le flux peut être décrite au premier ordre par :

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
\frac{\partial f_{01}}{\partial\Phi}\delta\Phi
+
\frac{1}{2}
\frac{\partial^2f_{01}}{\partial\Phi^2}
(\delta\Phi)^2.
]

Le premier terme décrit la sensibilité linéaire.

Le second décrit la contribution de la courbure fréquentielle.

Cette distinction devient importante au voisinage d’un point doux, où :

[
\frac{\partial f_{01}}{\partial\Phi}
]

peut être faible alors que :

[
\frac{\partial^2f_{01}}{\partial\Phi^2}
]

reste non nul.

---

# 12. Quantum geometric tensor et métrique de Fubini–Study

Pour un état pur normalisé :

[
|\psi(\lambda)\rangle,
]

le quantum geometric tensor peut être défini par :

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

Pour le flux :

[
g_{\Phi\Phi}^{FS}

\operatorname{Re}(Q_{\Phi\Phi}).
]

Cette métrique caractérise la sensibilité géométrique de l’état quantique à une variation du paramètre.

Elle ne constitue pas, à elle seule :

- un taux de relaxation ;
- un taux de déphasage ;
- une probabilité d’erreur ;
- un canal de bruit ;
- un couplage physique entre qubits.

12.1 Calcul par différence finie

Une estimation locale peut utiliser une quantité de la forme :

[
g_{\Phi\Phi}^{FS}
\approx
\frac{
1-
|\langle\psi(\Phi)|\psi(\Phi+\delta\Phi)\rangle|^2
}{
(\delta\Phi)^2
}.
]

Le calcul doit préciser :

- l’état considéré ;
- le pas \delta\Phi ;
- la méthode de calcul des états ;
- la gestion de la phase globale ;
- la convergence lorsque \delta\Phi\rightarrow0.

La métrique doit être calculée sur l’état explicitement défini par le modèle : état fondamental, état excité ou sous-espace logique.

---

# 13. Interface géométrie — bruit

Le projet distingue :

[
g_{FS}
]

d’un canal de bruit.

Il n’existe pas dans le pipeline actuel de relation physique universelle permettant d’écrire directement :

[
g_{FS}
\longrightarrow
p_{\mathrm{erreur}}.
]

Le paramètre :

fs_coupling_gamma

doit donc être interprété comme une hypothèse de couplage phénoménologique tant qu’une dérivation physique indépendante et testable n’est pas établie.

Une telle relation devrait notamment préciser :

1. la grandeur physique couplée ;
2. les unités ;
3. l’échelle temporelle ;
4. la dépendance en flux ;
5. le mécanisme microscopique ou effectif ;
6. les paramètres indépendants ;
7. une méthode de calibration ;
8. une prédiction falsifiable.

---

# 14. Quasiparticules froides et petite jonction

Le modèle de relaxation par quasiparticules concerne la petite jonction Josephson du Fluxonium dans l’approximation des quasiparticules froides.

Une expression de taux peut être écrite sous la forme :

[
\Gamma_{\mathrm{qp},\downarrow}

\frac{8}{\pi}
\frac{E_J}{\hbar}
x_{qp}
\sqrt{\frac{2\Delta}{hf_{01}}}
\left|
\left\langle
0
\left|
\sin
\left(
\frac{\hat\phi-\phi_{\mathrm{ext}}}{2}
\right)
\right|
1
\right\rangle
\right|^2.
]

L’expression doit être interprétée avec les conventions de normalisation utilisées pour :

- x_{qp} ;
- E_J ;
- \Delta ;
- f_{01} ;
- l’opérateur de phase.

Le préfacteur et les conventions de normalisation doivent être confrontés à la littérature de référence avant toute interprétation quantitative.

---

# 15. Hypothèses quasiparticulaires

Le modèle considéré repose notamment sur :

- quasiparticules froides ;
- régime hf_{01}\ll\Delta ;
- contribution de la petite jonction ;
- absence de taux montant QP dans le modèle considéré ;
- absence de contribution QP du réseau inductif ;
- absence de modélisation complète du piégeage ;
- absence de modélisation microscopique complète de la recombinaison ;
- absence de description complète des asymétries de gap.

Le taux obtenu représente donc une contribution de relaxation dans les hypothèses du modèle, et non le taux total de relaxation QP d’un Fluxonium réel.

---

# 16. Canal de bruit

Le module "noise_channel.py" construit un canal de bruit effectif sur une durée :

[
\tau.
]

Les mécanismes considérés sont notamment :

- amplitude damping ;
- déphasage pur ;
- représentation PTM ;
- Pauli twirling ;
- composition de canaux.

Les taux physiques sont convertis en paramètres de canal selon les conventions explicites du modèle.

---

# 17. Amplitude damping et Pauli twirling

Pour :

[
\gamma

1-e^{-\tau/T_1},
]

la matrice de transfert de Pauli du canal d’amplitude damping possède, dans la convention utilisée, les valeurs propres :

[
\lambda_X=\sqrt{1-\gamma},
]

[
\lambda_Y=\sqrt{1-\gamma},
]

[
\lambda_Z=1-\gamma.
]

Le Pauli twirling associé donne :

[
p_X=p_Y=\frac{\gamma}{4},
]

et :

[
p_Z

\frac{
2-\gamma-2\sqrt{1-\gamma}
}{4}.
]

Enfin :

[
p_I

1-p_X-p_Y-p_Z.
]

Pour \gamma\ll1 :

[
p_Z
\approx
\frac{\gamma^2}{16}.
]

Ainsi, l’approximation :

[
p_Z=\frac{\gamma}{2}
]

ne doit pas être utilisée comme résultat du Pauli twirling exact de l’amplitude damping.

Le canal d’amplitude damping initial n’est pas lui-même un canal de Pauli. Le twirling constitue une réduction de sa représentation.

## 17.1 Test numérique

Le module doit vérifier les valeurs fermées pour plusieurs régimes, par exemple :

[
\gamma\in
{10^{-4},10^{-2},0.3,0.9}.
]

Pour chaque valeur :

[
p_I+p_X+p_Y+p_Z=1
]

doit être vérifié dans la tolérance numérique choisie.

Le test doit également comparer les probabilités calculées aux expressions analytiques.

---

# 18. Déphasage pur

Pour le canal de déphasage pur :

[
p_Z

\frac{
1-e^{-\tau/T_\phi}
}{2},
]

et :

[
p_I=1-p_Z.
]

Cette expression dépend de la convention de canal utilisée.

Lorsque la relaxation d’amplitude et le déphasage pur sont modélisés comme mécanismes markoviens indépendants :

[
\frac{1}{T_2}

\frac{1}{2T_1}
+
\frac{1}{T_\phi}.
]

Il faut distinguer ce modèle exponentiel d’un bruit de flux quasi statique ou coloré.

---

# 19. Composition des canaux en PTM

Pour deux canaux :

[
\mathcal E_1
\quad\text{et}\quad
\mathcal E_2,
]

la composition :

[
\mathcal E_2\circ\mathcal E_1
]

est représentée par :

[
R_{\mathcal E_2\circ\mathcal E_1}

R_{\mathcal E_2}R_{\mathcal E_1}.
]

Lorsque les PTM sont diagonales dans la base de Pauli, la composition se réduit au produit composante par composante.

Le résultat peut ensuite être reconverti en probabilités :

[
{p_I,p_X,p_Y,p_Z}.
]

Cette opération est algébriquement exacte pour les canaux représentés.

Elle ne signifie pas que la réduction de Pauli conserve toute l’information du canal quantique initial.

---

# 20. Bruit quasi statique et bruit coloré

Un canal de Pauli indépendant par cycle ne reproduit pas intégralement :

- les corrélations temporelles entre cycles ;
- le bruit quasi statique ;
- les spectres de type 1/f ;
- les effets non markoviens ;
- les formes non exponentielles de cohérence ;
- les contributions de la courbure fréquentielle ;
- les corrélations entre erreurs d’amplitude et erreurs de phase.

Une fluctuation quasi statique du flux ne doit donc pas être identifiée automatiquement à un taux de déphasage markovien constant.

Une réduction en canal effectif doit préciser :

- la durée du cycle ;
- la distribution du bruit ;
- le spectre considéré ;
- les hypothèses temporelles ;
- la méthode de moyenne ou de twirling ;
- la quantité observée après réduction.

---

# 21. Qubit bosonique Kerr-Cat

## 21.1 Modèle

Le module "kerr_cat_simulator.py" étudie une dynamique bosonique de type Kerr-Cat.

Le modèle peut comprendre :

- un Hamiltonien de Kerr ;
- un pompage à deux photons ;
- des dissipateurs de Lindblad ;
- une intégration numérique ;
- une estimation de fidélité logique ;
- une estimation du leakage.

La dynamique est décrite par :

[
\frac{d\rho}{dt}

-\frac{i}{\hbar}[H,\rho]
+
\sum_j
\mathcal D"L_j" (\rho),
]

avec :

[
\mathcal D"L" (\rho)

L\rho L^\dagger

\frac12
\left{
L^\dagger L,\rho
\right}.
]

La signification physique des opérateurs L_j, leurs unités et leurs taux doivent être définis par les paramètres du modèle.

## 21.2 Indépendance du Kerr-Cat

Le simulateur Kerr-Cat constitue actuellement une branche indépendante.

Il ne doit pas être décrit comme une simulation couplée Fluxonium–Kerr-Cat.

Les résultats de fidélité et de leakage dépendent notamment :

- du Hamiltonien ;
- des dissipateurs ;
- des taux de perte ;
- de la troncature de Fock ;
- du pas d’intégration ;
- des conditions initiales ;
- de la définition de la fidélité.

Toute fonction non implémentée ou levant "NotImplementedError" doit être considérée comme non opérationnelle.

---

# 22. Correction d’erreurs quantique

## 22.1 Probabilités locales

Le modèle QEC utilise des probabilités d’erreur locales associées à une durée de cycle :

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

Cette conversion dépend du canal choisi.

Elle ne constitue pas une description complète :

- des erreurs cohérentes ;
- des erreurs corrélées ;
- des erreurs temporellement persistantes ;
- des erreurs de mesure ;
- des corrélations entre qubits.

---

# 23. Code de répétition de phase

Pour une distance impaire d, le code considère :

[
|0_L\rangle

|+\rangle^{\otimes d},
]

[
|1_L\rangle

|-\rangle^{\otimes d}.
]

Pour des erreurs de phase indépendantes de probabilité p_Z, la probabilité d’échec par majorité est :

[
P_L^{(Z)}

\sum_{k=(d+1)/2}^{d}
\binom{d}{k}
p_Z^k
(1-p_Z)^{d-k}.
]

Cette expression suppose :

- erreurs indépendantes ;
- probabilité identique sur les qubits ;
- absence d’erreur de mesure ;
- règle de décodage par majorité.

Elle ne constitue pas un modèle général du code de surface.

---

# 24. Budget d’erreur logique

Le budget logique agrège différentes contributions d’erreur selon les hypothèses du modèle.

Lorsqu’une combinaison utilise une hypothèse d’indépendance, celle-ci doit être explicitement documentée.

Des probabilités marginales :

[
p_A,\quad p_B
]

ne déterminent pas une distribution conjointe exacte sans information supplémentaire sur :

[
P(A\cap B).
]

Le budget logique doit donc distinguer :

- probabilités marginales ;
- hypothèses d’indépendance ;
- corrélations éventuelles ;
- contributions de lecture ;
- contributions de portes ;
- contributions de mémoire.

---

# 25. Chirurgie de réseau

Le projet contient un prototype de chirurgie de réseau associé à une configuration annoncée de :

[
d=3
]

et :

[
21\ \text{qubits}.
]

Cette valeur de 21 qubits doit être interprétée selon la composition exacte de l’architecture implémentée.

Le dépôt doit préciser explicitement :

- quels qubits sont des qubits de données ;
- quels qubits sont des ancillas ;
- quels ancillas servent aux stabilisateurs ;
- quels qubits sont utilisés comme interface ;
- quelles opérations sont réellement simulées.

Le nombre total de qubits ne doit donc pas être assimilé automatiquement au seul nombre de qubits de données d’un patch de surface code.

## 25.1 Statut du prototype

Le protocole est considéré comme un prototype numérique, notamment pour explorer une séquence du type :

INIT
  │
  ▼
MERGE
  │
  ▼
d ROUNDS
  │
  ▼
SPLIT
  │
  ▼
MEASURE Z_L1

Le prototype ne doit pas être présenté comme :

- une simulation complète des stabilisateurs ;
- une simulation complète des circuits physiques ;
- une modélisation exhaustive des erreurs de lecture ;
- une démonstration de seuil QEC ;
- une validation matérielle ;
- une preuve de correction universelle.

---

# 26. Décodage géométrique

Le module "geometric_mwpm_decoder.py" explore une pondération du décodage par des informations locales telles que :

- le biais du bruit ;
- les paramètres locaux ;
- des informations géométriques.

Cette approche constitue un prototype de décodage pondéré.

Pour établir un décodeur MWPM complet de code de surface, il faudrait notamment définir et valider :

- le graphe des événements de syndrome ;
- les erreurs de données ;
- les erreurs de mesure ;
- les frontières logiques ;
- les stabilisateurs X ;
- les stabilisateurs Z ;
- l’historique espace-temps ;
- les règles de merge ;
- les règles de split ;
- les chemins logiques ;
- les poids des arêtes ;
- la règle de décision finale.

Une pondération géométrique ne suffit pas, à elle seule, à établir un MWPM espace-temps complet.

---

# 27. Structure du dépôt

L’organisation documentée du projet est :

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

Cette arborescence doit rester synchronisée avec les fichiers réellement suivis dans Git.

Les chemins "test/" et "tests/" ne doivent pas être utilisés indistinctement dans la documentation.

---

# 28. Rôle des modules

Module| Responsabilité
"fluxonium_core.py"| Structures et constantes liées au modèle Fluxonium
"fluxonium_hamiltonian.py"| Construction et diagonalisation de l’Hamiltonien
"fluxonium_model.py"| Observables, éléments de matrice et paramètres de cohérence
"fubini_study_manifold.py"| QGT et métrique de Fubini–Study
"fluxon_engine.py"| Modulation paramétrique et bilan effectif de quasiparticules
"noise_channel.py"| Canaux de bruit, PTM et Pauli twirling
"kerr_cat_simulator.py"| Dynamique bosonique Kerr-Cat
"pulse_calibration.py"| Outils de calibration d’impulsions
"error_correction.py"| Codes de répétition et budget d’erreur
"geometric_mwpm_decoder.py"| Prototype de décodage pondéré
"pont_lourd_unifie.py"| Orchestration du pont et prototype de chirurgie
"edifice.py"| Architecture globale et intégration des couches
"run_pipeline.py"| Point d’entrée du pipeline standard

---

# 29. Installation

Créer un environnement virtuel :

python -m venv .venv

Activer l’environnement.

Linux / macOS

source .venv/bin/activate

Windows PowerShell

.\.venv\Scripts\Activate.ps1

Installer les dépendances :

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

Les dépendances doivent rester cohérentes entre :

- "requirements.txt" ;
- "pyproject.toml" ;
- les imports effectivement utilisés.

---

# 30. Exécution

Les points d’entrée documentés sont :

Pipeline standard

python run_pipeline.py

Orchestrateur du pont

python pont_lourd_unifie.py

La présence d’un point d’entrée ne garantit pas son bon fonctionnement.

Avant de le qualifier d’opérationnel, il faut vérifier :

- les imports ;
- les paramètres requis ;
- les chemins ;
- les API ;
- les dépendances ;
- l’exécution dans un environnement propre.

---

# 31. Tests

La suite peut être lancée avec :

python -m pytest test/ -v

ou :

python -m pytest -q

Les domaines de test documentés sont :

Fichier| Domaine
"test_fluxonium.py"| Spectre, états propres et propriétés du Fluxonium
"test_fubini_study.py"| QGT et métrique de Fubini–Study
"test_kerr_cat.py"| Dynamique Lindblad, fidélité et leakage
"test_error_correction.py"| Codes de répétition et budget d’erreur
"test_chirurgie_21q_pont.py"| Prototype de chirurgie 21 qubits

Les tests peuvent vérifier :

- invariants algébriques ;
- cohérence numérique ;
- convergence ;
- stabilité d’une simulation ;
- comportement d’une abstraction logicielle.

Ils ne remplacent pas :

- une validation analytique indépendante ;
- une comparaison à une méthode de référence ;
- une validation expérimentale.

Aucun résultat de test ne doit être présenté comme réussi sans exécution effective sur le commit concerné.

---

# 32. Tests minimaux à renforcer

## 32.1 Amplitude damping

Tester :

[
\gamma
\in
{10^{-4},10^{-2},0.3,0.9}.
]

Vérifier :

[
p_X=p_Y=\frac{\gamma}{4},
]

[
p_Z

\frac{
2-\gamma-2\sqrt{1-\gamma}
}{4},
]

[
p_I=1-p_X-p_Y-p_Z,
]

et :

[
\sum_Pp_P=1.
]

## 32.2 Bilan quasiparticulaire

Le test doit distinguer :

- fermeture arithmétique du bilan ;
- exactitude d’une solution dynamique indépendante.

Une identité obtenue directement à partir de l’équation de mise à jour ne constitue pas une validation indépendante.

## 32.3 Convergence Fluxonium

Les tests doivent examiner la stabilité de :

[
f_{01},
\quad
\langle0|\hat n|1\rangle,
\quad
\langle0|\hat\phi|1\rangle,
\quad
\frac{\partial f_{01}}{\partial\Phi},
\quad
g_{FS}
]

lorsque :

- la grille est raffinée ;
- le domaine est élargi ;
- le nombre d’états est augmenté.

---

# 33. Limites scientifiques

Domaine| Limite actuelle
Fluxonium| Dépendance aux paramètres, à la discrétisation et aux hypothèses de bruit
Opérateurs numériques| Approximation sur grille finie et conditions aux limites
Géométrie quantique| g_{FS} ne détermine pas directement une probabilité d’erreur
Quasiparticules| Modèle limité aux hypothèses de quasiparticules froides et à la petite jonction
Substrat| Bilan effectif, sans dynamique microscopique complète
Bruit| Canal de Pauli réduit
Bruit quasi statique| Corrélations temporelles non représentées intégralement par un canal markovien indépendant
Kerr-Cat| Branche indépendante du Fluxonium
Calibration| Contrôle DRAG non établi comme fonctionnalité opérationnelle
QEC| Modèles effectifs et prototypes
Décodage| Pas de démonstration d’un MWPM espace-temps complet
Chirurgie| Prototype numérique, pas une simulation stabilisateur-level complète
Expérience| Aucune validation expérimentale intégrée

---

# 34. Feuille de route de validation

P0 — Cohérence logicielle

- vérifier les imports ;
- aligner "src/__init__.py" sur les symboles réellement définis ;
- vérifier la cohérence entre API et tests ;
- rechercher les commandes shell accidentellement présentes dans les fichiers Python ;
- exécuter les tests dans un environnement propre ;
- documenter les résultats ;
- ajouter une CI GitHub Actions exécutant la suite de tests.

Exemple de commande CI :

python -m pytest -q

---

P1 — Solveur Fluxonium

- vérifier l’hermiticité de l’Hamiltonien ;
- vérifier les opérateurs discrets ;
- étudier la convergence en taille de grille ;
- étudier la convergence en étendue de domaine ;
- comparer les spectres à une méthode indépendante ;
- vérifier les éléments de matrice ;
- contrôler les effets de troncature.

---

P2 — Géométrie quantique

- vérifier la normalisation des états ;
- contrôler l’invariance de jauge ;
- étudier la convergence en pas de différence finie ;
- comparer plusieurs méthodes de calcul du QGT ;
- identifier explicitement l’état utilisé ;
- distinguer les métriques géométriques des grandeurs de bruit.

---

P3 — Bruit et canaux

- documenter les conventions spectrales ;
- distinguer bruit blanc, bruit coloré et bruit quasi statique ;
- comparer les canaux à une évolution de référence ;
- quantifier l’information perdue par le twirling ;
- vérifier la conversion des taux en probabilités par cycle ;
- vérifier les conventions dimensionnelles des PSD ;
- tester séparément T_1, T_\phi, bruit de flux et composition PTM.

---

P4 — QEC et décodage

- vérifier les probabilités logiques par énumération exhaustive pour les petits codes ;
- séparer erreurs de données et erreurs de mesure ;
- définir le graphe des événements de syndrome ;
- construire un graphe espace-temps ;
- comparer le décodeur à une implémentation de référence ;
- vérifier les frontières logiques ;
- documenter les poids utilisés par le décodage.

---

P5 — Chirurgie de réseau

- définir les qubits de données ;
- définir les ancillas ;
- définir les stabilisateurs initiaux ;
- définir les stabilisateurs finaux ;
- définir les circuits de préparation ;
- définir les opérations de merge ;
- définir les opérations de split ;
- modéliser les erreurs de portes ;
- modéliser les erreurs de lecture ;
- vérifier le suivi du cadre de Pauli ;
- vérifier les observables logiques ;
- étudier plusieurs distances avant toute affirmation concernant un seuil.

---

P6 — Intégration physique

- définir les hypothèses de couplage entre qubits ;
- relier les paramètres matériels aux canaux locaux ;
- établir une interface physique testable entre géométrie et bruit, si cette hypothèse est conservée ;
- comparer les prédictions à des données expérimentales ;
- publier les incertitudes ;
- publier les domaines de validité ;
- distinguer systématiquement modèle, simulation, calibration et mesure.

---

# 35. Hygiène du dépôt

Les noms de fichiers doivent rester compatibles avec les conventions Python et Git.

Éviter notamment :

NOYAU QUANTIQUE V52.py

Préférer un nom de module stable, par exemple :

quantum_core.py

Les numéros de version doivent être portés par Git, les tags ou les releases plutôt que par des noms de fichiers successifs.

L’arborescence documentée doit également correspondre à l’arborescence réelle :

- "src/" ;
- "test/" ou "tests/" ;
- "notebooks/" si présent ;
- "results/" si présent ;
- "run_pipeline.py" s’il est effectivement présent.

La documentation ne doit pas annoncer un fichier absent du commit considéré.

---

# 36. Reproductibilité

Chaque résultat numérique important devrait être accompagné autant que possible de :

1. paramètres matériels ;
2. unités ;
3. hypothèses physiques ;
4. conditions aux limites ;
5. taille de grille ;
6. nombre d’états retenus ;
7. tolérances numériques ;
8. durée d’évolution ;
9. durée de cycle ;
10. modèle de bruit ;
11. convention de twirling ;
12. graine aléatoire ;
13. nombre de tirs Monte-Carlo ;
14. incertitudes numériques ;
15. incertitudes statistiques ;
16. commit Git utilisé.

La reproductibilité doit permettre de distinguer une propriété du modèle d’un résultat dépendant d’un réglage numérique particulier.

---

# 37. Bonnes pratiques d’interprétation

Les résultats doivent être présentés avec leur niveau de preuve.

Quantum-Lab ne doit pas être présenté comme :

- une dérivation d’un seuil QEC ;
- une validation d’une architecture matérielle ;
- une prédiction directe d’une fidélité expérimentale ;
- une implémentation complète de correction d’erreurs ;
- un modèle exhaustif des quasiparticules ;
- une démonstration d’un couplage géométrique physique ;
- un remplacement d’une calibration expérimentale ;
- une reproduction intégrale d’un bruit non markovien par un canal de Pauli indépendant.

Un résultat numérique doit rester associé au modèle qui l’a produit.

---

# 38. Références et vérification documentaire

Les références bibliographiques utilisées pour justifier les modèles doivent être vérifiées avant publication.

Les domaines à documenter comprennent notamment :

- Pauli twirling de l’amplitude damping ;
- modèles de quasiparticules dans les circuits supraconducteurs ;
- Fluxonium ;
- code de surface tourné ;
- chirurgie de réseau ;
- Kerr-Cat ;
- géométrie quantique des états quantiques.

Une référence non vérifiée ne doit pas être présentée comme une source bibliographique établie.

Lorsque l’implémentation utilise une convention différente de celle d’un article de référence, la différence doit être explicitement documentée.

---

# 39. Positionnement scientifique

Quantum-Lab étudie une question centrale :

«Comment relier de manière traçable les propriétés microscopiques d’un circuit supraconducteur, la géométrie de ses états quantiques, ses canaux de bruit effectifs et les performances d’une architecture de correction d’erreurs ?»

L’intérêt du projet réside dans la construction d’interfaces explicites entre ces niveaux de description.

Ces niveaux ne sont pas identiques :

[
\text{Hamiltonien}
\neq
\text{géométrie quantique}
\neq
\text{canal de bruit}
\neq
\text{modèle QEC}.
]

Le passage de l’un à l’autre doit être défini par une transformation mathématique ou une hypothèse physique explicite.

La géométrie quantique peut caractériser la réponse locale des états.

Le canal de bruit décrit une réduction de la dynamique dissipative.

Le modèle QEC transforme ensuite ce canal en objets utilisables pour l’étude de la correction d’erreurs.

Aucune de ces étapes ne doit être confondue avec une validation expérimentale.

---

# 40. Principe méthodologique

Le pipeline suit une règle générale :

[
\boxed{
\text{modèle}
\rightarrow
\text{réduction}
\rightarrow
\text{simulation}
\rightarrow
\text{validation}
\rightarrow
\text{interprétation}
}
]

Chaque flèche doit être identifiable.

Une réduction non documentée ne doit pas être présentée comme une loi physique.

Une simulation stable ne constitue pas nécessairement une simulation physiquement correcte.

Une correspondance numérique ne constitue pas nécessairement une causalité physique.

Une hypothèse de couplage doit rester une hypothèse jusqu’à ce qu’elle produise une prédiction indépendante, quantitative et falsifiable.

---

# 41. Édifice quantique

Le module "edifice.py" peut être considéré comme la couche d’architecture globale du projet.

Son rôle est de rendre visibles les trajectoires entre niveaux :

PARAMÈTRES DU SUBSTRAT
          │
          ▼
     FLUXONIUM
          │
          ├──────────────► GÉOMÉTRIE
          │                   │
          │                   ▼
          │              gFS / QGT
          │
          ▼
        BRUIT
          │
          ├── T1
          ├── Tφ
          ├── bruit de flux
          └── quasiparticules
          │
          ▼
      CANAL EFFECTIF
          │
          ▼
       PAULI PTM
          │
          ▼
          QEC
          │
          ├── biais
          ├── budget
          ├── décodage
          └── chirurgie

La trajectoire observée dans les simulations doit être interprétée comme une trajectoire dans l’espace des modèles et des observables, et non comme la trajectoire physique complète d’un dispositif expérimental.

L’édifice constitue ainsi une représentation structurée des interfaces du projet.

---

# 42. Conclusion

Quantum-Lab propose un environnement numérique pour étudier les interfaces entre :

[
\text{Fluxonium}
\rightarrow
\text{géométrie quantique}
\rightarrow
\text{bruit}
\rightarrow
\text{réduction de Pauli}
\rightarrow
\text{QEC}.
]

Une branche complémentaire explore :

[
\text{Kerr-Cat}
\rightarrow
\text{dynamique bosonique}
\rightarrow
\text{fidélité / leakage}.
]

Une autre couche explore :

[
\text{inhomogénéité du substrat}
\rightarrow
\text{paramètres locaux}
\rightarrow
\text{bruit local}
\rightarrow
\text{décodage}.
]

Le point central du projet n’est pas de confondre ces descriptions, mais de construire des interfaces explicites entre elles.

La question ouverte principale reste l’interface :

[
\boxed{
g_{FS}
;; ? ;;
\text{canal de bruit}
}
]

Tant qu’aucune relation physique indépendante ne la justifie, cette interface demeure un objet de recherche.

Quantum-Lab — Modéliser les interfaces, contrôler les réductions, éprouver les correspondances.
