
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module : geometric_mwpm_decoder.py

GEOMETRIC LATTICE SURGERY — REDUCED DECODER
===========================================

Benchmark d'un décodage de répétition de phase à 5 qubits de données,
pondéré par une métrique géométrique de Fubini–Study.

Périmètre
---------
- 21 valeurs géométriques en entrée ;
- extraction d'une couture de 5 qubits de données ;
- modèle indépendant d'erreurs Z ;
- syndrome obtenu par les stabilisateurs X_i X_{i+1} ;
- décodage MAP exact par énumération des classes compatibles ;
- comparaison avec un décodeur uniforme ;
- évaluation Monte-Carlo sur des échantillons communs ;
- intervalles d'incertitude binomiaux de Wilson.

IMPORTANT
---------
Ce module n'implémente PAS un MWPM espace-temps de code de surface
tourné d=3.

Il implémente un décodage MAP exact pour un code de répétition de
phase à cinq qubits de données.

La pondération géométrique :

    p_i = p_base_Z + gamma_FS * D_FS(i)**2

est phénoménologique.

Elle ne constitue pas une dérivation physique de la probabilité
d'erreur à partir de la métrique de Fubini–Study.

La validité du modèle dépend notamment :
- de la définition des distances fournies ;
- de leur normalisation ;
- de la calibration de gamma_FS ;
- de l'hypothèse d'indépendance des erreurs ;
- de l'absence d'erreurs de mesure dans ce benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GeometricSurgeryConfig:
    """
    Configuration du benchmark réduit.

    n_qubits :
        Nombre de valeurs géométriques disponibles.

    n_data :
        Nombre de qubits de données de la chaîne décodée.

    seam_indices :
        Indices des qubits extraits du réseau de 21 qubits.

    p_base_Z :
        Probabilité de base d'une erreur Z.

    fs_coupling_gamma :
        Coefficient phénoménologique de couplage géométrique.

    n_shots :
        Nombre de réalisations Monte-Carlo.

    seed :
        Graine aléatoire.
    """

    n_qubits: int = 21
    n_data: int = 5

    seam_indices: Tuple[int, ...] = (7, 8, 9, 10, 11)

    p_base_Z: float = 0.0025
    fs_coupling_gamma: float = 0.35

    n_shots: int = 25_000
    seed: int = 42

    probability_floor: float = 1e-12
    probability_ceiling: float = 0.499999


# ============================================================
# DÉCODEUR
# ============================================================

class GeometricLatticeSurgeryDecoder:
    """
    Décodeur MAP exact pour une chaîne de répétition de phase.

    Convention
    ----------
    Les erreurs physiques considérées sont des erreurs Z.

    Les stabilisateurs mesurés sont :

        S_i = X_i X_{i+1}

    Pour un vecteur d'erreurs :

        e = (e_0, ..., e_{n-1})

    le syndrome est :

        s_i = e_i XOR e_{i+1}

    Deux configurations d'erreurs produisent le même syndrome
    si elles diffèrent par le vecteur logique :

        (1, 1, ..., 1)

    Le décodeur compare les probabilités des deux classes
    compatibles avec le syndrome.

    Ce calcul est exact pour le modèle probabiliste indépendant
    spécifié. Il ne représente pas un décodeur de surface code.
    """

    def __init__(
        self,
        cfg: GeometricSurgeryConfig,
        fs_distances: np.ndarray,
    ) -> None:

        self.cfg = cfg

        self._validate_config()

        self.fs_distances = np.asarray(
            fs_distances,
            dtype=np.float64,
        )

        self._validate_distances()

        self.p_Z_nodes = self._compute_geometric_probabilities()

        self.seam_indices = np.asarray(
            self.cfg.seam_indices,
            dtype=np.int64,
        )

        self.p_seam = self.p_Z_nodes[self.seam_indices]

        self.uniform_probability = float(np.mean(self.p_seam))

        self._syndrome_lookup = self._build_syndrome_lookup()

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_config(self) -> None:

        cfg = self.cfg

        if cfg.n_qubits <= 0:
            raise ValueError("n_qubits doit être strictement positif.")

        if cfg.n_data < 3 or cfg.n_data % 2 == 0:
            raise ValueError(
                "n_data doit être impair et supérieur ou égal à 3."
            )

        if cfg.n_shots <= 0:
            raise ValueError("n_shots doit être strictement positif.")

        if not np.isfinite(cfg.p_base_Z):
            raise ValueError("p_base_Z doit être fini.")

        if not 0.0 < cfg.p_base_Z < 0.5:
            raise ValueError("p_base_Z doit appartenir à ]0, 0.5[.")

        if not np.isfinite(cfg.fs_coupling_gamma):
            raise ValueError("fs_coupling_gamma doit être fini.")

        if cfg.fs_coupling_gamma < 0.0:
            raise ValueError(
                "fs_coupling_gamma doit être positif ou nul."
            )

        if not (
            0.0
            < cfg.probability_floor
            < cfg.probability_ceiling
            < 0.5
        ):
            raise ValueError("Bornes de probabilité invalides.")

        if len(cfg.seam_indices) != cfg.n_data:
            raise ValueError(
                "Le nombre d'indices de couture doit être égal à n_data."
            )

        if len(set(cfg.seam_indices)) != len(cfg.seam_indices):
            raise ValueError("Les indices de couture doivent être uniques.")

        for index in cfg.seam_indices:
            if not 0 <= index < cfg.n_qubits:
                raise ValueError(
                    f"Indice de couture invalide : {index}."
                )

    def _validate_distances(self) -> None:

        if self.fs_distances.ndim != 1:
            raise ValueError(
                "fs_distances doit être un tableau unidimensionnel."
            )

        if len(self.fs_distances) != self.cfg.n_qubits:
            raise ValueError(
                f"Attendu {self.cfg.n_qubits} distances, "
                f"reçu {len(self.fs_distances)}."
            )

        if not np.all(np.isfinite(self.fs_distances)):
            raise ValueError(
                "Les distances contiennent des valeurs non finies."
            )

        if np.any(self.fs_distances < 0.0):
            raise ValueError(
                "Une distance géométrique ne peut pas être négative."
            )

    # ========================================================
    # MODÈLE GÉOMÉTRIQUE
    # ========================================================

    def _compute_geometric_probabilities(self) -> np.ndarray:
        """
        Convertit les distances FS en probabilités locales.

        Relation phénoménologique :

            p_Z(i) = p_base_Z + gamma_FS * D_FS(i)^2

        Les probabilités sont bornées dans ]0, 0.5[.

        Cette transformation n'est pas une conséquence
        mathématique de la métrique de Fubini–Study.
        """

        p_geom = (
            self.cfg.p_base_Z
            + self.cfg.fs_coupling_gamma
            * self.fs_distances**2
        )

        p_geom = np.clip(
            p_geom,
            self.cfg.probability_floor,
            self.cfg.probability_ceiling,
        )

        return p_geom.astype(np.float64)

    # ========================================================
    # SYNDROME
    # ========================================================

    @staticmethod
    def compute_syndrome(errors: np.ndarray) -> np.ndarray:
        """
        Calcule le syndrome de répétition de phase.

        Convention :

            s_i = e_i XOR e_{i+1}

        Pour n qubits de données, le syndrome contient n-1 bits.
        """

        errors = np.asarray(errors, dtype=np.int8)

        if errors.ndim != 1:
            raise ValueError("errors doit être un vecteur.")

        if len(errors) < 2:
            raise ValueError("Au moins deux qubits sont nécessaires.")

        if np.any((errors != 0) & (errors != 1)):
            raise ValueError("Les erreurs doivent être binaires.")

        return np.bitwise_xor(
            errors[:-1],
            errors[1:],
        ).astype(np.int8)

    # ========================================================
    # ÉNUMÉRATION DES CLASSES
    # ========================================================

    def _build_syndrome_lookup(
        self,
    ) -> Dict[Tuple[int, ...], List[np.ndarray]]:
        """
        Construit les classes de configurations compatibles
        avec chaque syndrome.

        Pour un code de répétition de phase de longueur n :

            2**n configurations
            2**(n-1) syndromes
            2 configurations par syndrome

        Les deux configurations diffèrent par une erreur logique
        de poids n.
        """

        n = self.cfg.n_data

        lookup: Dict[
            Tuple[int, ...],
            List[np.ndarray],
        ] = {}

        for bits in product((0, 1), repeat=n):

            error = np.asarray(bits, dtype=np.int8)

            syndrome = tuple(
                self.compute_syndrome(error).tolist()
            )

            lookup.setdefault(syndrome, []).append(error)

        expected_syndromes = 2 ** (n - 1)

        if len(lookup) != expected_syndromes:
            raise RuntimeError(
                "Nombre inattendu de syndromes dans l'énumération."
            )

        for syndrome, candidates in lookup.items():

            if len(candidates) != 2:
                raise RuntimeError(
                    f"Classe de syndrome incorrecte : {syndrome}."
                )

        return lookup

    # ========================================================
    # LOG-VRAISEMBLANCE
    # ========================================================

    @staticmethod
    def _log_likelihood(
        error: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        """
        Log-vraisemblance d'une configuration indépendante.

        P(e) = produit_i p_i**e_i * (1-p_i)**(1-e_i)

        Le calcul logarithmique évite les sous-flux numériques.
        """

        error = np.asarray(error, dtype=np.int8)

        p = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if error.shape != p.shape:
            raise ValueError(
                "Dimensions incompatibles entre erreur et probabilités."
            )

        return float(
            np.sum(
                error * np.log(p)
                + (1 - error) * np.log1p(-p)
            )
        )

    # ========================================================
    # DÉCODAGE MAP
    # ========================================================

    def decode_map(
        self,
        syndrome: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Décode un syndrome par maximum a posteriori.

        Le décodeur sélectionne la configuration la plus probable
        parmi les deux configurations compatibles avec le syndrome.

        En cas d'égalité numérique, la configuration de poids
        de Hamming minimal est sélectionnée.
        """

        syndrome = np.asarray(
            syndrome,
            dtype=np.int8,
        )

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if syndrome.shape != (self.cfg.n_data - 1,):
            raise ValueError("Dimension incorrecte du syndrome.")

        if probabilities.shape != (self.cfg.n_data,):
            raise ValueError(
                "Dimension incorrecte des probabilités."
            )

        if np.any((syndrome != 0) & (syndrome != 1)):
            raise ValueError("Le syndrome doit être binaire.")

        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or np.any(probabilities >= 0.5)
        ):
            raise ValueError(
                "Les probabilités doivent être finies et dans ]0, 0.5[."
            )

        key = tuple(syndrome.tolist())

        candidates = self._syndrome_lookup.get(key)

        if candidates is None:
            raise ValueError("Syndrome non reconnu.")

        scores = [
            self._log_likelihood(candidate, probabilities)
            for candidate in candidates
        ]

        tolerance = 1e-12

        if abs(scores[0] - scores[1]) <= tolerance:

            selected = min(
                candidates,
                key=lambda candidate: int(np.sum(candidate)),
            )

        else:

            selected = candidates[int(scores[1] > scores[0])]

        return selected.copy()

    # ========================================================
    # DÉCODAGE UNIFORME
    # ========================================================

    def decode_uniform(
        self,
        syndrome: np.ndarray,
    ) -> np.ndarray:
        """
        Décode avec une probabilité uniforme.

        La probabilité uniforme est la moyenne des probabilités
        géométriques de la couture.

        Cela permet de comparer les deux décodeurs à taux moyen
        d'erreur identique.
        """

        probabilities = np.full(
            self.cfg.n_data,
            self.uniform_probability,
            dtype=np.float64,
        )

        return self.decode_map(
            syndrome,
            probabilities,
        )

    # ========================================================
    # DÉCODAGE GÉOMÉTRIQUE
    # ========================================================

    def decode_geometric(
        self,
        syndrome: np.ndarray,
    ) -> np.ndarray:
        """
        Décode avec les probabilités locales géométriques.
        """

        return self.decode_map(
            syndrome,
            self.p_seam,
        )

    # ========================================================
    # GÉNÉRATION MONTE-CARLO
    # ========================================================

    def _generate_errors(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Génère des erreurs Z indépendantes selon les probabilités
        locales de la couture.
        """

        return (
            rng.random(
                (self.cfg.n_shots, self.cfg.n_data)
            )
            < self.p_seam[None, :]
        ).astype(np.int8)

    # ========================================================
    # ÉVALUATION LOGIQUE
    # ========================================================

    @staticmethod
    def _logical_failure(
        actual_error: np.ndarray,
        correction: np.ndarray,
    ) -> bool:
        """
        Vérifie si l'erreur résiduelle appartient à la classe
        logique non triviale.

        Pour un syndrome correctement reproduit, le résidu doit
        être soit :

            (0, 0, ..., 0)

        soit :

            (1, 1, ..., 1)

        Le second cas correspond à une erreur logique Z_L.

        Le critère est donc :

            parité(erreur XOR correction) == 1

        pour une longueur impaire.
        """

        residual = np.bitwise_xor(
            actual_error,
            correction,
        )

        if len(residual) % 2 == 0:
            raise ValueError(
                "La longueur doit être impaire pour cette convention."
            )

        if np.all(residual == 0):
            return False

        if np.all(residual == 1):
            return True

        raise RuntimeError(
            "La correction ne reproduit pas le syndrome observé."
        )

    # ========================================================
    # INTERVALLE DE CONFIANCE
    # ========================================================

    @staticmethod
    def _wilson_interval(
        failures: int,
        shots: int,
        z: float = 1.959963984540054,
    ) -> Tuple[float, float]:
        """
        Intervalle de confiance binomial de Wilson à 95 %.
        """

        if shots <= 0:
            raise ValueError("shots doit être positif.")

        p = failures / shots

        denominator = 1.0 + z**2 / shots

        center = (
            p + z**2 / (2.0 * shots)
        ) / denominator

        radius = (
            z
            * np.sqrt(
                p * (1.0 - p) / shots
                + z**2 / (4.0 * shots**2)
            )
            / denominator
        )

        return (
            max(0.0, center - radius),
            min(1.0, center + radius),
        )

    # ========================================================
    # BENCHMARK
    # ========================================================

    def benchmark_vs_uniform(self) -> Dict[str, float]:
        """
        Compare les performances du décodeur uniforme et du
        décodeur géométrique sur les mêmes réalisations physiques.

        Les erreurs sont générées avec les probabilités locales
        p_seam.

        Le décodeur uniforme connaît uniquement la probabilité
        moyenne de la couture.

        Le décodeur géométrique utilise les probabilités locales.

        Les deux décodeurs sont donc confrontés au même bruit.
        """

        rng = np.random.default_rng(
            self.cfg.seed
        )

        errors = self._generate_errors(rng)

        failures_uniform = 0
        failures_geometric = 0

        for error in errors:

            syndrome = self.compute_syndrome(error)

            correction_uniform = self.decode_uniform(
                syndrome
            )

            correction_geometric = self.decode_geometric(
                syndrome
            )

            if self._logical_failure(
                error,
                correction_uniform,
            ):
                failures_uniform += 1

            if self._logical_failure(
                error,
                correction_geometric,
            ):
                failures_geometric += 1

        n = self.cfg.n_shots

        p_uniform = failures_uniform / n
        p_geometric = failures_geometric / n

        ci_uniform = self._wilson_interval(
            failures_uniform,
            n,
        )

        ci_geometric = self._wilson_interval(
            failures_geometric,
            n,
        )

        if p_geometric > 0.0:
            gain = p_uniform / p_geometric
        elif p_uniform > 0.0:
            gain = float("inf")
        else:
            gain = float("nan")

        return {
            "p_L_uniform": float(p_uniform),
            "p_L_geometric": float(p_geometric),

            "ci95_uniform_low": float(ci_uniform[0]),
            "ci95_uniform_high": float(ci_uniform[1]),

            "ci95_geometric_low": float(ci_geometric[0]),
            "ci95_geometric_high": float(ci_geometric[1]),

            "geometric_gain_factor": float(gain),

            "mean_node_error_Z": float(np.mean(self.p_seam)),
            "max_node_error_Z": float(np.max(self.p_seam)),
            "min_node_error_Z": float(np.min(self.p_seam)),

            "n_shots": int(n),
            "n_failures_uniform": int(failures_uniform),
            "n_failures_geometric": int(failures_geometric),

            "uniform_probability": float(
                self.uniform_probability
            ),

            "std_node_error_Z": float(
                np.std(self.p_seam)
            ),
        }

    # ========================================================
    # RAPPORT
    # ========================================================

    def print_report(
        self,
        results: Dict[str, float],
    ) -> None:
        """
        Affiche les résultats avec leur périmètre méthodologique.
        """

        print()
        print("=" * 66)
        print(" GEOMETRIC LATTICE SURGERY — REDUCED MAP BENCHMARK")
        print("=" * 66)

        print(
            f" Réseau d'entrée             : "
            f"{self.cfg.n_qubits} valeurs géométriques"
        )

        print(
            f" Chaîne décodée              : "
            f"{self.cfg.n_data} qubits de données"
        )

        print(
            f" Indices de couture          : "
            f"{self.cfg.seam_indices}"
        )

        print(
            f" Nombre de tirs              : "
            f"{results['n_shots']}"
        )

        print(
            f" Graine                       : "
            f"{self.cfg.seed}"
        )

        print("-" * 66)

        print(
            f" Erreur Z moyenne             : "
            f"{results['mean_node_error_Z'] * 100:.6f} %"
        )

        print(
            f" Erreur Z minimale            : "
            f"{results['min_node_error_Z'] * 100:.6f} %"
        )

        print(
            f" Erreur Z maximale            : "
            f"{results['max_node_error_Z'] * 100:.6f} %"
        )

        print(
            f" Écart-type spatial           : "
            f"{results['std_node_error_Z'] * 100:.6f} %"
        )

        print("-" * 66)

        print(
            f" Échec logique uniforme       : "
            f"{results['p_L_uniform'] * 100:.6f} %"
        )

        print(
            f" IC95 uniforme                : "
            f"[{results['ci95_uniform_low'] * 100:.6f} %, "
            f"{results['ci95_uniform_high'] * 100:.6f} %]"
        )

        print(
            f" Échecs uniformes             : "
            f"{results['n_failures_uniform']}"
        )

        print("-" * 66)

        print(
            f" Échec logique géométrique    : "
            f"{results['p_L_geometric'] * 100:.6f} %"
        )

        print(
            f" IC95 géométrique             : "
            f"[{results['ci95_geometric_low'] * 100:.6f} %, "
            f"{results['ci95_geometric_high'] * 100:.6f} %]"
        )

        print(
            f" Échecs géométriques          : "
            f"{results['n_failures_geometric']}"
        )

        print("-" * 66)

        print(
            f" Gain géométrique             : "
            f"{results['geometric_gain_factor']:.6g}"
        )

        print("=" * 66)

        print(" PÉRIMÈTRE SCIENTIFIQUE")
        print("-" * 66)

        print(
            " Décodage : MAP exact pour une répétition de phase."
        )

        print(
            " Bruit : erreurs Z indépendantes, sans erreurs de mesure."
        )

        print(
            " Géométrie : couplage phénoménologique D_FS²."
        )

        print(
            " Comparaison : mêmes échantillons physiques."
        )

        print(
            " Non implémenté : MWPM espace-temps de surface code."
        )

        print(
            " Non démontré : chirurgie complète à 21 qubits."
        )

        print("=" * 66)
        print()


# ============================================================
# EXÉCUTION AUTONOME
# ============================================================

def main() -> None:
    """
    Démonstration reproductible.

    Le profil de distance est synthétique.
    Il ne constitue pas une sortie mesurée du solveur Fluxonium.
    """

    cfg = GeometricSurgeryConfig()

    x = np.linspace(
        -1.0,
        1.0,
        cfg.n_qubits,
    )

    fs_distances_demo = (
        0.14 * np.abs(np.sin(np.pi * x))
    )

    # Perturbation géométrique localisée.
    fs_distances_demo[11] = 0.24

    decoder = GeometricLatticeSurgeryDecoder(
        cfg=cfg,
        fs_distances=fs_distances_demo,
    )

    results = decoder.benchmark_vs_uniform()

    decoder.print_report(results)


if __name__ == "__main__":
    main()
