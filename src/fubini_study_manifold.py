
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module : fubini_study_manifold.py

FLUXONIUM QUANTUM GEOMETRY
==========================

Calcul numérique de la géométrie quantique du Fluxonium :

    1. Construction du Hamiltonien Fluxonium.
    2. Résolution de l'état fondamental.
    3. Calcul de la métrique de Fubini-Study.
    4. Calcul de la distance géodésique entre états fondamentaux.
    5. Projection géométrique sur un réseau de 21 nœuds.
    6. Estimation d'un indicateur de sensibilité géométrique
       au bruit de flux.
    7. Vérifications numériques et diagnostics de convergence.

HAMILTONIEN
-----------

Convention utilisée :

    H = 4 E_C n²
        + (E_L / 2) (phi - phi_ext)²
        - E_J cos(phi)

avec :

    [phi, n] = i
    n = -i d/dphi

Les paramètres énergétiques sont exprimés en GHz,
c'est-à-dire en unités E/h.

Le Hamiltonien numérique représente donc H/h en GHz.

MÉTRIQUE DE FUBINI-STUDY
------------------------

Pour un état fondamental normalisé |psi(phi_ext)> :

    g_phi_phi =
        <dpsi|dpsi> - |<psi|dpsi>|²

La dérivée est estimée par différence centrale,
après alignement de phase des états.

La métrique est la partie réelle du tenseur
géométrique quantique.

DISTANCE DE FUBINI-STUDY
-------------------------

Pour deux états purs :

    D_FS(a,b) = arccos(|<psi(a)|psi(b)>|)

Cette distance est une distance géodésique sur
l'espace projectif des états purs.

Elle n'est pas la distance de Bures au sens strict.

PROJECTION SUR 21 NŒUDS
-----------------------

Le module produit :

    - phi_nodes
    - g_PhiPhi_nodes
    - D_FS(pi, phi_i)
    - geometric_phase_variance_proxy

Le dernier terme est un indicateur sans dimension :

    V_geom(i) = g_PhiPhi(i) * sigma_phi²

Il ne constitue PAS un taux de déphasage en Hz.

LIMITES
-------

- Discrétisation spatiale par différences finies d'ordre 2.
- Conditions aux limites de Dirichlet.
- Bruit de flux considéré uniquement comme une amplitude
  quadratique locale dans l'indicateur géométrique.
- Pas de calcul de spectre de bruit S_phi(omega).
- Pas de calcul de T1 ou T_phi.
- Pas de dérivation d'un canal quantique.
- Pas de validation expérimentale.

Le couplage entre géométrie quantique et bruit doit être
établi dans un module physique séparé.

Auteur : Xioranne Lessard
Licence : MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import eigh


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FubiniStudyConfig:
    """
    Paramètres du modèle Fluxonium et de la discrétisation.

    Les énergies sont exprimées en GHz (E/h).

    n_grid :
        Nombre total de points de la grille, bornes incluses.

    phi_bound :
        Domaine de phase [-phi_bound, +phi_bound].

    dphi_ext :
        Pas utilisé pour la dérivée numérique par rapport
        au flux externe.

    boundary :
        Conditions aux limites de Dirichlet.
    """

    E_J_GHz: float = 4.0
    E_C_GHz: float = 1.0
    E_L_GHz: float = 0.5

    n_grid: int = 400
    phi_bound: float = 5.0 * np.pi

    dphi_ext: float = 1e-4

    boundary: str = "dirichlet"

    # Contrôle numérique
    overlap_tolerance: float = 1e-12
    degeneracy_tolerance_GHz: float = 1e-10

    # Cache activé pour les états déjà calculés
    use_cache: bool = True


# ============================================================
# GÉOMÉTRIE QUANTIQUE DU FLUXONIUM
# ============================================================

class FluxoniumQuantumGeometry:
    """
    Calculateur autonome de géométrie quantique Fluxonium.

    Le module construit son propre Hamiltonien discret.

    Convention :
        H/h en GHz.

    La coordonnée phi est sans dimension et exprimée en radians.
    """

    def __init__(
        self,
        cfg: Optional[FubiniStudyConfig] = None,
    ) -> None:

        self.cfg = (
            cfg
            if cfg is not None
            else FubiniStudyConfig()
        )

        self._validate_config()

        self.phi = np.linspace(
            -self.cfg.phi_bound,
            self.cfg.phi_bound,
            self.cfg.n_grid,
            dtype=np.float64,
        )

        self.h = float(
            self.phi[1] - self.phi[0]
        )

        # Points intérieurs : les deux extrémités sont
        # fixées à zéro par les conditions de Dirichlet.
        self.phi_interior = self.phi[1:-1]

        self.n_interior = len(self.phi_interior)

        self.T_kin = self._build_kinetic()

        self._state_cache: Dict[
            float,
            Tuple[float, np.ndarray],
        ] = {}

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_config(self) -> None:

        cfg = self.cfg

        energies = (
            cfg.E_J_GHz,
            cfg.E_C_GHz,
            cfg.E_L_GHz,
        )

        if not all(np.isfinite(x) for x in energies):
            raise ValueError(
                "Les paramètres énergétiques doivent être finis."
            )

        if cfg.E_J_GHz <= 0.0:
            raise ValueError("E_J_GHz doit être positif.")

        if cfg.E_C_GHz <= 0.0:
            raise ValueError("E_C_GHz doit être positif.")

        if cfg.E_L_GHz <= 0.0:
            raise ValueError("E_L_GHz doit être positif.")

        if cfg.n_grid < 20:
            raise ValueError(
                "n_grid doit être supérieur ou égal à 20."
            )

        if not np.isfinite(cfg.phi_bound):
            raise ValueError("phi_bound doit être fini.")

        if cfg.phi_bound <= 0.0:
            raise ValueError("phi_bound doit être positif.")

        if not np.isfinite(cfg.dphi_ext):
            raise ValueError("dphi_ext doit être fini.")

        if cfg.dphi_ext <= 0.0:
            raise ValueError("dphi_ext doit être positif.")

        if cfg.boundary != "dirichlet":
            raise ValueError(
                "Seules les conditions de Dirichlet sont implémentées."
            )

        if cfg.overlap_tolerance <= 0.0:
            raise ValueError(
                "overlap_tolerance doit être positif."
            )

        if cfg.degeneracy_tolerance_GHz <= 0.0:
            raise ValueError(
                "degeneracy_tolerance_GHz doit être positif."
            )

    # ========================================================
    # HAMILTONIEN CINÉTIQUE
    # ========================================================

    def _build_kinetic(self) -> np.ndarray:
        """
        Construit l'opérateur cinétique :

            T = -4 E_C d²/dphi²

        Discrétisation centrée d'ordre 2 :

            d²psi/dphi² =
                (psi[i+1] - 2 psi[i] + psi[i-1]) / h²

        Les deux extrémités du domaine sont imposées à zéro.

        La matrice intérieure est symétrique tridiagonale.
        """

        n = self.n_interior

        diagonal = np.full(
            n,
            2.0,
            dtype=np.float64,
        )

        off_diagonal = np.full(
            n - 1,
            -1.0,
            dtype=np.float64,
        )

        d2 = (
            np.diag(diagonal)
            + np.diag(off_diagonal, k=1)
            + np.diag(off_diagonal, k=-1)
        ) / (self.h**2)

        kinetic = (
            4.0
            * self.cfg.E_C_GHz
            * d2
        )

        return kinetic.astype(np.float64)

    # ========================================================
    # POTENTIEL FLUXONIUM
    # ========================================================

    def potential(
        self,
        phi_ext: float,
    ) -> np.ndarray:
        """
        Potentiel du Fluxonium :

            V(phi) =
                0.5 * E_L * (phi - phi_ext)**2
                - E_J * cos(phi)

        Résultat en GHz.
        """

        if not np.isfinite(phi_ext):
            raise ValueError("phi_ext doit être fini.")

        phi = self.phi_interior

        return (
            0.5
            * self.cfg.E_L_GHz
            * (phi - phi_ext) ** 2

            - self.cfg.E_J_GHz
            * np.cos(phi)
        )

    # ========================================================
    # HAMILTONIEN TOTAL
    # ========================================================

    def hamiltonian(
        self,
        phi_ext: float,
    ) -> np.ndarray:
        """
        Construit :

            H/h = T + V(phi_ext)

        La matrice est réelle et symétrique.
        """

        V = self.potential(phi_ext)

        H = (
            self.T_kin
            + np.diag(V)
        )

        return H

    # ========================================================
    # CONTRÔLE D'HERMITICITÉ
    # ========================================================

    def hermiticity_error(
        self,
        phi_ext: float,
    ) -> float:
        """
        Mesure :

            epsilon_H = ||H - H†||_F

        Pour cette construction réelle symétrique,
        cette quantité doit être numériquement nulle.
        """

        H = self.hamiltonian(phi_ext)

        return float(
            np.linalg.norm(
                H - H.conj().T,
                ord="fro",
            )
        )

    # ========================================================
    # NORMALISATION
    # ========================================================

    def _normalize_state(
        self,
        psi: np.ndarray,
    ) -> np.ndarray:
        """
        Normalisation continue discrétisée :

            h * sum_i |psi_i|² = 1

        Les deux composantes aux frontières sont nulles.
        """

        psi = np.asarray(
            psi,
            dtype=np.complex128,
        )

        norm_squared = (
            np.sum(np.abs(psi) ** 2)
            * self.h
        )

        if not np.isfinite(norm_squared):
            raise RuntimeError(
                "Norme non finie de l'état."
            )

        if norm_squared <= 0.0:
            raise RuntimeError(
                "Norme nulle de l'état."
            )

        return psi / np.sqrt(norm_squared)

    def inner_product(
        self,
        psi_a: np.ndarray,
        psi_b: np.ndarray,
    ) -> complex:
        """
        Produit scalaire discret :

            <a|b> = h * sum_i conj(a_i) * b_i
        """

        if psi_a.shape != psi_b.shape:
            raise ValueError(
                "Dimensions incompatibles pour le produit scalaire."
            )

        return complex(
            self.h
            * np.vdot(psi_a, psi_b)
        )

    # ========================================================
    # ÉTAT FONDAMENTAL
    # ========================================================

    def ground_state(
        self,
        phi_ext: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Calcule le premier état propre du Hamiltonien.

        Retour :
            E0_GHz
            psi0 normalisé sur la grille complète.

        Les frontières sont nulles.

        L'énergie est exprimée en GHz, soit E0/h.
        """

        if not np.isfinite(phi_ext):
            raise ValueError("phi_ext doit être fini.")

        cache_key = float(phi_ext)

        if (
            self.cfg.use_cache
            and cache_key in self._state_cache
        ):
            energy, state = self._state_cache[cache_key]

            return energy, state.copy()

        H = self.hamiltonian(phi_ext)

        values, vectors = eigh(
            H,
            subset_by_index=[0, 1],
            check_finite=True,
            driver="evr",
        )

        energy_0 = float(values[0])
        energy_1 = float(values[1])

        gap = energy_1 - energy_0

        if gap <= self.cfg.degeneracy_tolerance_GHz:
            raise RuntimeError(
                "Écart énergétique trop faible pour identifier "
                f"un état fondamental non dégénéré : {gap:.6e} GHz."
            )

        psi_interior = vectors[:, 0]

        # Reconstruction de la grille complète.
        psi_full = np.zeros(
            self.cfg.n_grid,
            dtype=np.complex128,
        )

        psi_full[1:-1] = psi_interior

        psi_full = self._normalize_state(
            psi_full
        )

        if self.cfg.use_cache:
            self._state_cache[cache_key] = (
                energy_0,
                psi_full.copy(),
            )

        return energy_0, psi_full

    # ========================================================
    # ALIGNEMENT DE PHASE
    # ========================================================

    def _align_global_phase(
        self,
        reference: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Aligne la phase globale de state sur reference.

        Si :

            z = <reference|state>

        on applique :

            state_aligned = state * exp(-i arg(z))

        Le chevauchement devient réel positif.

        Cette opération élimine la phase globale arbitraire
        introduite par le solveur numérique.
        """

        overlap = self.inner_product(
            reference,
            state,
        )

        if abs(overlap) <= self.cfg.overlap_tolerance:
            raise RuntimeError(
                "Chevauchement trop faible pour aligner la phase."
            )

        phase = np.angle(overlap)

        return (
            state
            * np.exp(-1j * phase)
        )

    # ========================================================
    # MÉTRIQUE DE FUBINI-STUDY
    # ========================================================

    def metric_component(
        self,
        phi_ext: float,
        eps: Optional[float] = None,
    ) -> float:
        """
        Calcule la métrique de Fubini-Study :

            g_phi_phi =
                <dpsi|dpsi> - |<psi|dpsi>|²

        La dérivée est calculée par différence centrale :

            dpsi/dphi_ext =
                [psi(phi+eps)-psi(phi-eps)]/(2 eps)

        Les deux états sont alignés en phase sur l'état central.

        La projection :

            |dpsi_perp> =
                |dpsi> - |psi><psi|dpsi>

        élimine la composante correspondant à la phase globale.

        Résultat :
            g_phi_phi, en radian^-2.
        """

        if eps is None:
            eps = self.cfg.dphi_ext

        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError(
                "eps doit être fini et strictement positif."
            )

        _, psi_0 = self.ground_state(phi_ext)

        _, psi_minus = self.ground_state(
            phi_ext - eps
        )

        _, psi_plus = self.ground_state(
            phi_ext + eps
        )

        psi_minus = self._align_global_phase(
            psi_0,
            psi_minus,
        )

        psi_plus = self._align_global_phase(
            psi_0,
            psi_plus,
        )

        derivative = (
            psi_plus - psi_minus
        ) / (2.0 * eps)

        projection = self.inner_product(
            psi_0,
            derivative,
        )

        derivative_perp = (
            derivative
            - psi_0 * projection
        )

        metric = float(
            np.real(
                self.inner_product(
                    derivative_perp,
                    derivative_perp,
                )
            )
        )

        if not np.isfinite(metric):
            raise RuntimeError(
                "La métrique calculée n'est pas finie."
            )

        # Une très petite valeur négative peut apparaître
        # par arrondi numérique.
        if metric < -1e-10:
            raise RuntimeError(
                f"Métrique négative anormale : {metric:.6e}."
            )

        return max(0.0, metric)

    # ========================================================
    # DISTANCE DE FUBINI-STUDY
    # ========================================================

    def fubini_study_distance(
        self,
        phi_a: float,
        phi_b: float,
    ) -> float:
        """
        Distance géodésique de Fubini-Study :

            D_FS(a,b) = arccos(|<psi_a|psi_b>|)

        Les états sont normalisés avec le produit scalaire
        discret intégral.

        Résultat en radians.
        """

        if not np.isfinite(phi_a) or not np.isfinite(phi_b):
            raise ValueError(
                "Les paramètres de flux doivent être finis."
            )

        if phi_a == phi_b:
            return 0.0

        _, psi_a = self.ground_state(phi_a)
        _, psi_b = self.ground_state(phi_b)

        overlap = abs(
            self.inner_product(
                psi_a,
                psi_b,
            )
        )

        overlap = float(
            np.clip(overlap, 0.0, 1.0)
        )

        return float(
            np.arccos(overlap)
        )

    # ========================================================
    # DISTANCE DE BURES STRICTE POUR ÉTATS PURS
    # ========================================================

    def bures_distance_pure(
        self,
        phi_a: float,
        phi_b: float,
    ) -> float:
        """
        Distance de Bures stricte pour deux états purs :

            D_B = sqrt(2 * (1 - |<psi_a|psi_b>|))

        Cette quantité est distincte de la distance géodésique
        de Fubini-Study.

        Elle est fournie séparément pour éviter toute confusion.
        """

        _, psi_a = self.ground_state(phi_a)
        _, psi_b = self.ground_state(phi_b)

        overlap = abs(
            self.inner_product(
                psi_a,
                psi_b,
            )
        )

        overlap = float(
            np.clip(overlap, 0.0, 1.0)
        )

        return float(
            np.sqrt(
                max(0.0, 2.0 * (1.0 - overlap))
            )
        )

    # ========================================================
    # COMPATIBILITÉ API
    # ========================================================

    def bures_geodesic_distance(
        self,
        phi_a: float,
        phi_b: float,
    ) -> float:
        """
        Alias rétrocompatible.

        Malgré son ancien nom, cette méthode renvoie
        la distance géodésique de Fubini-Study.

        Utiliser de préférence :
            fubini_study_distance()
        """

        return self.fubini_study_distance(
            phi_a,
            phi_b,
        )

    # ========================================================
    # PROJECTION SUR 21 NŒUDS
    # ========================================================

    def project_onto_21q_lattice(
        self,
        gradient_flux: float = 0.04,
        sigma_flux_uPhi0: float = 1.5,
        center_flux: float = np.pi,
        envelope_width: float = 0.6,
        n_qubits: int = 21,
        metric_eps: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Projette les grandeurs géométriques sur un réseau de nœuds.

        Profil de flux :

            x_i = linspace(-1,1,n)

            delta_phi_i =
                gradient_flux
                * exp(-0.5*(x_i/envelope_width)**2)
                * sin(pi*x_i)

            phi_i = center_flux + delta_phi_i

        Calcule :

            g_i = g_phi_phi(phi_i)

            D_i = D_FS(center_flux, phi_i)

            V_geom_i = g_i * sigma_phi²

        Conversion du bruit de flux :

            sigma_phi = 2*pi*sigma_flux/Phi_0

        sigma_flux_uPhi0 est exprimé en micro-flux quantums.

        IMPORTANT :
        V_geom est un indicateur quadratique sans dimension.
        Ce n'est pas un taux de déphasage.
        """

        if n_qubits < 2:
            raise ValueError(
                "n_qubits doit être supérieur ou égal à 2."
            )

        values = (
            gradient_flux,
            sigma_flux_uPhi0,
            center_flux,
            envelope_width,
        )

        if not all(np.isfinite(v) for v in values):
            raise ValueError(
                "Les paramètres de projection doivent être finis."
            )

        if gradient_flux < 0.0:
            raise ValueError(
                "gradient_flux doit être positif ou nul."
            )

        if sigma_flux_uPhi0 < 0.0:
            raise ValueError(
                "sigma_flux_uPhi0 doit être positif ou nul."
            )

        if envelope_width <= 0.0:
            raise ValueError(
                "envelope_width doit être strictement positif."
            )

        x_coords = np.linspace(
            -1.0,
            1.0,
            n_qubits,
        )

        delta_phi = (
            gradient_flux
            * np.exp(
                -0.5
                * (x_coords / envelope_width) ** 2
            )
            * np.sin(np.pi * x_coords)
        )

        phi_nodes = (
            center_flux + delta_phi
        )

        g_nodes = np.empty(
            n_qubits,
            dtype=np.float64,
        )

        d_fs_nodes = np.empty(
            n_qubits,
            dtype=np.float64,
        )

        d_bures_nodes = np.empty(
            n_qubits,
            dtype=np.float64,
        )

        for i, phi_node in enumerate(phi_nodes):

            g_nodes[i] = self.metric_component(
                phi_node,
                eps=metric_eps,
            )

            d_fs_nodes[i] = self.fubini_study_distance(
                center_flux,
                phi_node,
            )

            d_bures_nodes[i] = self.bures_distance_pure(
                center_flux,
                phi_node,
            )

        # Conversion micro-Phi_0 -> Phi/Phi_0.
        sigma_flux_normalized = (
            sigma_flux_uPhi0 * 1e-6
        )

        # Conversion flux normalisé -> phase en radians.
        sigma_phi_rad = (
            2.0 * np.pi
            * sigma_flux_normalized
        )

        # Indicateur quadratique de sensibilité géométrique.
        geometric_phase_variance_proxy = (
            g_nodes * sigma_phi_rad**2
        )

        return {
            "x_coords": x_coords,

            "phi_nodes": phi_nodes,
            "delta_phi_nodes": delta_phi,

            "g_PhiPhi_nodes": g_nodes,

            "fubini_study_distance_from_center": d_fs_nodes,

            "bures_distance_pure_from_center": d_bures_nodes,

            "sigma_phi_rad": np.full(
                n_qubits,
                sigma_phi_rad,
                dtype=np.float64,
            ),

            "geometric_phase_variance_proxy":
                geometric_phase_variance_proxy,

        }

    # ========================================================
    # CONVERGENCE DE LA MÉTRIQUE
    # ========================================================

    def metric_step_convergence(
        self,
        phi_ext: float,
        eps_values: Tuple[float, ...] = (
            1e-2,
            5e-3,
            1e-3,
            5e-4,
            1e-4,
        ),
    ) -> Dict[str, np.ndarray]:
        """
        Évalue la sensibilité du calcul de la métrique au pas eps.

        Une convergence acceptable doit montrer une zone
        relativement stable avant que les erreurs d'arrondi
        ou de résolution numérique ne dominent.
        """

        if len(eps_values) < 2:
            raise ValueError(
                "Au moins deux pas sont nécessaires."
            )

        eps_array = np.asarray(
            eps_values,
            dtype=np.float64,
        )

        if np.any(~np.isfinite(eps_array)):
            raise ValueError(
                "Les pas doivent être finis."
            )

        if np.any(eps_array <= 0.0):
            raise ValueError(
                "Les pas doivent être strictement positifs."
            )

        metrics = np.array(
            [
                self.metric_component(
                    phi_ext,
                    eps=float(eps),
                )
                for eps in eps_array
            ],
            dtype=np.float64,
        )

        relative_changes = np.full(
            len(metrics),
            np.nan,
            dtype=np.float64,
        )

        for i in range(1, len(metrics)):

            denominator = max(
                abs(metrics[i - 1]),
                1e-15,
            )

            relative_changes[i] = (
                abs(metrics[i] - metrics[i - 1])
                / denominator
            )

        return {
            "eps_values": eps_array,
            "metric_values": metrics,
            "relative_changes": relative_changes,
        }

    # ========================================================
    # CONVERGENCE DE LA GRILLE
    # ========================================================

    def grid_convergence(
        self,
        phi_ext: float,
        grid_sizes: Tuple[int, ...] = (
            200,
            300,
            400,
            600,
        ),
    ) -> Dict[str, np.ndarray]:
        """
        Compare l'énergie fondamentale et la métrique
        pour plusieurs résolutions de grille.

        Chaque résolution utilise les mêmes paramètres
        physiques et la même extension phi_bound.

        La comparaison reste dépendante des conditions
        aux limites et du domaine retenu.
        """

        if len(grid_sizes) < 2:
            raise ValueError(
                "Au moins deux tailles de grille sont nécessaires."
            )

        energies = []
        metrics = []
        spacings = []

        for n in grid_sizes:

            cfg_local = FubiniStudyConfig(
                E_J_GHz=self.cfg.E_J_GHz,
                E_C_GHz=self.cfg.E_C_GHz,
                E_L_GHz=self.cfg.E_L_GHz,

                n_grid=int(n),
                phi_bound=self.cfg.phi_bound,

                dphi_ext=self.cfg.dphi_ext,

                boundary=self.cfg.boundary,

                overlap_tolerance=self.cfg.overlap_tolerance,
                degeneracy_tolerance_GHz=(
                    self.cfg.degeneracy_tolerance_GHz
                ),

                use_cache=False,
            )

            model = FluxoniumQuantumGeometry(
                cfg_local
            )

            energy, _ = model.ground_state(
                phi_ext
            )

            metric = model.metric_component(
                phi_ext
            )

            energies.append(energy)
            metrics.append(metric)
            spacings.append(model.h)

        return {
            "grid_sizes": np.asarray(
                grid_sizes,
                dtype=np.int64,
            ),

            "grid_spacing": np.asarray(
                spacings,
                dtype=np.float64,
            ),

            "ground_energy_GHz": np.asarray(
                energies,
                dtype=np.float64,
            ),

            "metric_values": np.asarray(
                metrics,
                dtype=np.float64,
            ),
        }

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def diagnostics(
        self,
        phi_ext: float = np.pi,
    ) -> Dict[str, float]:
        """
        Produit des diagnostics du Hamiltonien et de l'état.

        Aucun minimum de métrique au sweet spot n'est imposé.

        Cette propriété doit être étudiée numériquement
        et ne constitue pas une identité universelle.
        """

        energy, psi = self.ground_state(
            phi_ext
        )

        H = self.hamiltonian(
            phi_ext
        )

        hermiticity = self.hermiticity_error(
            phi_ext
        )

        norm = float(
            self.h
            * np.sum(np.abs(psi) ** 2)
        )

        values = eigh(
            H,
            subset_by_index=[0, 1],
            check_finite=True,
            driver="evr",
        )[0]

        gap = float(
            values[1] - values[0]
        )

        metric = self.metric_component(
            phi_ext
        )

        return {
            "ground_energy_GHz": float(energy),

            "first_excited_energy_GHz": float(values[1]),

            "energy_gap_GHz": gap,

            "state_norm": norm,

            "hermiticity_error": hermiticity,

            "metric_phi_phi": metric,

            "grid_spacing": float(self.h),

            "n_grid": int(self.cfg.n_grid),
        }


# ============================================================
# VÉRIFICATIONS NUMÉRIQUES
# ============================================================

def verify_hamiltonian() -> Dict[str, float]:
    """
    Vérifie :
        - hermiticité ;
        - normalisation ;
        - positivité de la métrique ;
        - écart énergétique positif.
    """

    geom = FluxoniumQuantumGeometry(
        FubiniStudyConfig()
    )

    result = geom.diagnostics(
        phi_ext=np.pi
    )

    assert result["hermiticity_error"] < 1e-10

    assert abs(
        result["state_norm"] - 1.0
    ) < 1e-10

    assert result["metric_phi_phi"] >= 0.0

    assert result["energy_gap_GHz"] > 0.0

    return result


def verify_distance_properties() -> Dict[str, float]:
    """
    Vérifie les propriétés élémentaires des distances :

        D(a,a) = 0
        D(a,b) >= 0
        D(a,b) = D(b,a)
    """

    geom = FluxoniumQuantumGeometry(
        FubiniStudyConfig()
    )

    a = np.pi
    b = 0.95 * np.pi

    d_aa = geom.fubini_study_distance(a, a)

    d_ab = geom.fubini_study_distance(a, b)

    d_ba = geom.fubini_study_distance(b, a)

    assert abs(d_aa) < 1e-12
    assert d_ab >= 0.0
    assert abs(d_ab - d_ba) < 1e-10

    return {
        "distance_same_state": d_aa,
        "distance_a_b": d_ab,
        "distance_b_a": d_ba,
    }


def verify_sweet_spot_observation() -> Dict[str, float]:
    """
    Compare la métrique au point phi_ext = pi et hors du point.

    Cette fonction est un diagnostic, PAS une assertion physique.

    On ne suppose pas que la métrique doit nécessairement
    être minimale au sweet spot.
    """

    geom = FluxoniumQuantumGeometry(
        FubiniStudyConfig()
    )

    phi_sweet = np.pi
    phi_off = 0.95 * np.pi

    g_sweet = geom.metric_component(
        phi_sweet
    )

    g_off = geom.metric_component(
        phi_off
    )

    return {
        "phi_sweet": float(phi_sweet),
        "phi_off": float(phi_off),

        "g_sweet": float(g_sweet),
        "g_off": float(g_off),

        "sweet_spot_metric_is_lower": bool(
            g_sweet < g_off
        ),
    }


# ============================================================
# RAPPORT
# ============================================================

def print_projection_report(
    projection: Dict[str, np.ndarray],
) -> None:
    """
    Affiche un résumé des 21 nœuds géométriques.
    """

    g = projection["g_PhiPhi_nodes"]

    d_fs = projection[
        "fubini_study_distance_from_center"
    ]

    variance = projection[
        "geometric_phase_variance_proxy"
    ]

    print()
    print("=" * 70)
    print(" FLUXONIUM — FUBINI-STUDY GEOMETRIC PROJECTION")
    print("=" * 70)

    print(
        f" Nombre de nœuds              : {len(g)}"
    )

    print(
        f" g_PhiPhi minimale            : {np.min(g):.8e}"
    )

    print(
        f" g_PhiPhi maximale            : {np.max(g):.8e}"
    )

    print(
        f" Distance FS minimale         : {np.min(d_fs):.8e}"
    )

    print(
        f" Distance FS maximale         : {np.max(d_fs):.8e}"
    )

    print(
        f" Indicateur géométrique min.  : {np.min(variance):.8e}"
    )

    print(
        f" Indicateur géométrique max.  : {np.max(variance):.8e}"
    )

    print("-" * 70)

    print(
        " Interprétation : susceptibilité géométrique quadratique."
    )

    print(
        " Ce résultat n'est pas un taux de déphasage en Hz."
    )

    print(
        " La relation avec T_phi nécessite un modèle de bruit."
    )

    print("=" * 70)
    print()


# ============================================================
# EXÉCUTION AUTONOME
# ============================================================

def main() -> None:

    cfg = FubiniStudyConfig()

    geom = FluxoniumQuantumGeometry(
        cfg
    )

    print()
    print("=" * 70)
    print(" FLUXONIUM QUANTUM GEOMETRY")
    print("=" * 70)

    print(
        f"E_J/h = {cfg.E_J_GHz:.6f} GHz"
    )

    print(
        f"E_C/h = {cfg.E_C_GHz:.6f} GHz"
    )

    print(
        f"E_L/h = {cfg.E_L_GHz:.6f} GHz"
    )

    print(
        f"Grille = {cfg.n_grid} points"
    )

    print(
        f"Domaine = [-{cfg.phi_bound:.6f}, "
        f"+{cfg.phi_bound:.6f}]"
    )

    # --------------------------------------------------------
    # 1. Diagnostics fondamentaux
    # --------------------------------------------------------

    print()
    print("[1] DIAGNOSTICS DU HAMILTONIEN")

    diagnostics = geom.diagnostics(
        np.pi
    )

    for key, value in diagnostics.items():
        print(f"{key:32s}: {value}")

    # --------------------------------------------------------
    # 2. Distances géométriques
    # --------------------------------------------------------

    print()
    print("[2] DISTANCES ENTRE ÉTATS")

    phi_a = np.pi
    phi_b = 0.95 * np.pi

    d_fs = geom.fubini_study_distance(
        phi_a,
        phi_b,
    )

    d_bures = geom.bures_distance_pure(
        phi_a,
        phi_b,
    )

    print(
        f"Distance Fubini-Study : {d_fs:.10e} rad"
    )

    print(
        f"Distance Bures stricte : {d_bures:.10e}"
    )

    # --------------------------------------------------------
    # 3. Projection sur 21 nœuds
    # --------------------------------------------------------

    print()
    print("[3] PROJECTION SUR LE RÉSEAU")

    projection = geom.project_onto_21q_lattice(
        gradient_flux=0.04,
        sigma_flux_uPhi0=1.5,
        center_flux=np.pi,
        envelope_width=0.6,
        n_qubits=21,
    )

    print_projection_report(
        projection
    )

    # --------------------------------------------------------
    # 4. Convergence du pas différentiel
    # --------------------------------------------------------

    print()
    print("[4] CONVERGENCE DU PAS DIFFÉRENTIEL")

    convergence = geom.metric_step_convergence(
        phi_ext=np.pi
    )

    for eps, metric in zip(
        convergence["eps_values"],
        convergence["metric_values"],
    ):

        print(
            f"eps={eps:.2e}  "
            f"g={metric:.10e}"
        )

    # --------------------------------------------------------
    # 5. Convergence de la grille
    # --------------------------------------------------------

    print()
    print("[5] CONVERGENCE DE LA GRILLE")

    grid_result = geom.grid_convergence(
        phi_ext=np.pi,
        grid_sizes=(200, 300, 400, 600),
    )

    for n, spacing, energy, metric in zip(
        grid_result["grid_sizes"],
        grid_result["grid_spacing"],
        grid_result["ground_energy_GHz"],
        grid_result["metric_values"],
    ):

        print(
            f"N={n:4d}  "
            f"h={spacing:.6e}  "
            f"E0={energy:.10f} GHz  "
            f"g={metric:.10e}"
        )

    # --------------------------------------------------------
    # 6. Vérifications
    # --------------------------------------------------------

    print()
    print("[6] VÉRIFICATIONS NUMÉRIQUES")

    verify_hamiltonian()

    verify_distance_properties()

    sweet_spot = verify_sweet_spot_observation()

    print(
        "Hermiticité, normalisation et distances : OK"
    )

    print(
        "Métrique au sweet spot : "
        f"{sweet_spot['g_sweet']:.10e}"
    )

    print(
        "Métrique hors sweet spot : "
        f"{sweet_spot['g_off']:.10e}"
    )

    print(
        "Minimum observé au sweet spot : "
        f"{sweet_spot['sweet_spot_metric_is_lower']}"
    )

    print()
    print("=" * 70)
    print(" FIN DU CALCUL")
    print("=" * 70)


if __name__ == "__main__":
    main()
