"""
kerr_cat_simulator.py

Simulateur modulaire de qubits bosoniques de chat Kerr.

Objectif
--------
Fournir une base logicielle reproductible pour étudier :
- la stabilisation d'états chats par drive à deux photons ;
- l'effet des pertes de photons et du bruit de déphasage ;
- la fidélité logique et le taux d'erreur ;
- la robustesse aux variations de paramètres ;
- la préparation de données pour la correction d'erreurs.

Modèle physique
---------------
Oscillateur Kerr unique en base de Fock tronquée, avec :
- hamiltonien Kerr : H_kerr = -K a†²a²
- drive à deux photons : H_2ph = eps2 (a†² + a²)
- drive à un photon : H_1ph = eps1 (a† + a)
- dissipation : perte de photons (kappa) et déphasage (gamma_phi)

L'équation maîtresse de Lindblad est intégrée par RK4.

Auteur : [Votre nom]
Licence : MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

LOGGER = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Utilitaires numériques
# ----------------------------------------------------------------------

def ensure_dense(matrix: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    """Convertit une matrice creuse en matrice dense si nécessaire."""
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def is_hermitian(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Vérifie l'hermiticité d'une matrice dense."""
    return np.allclose(matrix, matrix.conj().T, atol=tol)


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalise un vecteur d'état quantique."""
    norm = np.linalg.norm(state)
    if norm < 1e-14:
        raise ValueError("Impossible de normaliser un état de norme nulle.")
    return state / norm


def safe_divide(numerator: float, denominator: float) -> float:
    """Division sûre pour éviter les divisions par zéro."""
    if abs(denominator) < 1e-14:
        return 0.0
    return numerator / denominator


def clip_probability(value: float, tol: float = 1e-12) -> float:
    """Contraint une probabilité dans [0, 1] avec tolérance numérique."""
    return float(np.clip(value, 0.0, 1.0))


def trace_density_matrix(rho: sp.csr_matrix) -> float:
    """Calcule la trace d'une matrice densité (partie réelle)."""
    return float(np.real(np.trace(ensure_dense(rho))))


# ----------------------------------------------------------------------
# Opérateurs quantiques
# ----------------------------------------------------------------------

def creation_operator(dim: int) -> sp.csr_matrix:
    """Construit l'opérateur de création a† en base de Fock tronquée."""
    if dim < 2:
        raise ValueError("dim doit être au moins 2.")
    data = np.sqrt(np.arange(1, dim, dtype=float))
    return sp.diags(data, offsets=1, format="csr")


def annihilation_operator(dim: int) -> sp.csr_matrix:
    """Construit l'opérateur d'annihilation a en base de Fock tronquée."""
    return creation_operator(dim).conj().T.tocsr()


def number_operator(dim: int) -> sp.csr_matrix:
    """Construit l'opérateur de nombre n = a†a."""
    return sp.diags(np.arange(dim, dtype=float), format="csr")


def parity_operator(dim: int) -> sp.csr_matrix:
    """Construit l'opérateur de parité P = exp(i pi n)."""
    diag = np.array([1.0 if n % 2 == 0 else -1.0 for n in range(dim)])
    return sp.diags(diag, format="csr")


def identity_operator(dim: int) -> sp.csr_matrix:
    """Construit l'identité en base de Fock tronquée."""
    return sp.identity(dim, format="csr")


def commutator(A: sp.spmatrix, B: sp.spmatrix) -> sp.csr_matrix:
    """Calcule [A, B] = AB - BA."""
    return (A @ B - B @ A).tocsr()


def anticommutator(A: sp.spmatrix, B: sp.spmatrix) -> sp.csr_matrix:
    """Calcule {A, B} = AB + BA."""
    return (A @ B + B @ A).tocsr()


def dissipator(
    rho: sp.csr_matrix,
    L: sp.csr_matrix,
) -> sp.csr_matrix:
    """
    Calcule le dissipateur de Lindblad D[L] rho.

    D[L] rho = L rho L† - 1/2 {L†L, rho}
    """
    L_dag = L.conj().T.tocsr()
    term1 = L @ rho @ L_dag
    term2 = 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
    return (term1 - term2).tocsr()


# ----------------------------------------------------------------------
# États quantiques
# ----------------------------------------------------------------------

def fock_state(n: int, dim: int) -> np.ndarray:
    """Construit un état de Fock |n> dans une base tronquée."""
    if n < 0 or n >= dim:
        raise ValueError("n doit être dans [0, dim-1].")
    state = np.zeros(dim, dtype=complex)
    state[n] = 1.0
    return state


def coherent_state(alpha: complex, dim: int) -> np.ndarray:
    """
    Construit un état cohérent |alpha> dans une base de Fock tronquée.

    |alpha> = exp(-|alpha|^2 / 2) * sum_n alpha^n / sqrt(n!) |n>
    """
    if dim < 1:
        raise ValueError("dim doit être au moins 1.")
    coeffs = np.zeros(dim, dtype=complex)
    for n in range(dim):
        coeffs[n] = (alpha ** n) / math.sqrt(math.factorial(n))
    coeffs *= np.exp(-abs(alpha) ** 2 / 2.0)
    return normalize_state(coeffs)


def cat_state(
    alpha: complex,
    parity: str,
    dim: int,
) -> np.ndarray:
    """
    Construit un état chat approximé.

    |C_alpha^+> = N (|alpha> + |-alpha>)   (parité paire)
    |C_alpha^-> = N (|alpha> - |-alpha>)   (parité impaire)
    """
    if parity not in ("even", "odd"):
        raise ValueError("parity doit être 'even' ou 'odd'.")
    plus = coherent_state(alpha, dim)
    minus = coherent_state(-alpha, dim)
    if parity == "even":
        state = plus + minus
    else:
        state = plus - minus
    return normalize_state(state)


def orthogonalize_cats(
    alpha: complex,
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit deux états chats orthogonaux par Gram-Schmidt.

    Retourne (|0_L>, |1_L>) orthonormés.
    """
    even = cat_state(alpha, "even", dim)
    odd = cat_state(alpha, "odd", dim)

    overlap = np.vdot(even, odd)
    odd_orth = odd - overlap * even
    odd_orth = normalize_state(odd_orth)

    return even, odd_orth


def vacuum_state(dim: int) -> np.ndarray:
    """Construit l'état vide |0>."""
    return fock_state(0, dim)


# ----------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------

@dataclass
class KerrCatParameters:
    """
    Paramètres physiques d'un oscillateur Kerr piloté à deux photons.

    K         : force Kerr (positive = défocalisant)
    eps2      : amplitude du drive à deux photons
    eps1      : amplitude du drive à un photon (rotation logique)
    kappa     : taux de perte de photons
    gamma_phi : taux de déphasage
    dim       : dimension de la base de Fock tronquée
    """

    K: float = 1.0
    eps2: float = 0.5
    eps1: float = 0.0
    kappa: float = 0.01
    gamma_phi: float = 0.001
    dim: int = 30

    def __post_init__(self):
        if self.K <= 0:
            raise ValueError("K doit être strictement positif.")
        if self.kappa < 0:
            raise ValueError("kappa doit être non négatif.")
        if self.gamma_phi < 0:
            raise ValueError("gamma_phi doit être non négatif.")
        if self.dim < 5:
            raise ValueError("dim doit être au moins 5.")

    def to_dict(self) -> Dict[str, float]:
        """Sérialise les paramètres en dictionnaire."""
        return asdict(self)


@dataclass
class SimulationConfig:
    """
    Configuration numérique de la simulation.

    dt           : pas de temps
    n_steps      : nombre de pas
    record_every : fréquence d'enregistrement
    seed         : graine aléatoire
    """

    dt: float = 0.01
    n_steps: int = 1000
    record_every: int = 10
    seed: int = 42

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("dt doit être positif.")
        if self.n_steps < 0:
            raise ValueError("n_steps doit être non négatif.")
        if self.record_every < 1:
            raise ValueError("record_every doit être au moins 1.")

    def to_dict(self) -> Dict[str, float]:
        """Sérialise la configuration en dictionnaire."""
        return asdict(self)


# ----------------------------------------------------------------------
# Simulateur principal
# ----------------------------------------------------------------------

class KerrCatSimulator:
    """
    Simulateur d'un qubit de chat Kerr.

    Modèle :
    H = -K a†²a² + eps2 (a†² + a²) + eps1 (a† + a)

    Dissipation :
    kappa D[a] + gamma_phi D[n]

    Le simulateur utilise une base de Fock tronquée et une intégration
    RK4 de l'équation maîtresse de Lindblad.
    """

    def __init__(
        self,
        params: KerrCatParameters,
        config: SimulationConfig,
    ):
        self.params = params
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._build_operators()
        self._build_hamiltonian()
        self._build_jump_operators()
        self.rho_final: Optional[sp.csr_matrix] = None

    # ------------------------------------------------------------------
    # Construction des opérateurs
    # ------------------------------------------------------------------

    def _build_operators(self):
        """Construit les opérateurs de base."""
        d = self.params.dim
        self.a = annihilation_operator(d)
        self.a_dag = creation_operator(d)
        self.n = number_operator(d)
        self.parity = parity_operator(d)
        self.identity = identity_operator(d)

    def _build_hamiltonian(self):
        """Construit le hamiltonien Kerr + drives."""
        p = self.params
        a = self.a
        a_dag = self.a_dag

        # Terme Kerr
        H_kerr = -p.K * (a_dag @ a_dag @ a @ a)

        # Drive à deux photons
        H_two_photon = p.eps2 * (a_dag @ a_dag + a @ a)

        # Drive à un photon (rotation logique)
        H_one_photon = p.eps1 * (a_dag + a)

        self.H = (H_kerr + H_two_photon + H_one_photon).tocsr()

        if not is_hermitian(ensure_dense(self.H)):
            raise RuntimeError("Le hamiltonien n'est pas hermitien.")

    def _build_jump_operators(self):
        """Construit les opérateurs de saut de Lindblad."""
        p = self.params
        self.jump_ops: List[sp.csr_matrix] = []

        if p.kappa > 0:
            self.jump_ops.append(
                (np.sqrt(p.kappa) * self.a).tocsr()
            )

        if p.gamma_phi > 0:
            self.jump_ops.append(
                (np.sqrt(p.gamma_phi) * self.n).tocsr()
            )

    # ------------------------------------------------------------------
    # Équation maîtresse
    # ------------------------------------------------------------------

    def rhs(self, rho: sp.csr_matrix) -> sp.csr_matrix:
        """
        Calcule d rho / dt selon l'équation maîtresse de Lindblad.

        d rho / dt = -i [H, rho] + sum_j D[L_j] rho
        """
        comm = commutator(self.H, rho)
        drho = (-1j * comm).tocsr()

        for L in self.jump_ops:
            drho = (drho + dissipator(rho, L)).tocsr()

        return drho

    def step(self, rho: sp.csr_matrix, dt: float) -> sp.csr_matrix:
        """
        Fait un pas RK4 de l'équation maîtresse.

        RK4 :
        k1 = f(rho)
        k2 = f(rho + dt/2 k1)
        k3 = f(rho + dt/2 k2)
        k4 = f(rho + dt k3)
        rho_{n+1} = rho_n + dt/6 (k1 + 2k2 + 2k3 + k4)
        """
        k1 = self.rhs(rho)
        k2 = self.rhs((rho + 0.5 * dt * k1).tocsr())
        k3 = self.rhs((rho + 0.5 * dt * k2).tocsr())
        k4 = self.rhs((rho + dt * k3).tocsr())

        rho_new = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return rho_new.tocsr()

    def evolve(
        self,
        rho0: sp.csr_matrix,
        n_steps: Optional[int] = None,
        record_every: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Évolue l'état pendant n_steps pas.

        Retourne une liste de dictionnaires contenant les observables.
        """
        if n_steps is None:
            n_steps = self.config.n_steps
        if record_every is None:
            record_every = self.config.record_every

        rho = rho0.tocsr()
        history: List[Dict[str, float]] = []

        for step in range(n_steps):
            rho = self.step(rho, self.config.dt)

            if (step + 1) % record_every == 0:
                record = self.compute_observables(rho)
                record["step"] = step + 1
                history.append(record)

        self.rho_final = rho
        return history

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def compute_observables(self, rho: sp.csr_matrix) -> Dict[str, float]:
        """Calcule les observables physiques et logiques."""
        rho_dense = ensure_dense(rho)
        trace = trace_density_matrix(rho)

        if abs(trace - 1.0) > 1e-6:
            LOGGER.warning(
                "La trace n'est pas conservée : trace = %.6f", trace
            )

        # Nombre moyen de photons
        n_dense = ensure_dense(self.n)
        photon_number = float(np.real(np.trace(rho_dense @ n_dense)))

        # Parité moyenne
        parity_dense = ensure_dense(self.parity)
        parity_mean = float(np.real(np.trace(rho_dense @ parity_dense)))

        # Populations et fidélité logiques
        logical_pops = self.logical_populations(rho)
        fidelity = self.logical_fidelity(rho)

        return {
            "trace": trace,
            "photon_number": photon_number,
            "parity": parity_mean,
            "pop_0L": logical_pops["0L"],
            "pop_1L": logical_pops["1L"],
            "leakage": logical_pops["leakage"],
            "fidelity": fidelity,
        }

    def logical_populations(
        self,
        rho: sp.csr_matrix,
    ) -> Dict[str, float]:
        """
        Calcule les populations dans la base logique des chats.

        Retourne un dictionnaire avec pop_0L, pop_1L et leakage.
        """
        alpha = self._estimate_cat_amplitude()
        ket0, ket1 = orthogonalize_cats(alpha, self.params.dim)

        rho_dense = ensure_dense(rho)

        pop0 = float(np.real(np.vdot(ket0, rho_dense @ ket0)))
        pop1 = float(np.real(np.vdot(ket1, rho_dense @ ket1)))

        pop0 = clip_probability(pop0)
        pop1 = clip_probability(pop1)
        leakage = clip_probability(1.0 - pop0 - pop1)

        return {
            "0L": pop0,
            "1L": pop1,
            "leakage": leakage,
        }

    def logical_fidelity(self, rho: sp.csr_matrix) -> float:
        """
        Calcule la fidélité au sous-espace logique.

        F = pop_0L + pop_1L
        """
        pops = self.logical_populations(rho)
        return clip_probability(pops["0L"] + pops["1L"])

    def _estimate_cat_amplitude(self) -> complex:
        """
        Estime l'amplitude des chats stabilisés.

        Pour un oscillateur Kerr piloté à deux photons, l'amplitude
        stabilisée est approximativement alpha = sqrt(eps2 / K).
        """
        p = self.params
        if p.eps2 <= 0:
            return 0.0 + 0.0j
        return complex(np.sqrt(p.eps2 / p.K))

    # ------------------------------------------------------------------
    # États initiaux
    # ------------------------------------------------------------------

    def initial_state_vacuum(self) -> sp.csr_matrix:
        """Retourne la matrice densité de l'état vide."""
        ket = vacuum_state(self.params.dim)
        rho = np.outer(ket, ket.conj())
        return sp.csr_matrix(rho)

    def initial_state_cat(
        self,
        parity: str = "even",
    ) -> sp.csr_matrix:
        """Retourne la matrice densité d'un état chat."""
        alpha = self._estimate_cat_amplitude()
        ket = cat_state(alpha, parity, self.params.dim)
        rho = np.outer(ket, ket.conj())
        return sp.csr_matrix(rho)

    def initial_state_random(self) -> sp.csr_matrix:
        """Retourne une matrice densité aléatoire normalisée."""
        d = self.params.dim
        vec = self.rng.normal(size=d) + 1j * self.rng.normal(size=d)
        vec = normalize_state(vec)
        rho = np.outer(vec, vec.conj())
        return sp.csr_matrix(rho)

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def save_final_state(self, path: Path):
        """Sauvegarde l'état final en NPZ."""
        if self.rho_final is None:
            raise RuntimeError("Aucune évolution n'a été lancée.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            rho_dense=ensure_dense(self.rho_final),
            params=json.dumps(self.params.to_dict()),
            config=json.dumps(self.config.to_dict()),
        )
        LOGGER.info("État final sauvegardé : %s", path)


# ----------------------------------------------------------------------
# Expériences
# ----------------------------------------------------------------------

class ExperimentRunner:
    """
    Orchestre les expériences de stabilisation, de bruit et de sweeps.
    """

    def __init__(
        self,
        simulator: KerrCatSimulator,
        output_dir: Optional[Path] = None,
    ):
        self.sim = simulator
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_stabilization(
        self,
        initial_state: str = "vacuum",
    ) -> List[Dict[str, float]]:
        """
        Simule la stabilisation vers l'état chat.

        initial_state : 'vacuum', 'cat_even', 'cat_odd', 'random'
        """
        if initial_state == "vacuum":
            rho0 = self.sim.initial_state_vacuum()
        elif initial_state == "cat_even":
            rho0 = self.sim.initial_state_cat("even")
        elif initial_state == "cat_odd":
            rho0 = self.sim.initial_state_cat("odd")
        elif initial_state == "random":
            rho0 = self.sim.initial_state_random()
        else:
            raise ValueError("initial_state non reconnu.")

        history = self.sim.evolve(rho0)
        self._save_history(history, f"stabilization_{initial_state}.json")
        return history

    def run_noise_sweep(
        self,
        noise_type: str = "photon_loss",
        values: Optional[List[float]] = None,
    ) -> List[Dict[str, float]]:
        """
        Balaye un paramètre de bruit et mesure la fidélité logique.

        noise_type : 'photon_loss' ou 'dephasing'
        """
        if values is None:
            values = [0.0, 0.005, 0.01, 0.02, 0.05]

        results: List[Dict[str, float]] = []
        for value in values:
            params_dict = self.sim.params.to_dict()
            if noise_type == "photon_loss":
                params_dict["kappa"] = value
            elif noise_type == "dephasing":
                params_dict["gamma_phi"] = value
            else:
                raise ValueError("noise_type non reconnu.")

            params = KerrCatParameters(**params_dict)
            config = SimulationConfig(
                dt=self.sim.config.dt,
                n_steps=self.sim.config.n_steps,
                record_every=self.sim.config.record_every,
                seed=self.sim.config.seed,
            )
            sub_sim = KerrCatSimulator(params, config)
            history = sub_sim.evolve(sub_sim.initial_state_vacuum())
            final = history[-1] if history else {}
            results.append(
                {
                    "noise_type": noise_type,
                    "value": value,
                    "fidelity": final.get("fidelity", 0.0),
                    "leakage": final.get("leakage", 0.0),
                }
            )

        self._save_results(results, f"noise_sweep_{noise_type}.json")
        return results

    def run_parameter_sweep(
        self,
        param_name: str,
        values: List[float],
    ) -> List[Dict[str, float]]:
        """
        Balaye un paramètre physique et mesure la fidélité logique.

        param_name : 'K', 'eps2', 'eps1', 'kappa', 'gamma_phi'
        """
        if param_name not in ("K", "eps2", "eps1", "kappa", "gamma_phi"):
            raise ValueError("param_name non reconnu.")

        results: List[Dict[str, float]] = []
        for value in values:
            params_dict = self.sim.params.to_dict()
            params_dict[param_name] = value
            params = KerrCatParameters(**params_dict)
            config = SimulationConfig(
                dt=self.sim.config.dt,
                n_steps=self.sim.config.n_steps,
                record_every=self.sim.config.record_every,
                seed=self.sim.config.seed,
            )
            sub_sim = KerrCatSimulator(params, config)
            history = sub_sim.evolve(sub_sim.initial_state_vacuum())
            final = history[-1] if history else {}
            results.append(
                {
                    "param": param_name,
                    "value": value,
                    "fidelity": final.get("fidelity", 0.0),
                    "leakage": final.get("leakage", 0.0),
                    "photon_number": final.get("photon_number", 0.0),
                }
            )

        self._save_results(results, f"param_sweep_{param_name}.json")
        return results

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def _save_history(
        self,
        history: List[Dict[str, float]],
        filename: str,
    ):
        """Sauvegarde l'historique en JSON."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        LOGGER.info("Historique sauvegardé : %s", path)

    def _save_results(
        self,
        results: List[Dict[str, float]],
        filename: str,
    ):
        """Sauvegarde les résultats de sweep en JSON."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        LOGGER.info("Résultats sauvegardés : %s", path)


# ----------------------------------------------------------------------
# Tests unitaires
# ----------------------------------------------------------------------

def test_hermiticity():
    """Vérifie que le hamiltonien est hermitien."""
    params = KerrCatParameters()
    config = SimulationConfig()
    sim = KerrCatSimulator(params, config)
    H_dense = ensure_dense(sim.H)
    assert is_hermitian(H_dense), "Le hamiltonien n'est pas hermitien."


def test_trace_preservation():
    """Vérifie que la trace est conservée pendant l'évolution."""
    params = KerrCatParameters()
    config = SimulationConfig(n_steps=50)
    sim = KerrCatSimulator(params, config)
    rho0 = sim.initial_state_vacuum()
    history = sim.evolve(rho0)
    final_trace = history[-1]["trace"]
    assert abs(final_trace - 1.0) < 1e-4, "La trace n'est pas conservée."


def test_cat_orthogonality():
    """Vérifie que les états chats sont orthogonaux pour alpha grand."""
    alpha = 3.0
    dim = 50
    ket0, ket1 = orthogonalize_cats(alpha, dim)
    overlap = np.vdot(ket0, ket1)
    assert abs(overlap) < 1e-6, "Les chats ne sont pas orthogonaux."


def test_stabilization_improves_fidelity():
    """Vérifie que la fidélité augmente pendant la stabilisation."""
    params = KerrCatParameters(eps2=0.5)
    config = SimulationConfig(n_steps=200)
    sim = KerrCatSimulator(params, config)
    rho0 = sim.initial_state_vacuum()
    history = sim.evolve(rho0)
    initial_fidelity = history[0]["fidelity"]
    final_fidelity = history[-1]["fidelity"]
    assert final_fidelity > initial_fidelity, (
        "La fidélité n'augmente pas pendant la stabilisation."
    )


def run_all_tests():
    """Lance tous les tests unitaires."""
    test_hermiticity()
    test_trace_preservation()
    test_cat_orthogonality()
    test_stabilization_improves_fidelity()
    LOGGER.info("Tous les tests sont passés.")


# ----------------------------------------------------------------------
# Interface en ligne de commande
# ----------------------------------------------------------------------

def main():
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Simulateur de qubits de chat Kerr."
    )
    parser.add_argument("--K", type=float, default=1.0, help="Force Kerr.")
    parser.add_argument("--eps2", type=float, default=0.5, help="Drive 2 photons.")
    parser.add_argument("--eps1", type=float, default=0.0, help="Drive 1 photon.")
    parser.add_argument("--kappa", type=float, default=0.01, help="Perte de photons.")
    parser.add_argument("--gamma_phi", type=float, default=0.001, help="Déphasage.")
    parser.add_argument("--dim", type=int, default=30, help="Dimension de Fock.")
    parser.add_argument("--dt", type=float, default=0.01, help="Pas de temps.")
    parser.add_argument("--n_steps", type=int, default=1000, help="Nombre de pas.")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    parser.add_argument("--output", type=str, default="results", help="Dossier de sortie.")
    parser.add_argument("--test", action="store_true", help="Lance les tests.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.test:
        run_all_tests()
        return

    params = KerrCatParameters(
        K=args.K,
        eps2=args.eps2,
        eps1=args.eps1,
        kappa=args.kappa,
        gamma_phi=args.gamma_phi,
        dim=args.dim,
    )
    config = SimulationConfig(
        dt=args.dt,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    sim = KerrCatSimulator(params, config)
    runner = ExperimentRunner(sim, output_dir=Path(args.output))
    history = runner.run_stabilization("vacuum")
    LOGGER.info(
        "Stabilisation terminée. Fidélité finale : %.4f",
        history[-1]["fidelity"],
    )


if __name__ == "__main__":
    main()
