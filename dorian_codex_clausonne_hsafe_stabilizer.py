# dorian_codex_clausonne_hsafe_stabilizer.py
"""
Dorian Codex Protocol for AI – Clausonne Case Study
H_SAFE(t) = T(t) + V(t) - Z(t)

Python/JAX implementation for the historical-genealogical-link repo #19
(Studio SDFB Creative Dev Lab, 2026)

Author: Stefano Dorian Franco (ORCID: 0009-0007-4714-1627)
License: CC BY 4.0 (Open Science / Open Source)

References:
- Official Source-reference for DORIAN CODEX H_SAFE (Open Library, 2025)
- DC Clockwork (blueprint edition), 2025
- LinkedIn announcement (2026): Python/PyTorch & JAX implementation
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
from typing import Tuple, Dict, Any


class ClausonneHSafeStabilizer:
    """
    Secured Cognitive Hamiltonian for the Clausonne case study.

    H_SAFE(t) = T(t) + V(t) - Z(t)

    Where:
    - T(t): Semantic Kinetic Energy (information flux: genealogy, history, maps).
    - V(t): Ontological Potential (anchoring in authority sources, real territory).
    - Z(t): Entropy / Loss (noise, uncertainties, gaps in the historical record).

    This class implements:
    - A differentiable H_SAFE scalar function.
    - A simple gradient-based update of internal weights (lambda_T, lambda_V, lambda_Z).
    - A toy 'time' dimension representing the continuum 1464–2026.
    """

    def __init__(self, init_weights: Dict[str, float] | None = None):
        """
        Initialize weights for T, V, Z contributions.

        Default: equal weights (1.0, 1.0, 1.0).
        """
        if init_weights is None:
            init_weights = {"lambda_T": 1.0, "lambda_V": 1.0, "lambda_Z": 1.0}

        self.lambda_T = jnp.array(init_weights.get("lambda_T", 1.0))
        self.lambda_V = jnp.array(init_weights.get("lambda_V", 1.0))
        self.lambda_Z = jnp.array(init_weights.get("lambda_Z", 1.0))

    def h_safe(
        self,
        T: jnp.ndarray,
        V: jnp.ndarray,
        Z: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute H_SAFE(t) = T(t) + V(t) - Z(t), with learnable weights.

        Parameters
        ----------
        T : jnp.ndarray
            Semantic Kinetic Energy vector (time series or batch).
        V : jnp.ndarray
            Ontological Potential vector.
        Z : jnp.ndarray
            Entropy / Loss vector.

        Returns
        -------
        H : jnp.ndarray
            Secured Cognitive Hamiltonian scalar (or vector, depending on inputs).
        """
        H = (self.lambda_T * T) + (self.lambda_V * V) - (self.lambda_Z * Z)
        return H

    @staticmethod
    def toy_clausonne_signals(
        time_steps: int = 10,
        seed: int = 42,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate toy signals for T, V, Z representing the Clausonne continuum.

        This is a didactic example, not a historical model.

        - T: increases with 'modern' information flux (archives digitized, IA, etc.).
        - V: high and stable (strong anchoring in sources and territory).
        - Z: higher in early periods (uncertainties, gaps), lower in recent times.

        Parameters
        ----------
        time_steps : int
            Number of discrete time steps (e.g., centuries, decades).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        T, V, Z : jnp.ndarray
            Toy signals for Semantic Kinetic Energy, Ontological Potential, Entropy.
        """
        key = jax.random.PRNGKey(seed)
        t = jnp.linspace(0, 1, time_steps)

        # Toy models:
        # T: grows with time (more data, more flux)
        T = jnp.linspace(0.3, 1.0, time_steps)
        # V: strong anchor, slight noise
        V = 0.9 + 0.05 * jax.random.normal(key, shape=(time_steps,))
        # Z: higher in the past, lower now
        Z = jnp.linspace(0.8, 0.3, time_steps)

        return T, V, Z

    def update_weights_via_gradient(
        self,
        T: jnp.ndarray,
        V: jnp.ndarray,
        Z: jnp.ndarray,
        target_H: float | jnp.ndarray = 1.0,
        lr: float = 0.01,
        steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Adjust lambda_T, lambda_V, lambda_Z via gradient descent on H_SAFE.

        Objective: bring mean(H_SAFE) close to a target_H (e.g., 1.0).

        This is a minimal illustration of the 'Law of Cognitive Evolution'
        from the Dorian Codex Clockwork (automatic calibration via grad(H_SAFE)).

        Parameters
        ----------
        T, V, Z : jnp.ndarray
            Input signals.
        target_H : float or jnp.ndarray
            Desired average value for H_SAFE.
        lr : float
            Learning rate.
        steps : int
            Number of gradient steps.

        Returns
        -------
        history : dict
            Dictionary with lists of lambda_* and loss over steps.
        """

        def loss_fn(lT: jnp.ndarray, lV: jnp.ndarray, lZ: jnp.ndarray) -> jnp.ndarray:
            H = (lT * T) + (lV * V) - (lZ * Z)
            return jnp.mean((H - target_H) ** 2)

        grad_fn = grad(loss_fn, argnums=(0, 1, 2))

        lT, lV, lZ = self.lambda_T, self.lambda_V, self.lambda_Z

        history = {
            "lambda_T": [],
            "lambda_V": [],
            "lambda_Z": [],
            "loss": [],
        }

        for _ in range(steps):
            gT, gV, gZ = grad_fn(lT, lV, lZ)
            lT = lT - lr * gT
            lV = lV - lr * gV
            lZ = lZ - lr * gZ

            loss_val = loss_fn(lT, lV, lZ)

            history["lambda_T"].append(float(lT))
            history["lambda_V"].append(float(lV))
            history["lambda_Z"].append(float(lZ))
            history["loss"].append(float(loss_val))

        self.lambda_T = lT
        self.lambda_V = lV
        self.lambda_Z = lZ

        return history


@jit
def demo_clausonne_hsafe_pipeline(
    time_steps: int = 20,
    seed: int = 42,
    lr: float = 0.01,
    steps: int = 100,
) -> Dict[str, Any]:
    """
    End-to-end demo: generate toy Clausonne signals and calibrate H_SAFE.

    Returns a dictionary with:
    - T, V, Z signals
    - H before and after calibration
    - history of weights and loss
    """
    stabilizer = ClausonneHSafeStabilizer()

    T, V, Z = stabilizer.toy_clausonne_signals(time_steps=time_steps, seed=seed)

    H_before = stabilizer.h_safe(T, V, Z)

    history = stabilizer.update_weights_via_gradient(
        T, V, Z, target_H=1.0, lr=lr, steps=steps
    )

    H_after = stabilizer.h_safe(T, V, Z)

    return {
        "T": T,
        "V": V,
        "Z": Z,
        "H_before": H_before,
        "H_after": H_after,
        "lambda_T": stabilizer.lambda_T,
        "lambda_V": stabilizer.lambda_V,
        "lambda_Z": stabilizer.lambda_Z,
        "history": history,
    }


if __name__ == "__main__":
    # Simple demo when running:
    # python dorian_codex_clausonne_hsafe_stabilizer.py

    print("Dorian Codex Protocol – Clausonne H_SAFE Stabilizer (JAX)")
    print("=" * 60)

    result = demo_clausonne_hsafe_pipeline(
        time_steps=20,
        seed=42,
        lr=0.01,
        steps=100,
    )

    print("
Final weights:")
    print(f"lambda_T = {result['lambda_T']:.4f}")
    print(f"lambda_V = {result['lambda_V']:.4f}")
    print(f"lambda_Z = {result['lambda_Z']:.4f}")

    print("
H_SAFE before calibration (mean):")
    print(f"{jnp.mean(result['H_before']):.4f}")

    print("
H_SAFE after calibration (mean):")
    print(f"{jnp.mean(result['H_after']):.4f}")

    print("
Last 5 loss values:")
    print(result["history"]["loss"][-5:])
