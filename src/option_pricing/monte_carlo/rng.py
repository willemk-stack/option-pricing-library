"""Random number generation helpers for Monte Carlo simulation.

Responsibilities:
- construct NumPy random generators from ``RandomConfig`` or ``MCConfig``
- draw reproducible standard-normal arrays
- produce correlated normal shocks for multi-factor models
- build antithetic pairs without making model simulators own random state

Model simulators should consume normals or Brownian increments supplied by this
module rather than constructing their own RNGs internally.
"""

import numpy as np

from .config import MCConfig, RandomConfig


def rng_from_random_config(rc: RandomConfig) -> np.random.Generator:
    """Construct a NumPy RNG from a `RandomConfig`.

    Notes
    -----
    - ``pcg64`` uses `numpy.random.default_rng`.
    - ``mt19937`` uses `numpy.random.MT19937`.
    - ``sobol`` is not provided by NumPy; raise to make the limitation explicit.
    """
    seed = int(rc.seed)
    if rc.rng_type == "pcg64":
        return np.random.default_rng(seed)
    if rc.rng_type == "mt19937":
        return np.random.Generator(np.random.MT19937(seed))
    if rc.rng_type == "sobol":
        raise NotImplementedError(
            "Sobol sequences are not available in NumPy. "
            "Either pass an explicit rng=... in MCConfig, or choose rng_type='pcg64'/'mt19937'."
        )
    # Defensive: RandomConfig is type-restricted, but keep runtime robust.
    raise ValueError(f"Unknown rng_type: {rc.rng_type!r}")


def make_rng(cfg: MCConfig) -> np.random.Generator:
    """Select the RNG for MC from MCConfig."""
    return cfg.rng if cfg.rng is not None else rng_from_random_config(cfg.random)


def standard_normals(
    rng: np.random.Generator,
    n_paths: int,
    sample_shape: tuple[int, ...] = (),
    *,
    antithetic: bool = False,
    dtype: np.dtype[np.floating] | type[np.floating] = np.float64,
) -> np.ndarray:
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    if antithetic and n_paths % 2 != 0:
        raise ValueError("antithetic=True requires an even n_paths")

    draw_paths = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((draw_paths, *sample_shape))

    if antithetic:
        z = np.concatenate([z, -z], axis=0)

    return np.asarray(z, dtype=dtype)


def correlated_normals(
    rng: np.random.Generator,
    n_paths: int,
    corr: np.ndarray,
    sample_shape: tuple[int, ...] = (),
    *,
    antithetic: bool = False,
    dtype: np.dtype[np.floating] | type[np.floating] = np.float64,
) -> np.ndarray:
    """
    Generate correlated standard normals.

    `sample_shape` excludes both the path dimension and the factor dimension.

    Examples
    --------
    Terminal 2-factor draw:
        n_paths=1000, sample_shape=(), corr shape=(2, 2)
        returns shape=(1000, 2)

    Pathwise 2-factor draw:
        n_paths=1000, sample_shape=(252,), corr shape=(2, 2)
        returns shape=(1000, 252, 2)
    """
    corr = np.asarray(corr, dtype=float)

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square matrix")

    if not np.allclose(corr, corr.T):
        raise ValueError("corr must be symmetric")

    if not np.allclose(np.diag(corr), 1.0):
        raise ValueError("corr must have ones on the diagonal")

    n_factors = corr.shape[0]

    try:
        # Cholesky gives corr = L @ L.T
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("corr must be positive definite") from exc

    z_ind = standard_normals(
        rng,
        n_paths=n_paths,
        sample_shape=(*sample_shape, n_factors),
        antithetic=antithetic,
        dtype=dtype,
    )

    z_corr = z_ind @ L.T
    return np.asarray(z_corr, dtype=dtype)
