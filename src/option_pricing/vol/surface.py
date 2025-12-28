from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

ArrayLike = float | np.ndarray


@dataclass(frozen=True, slots=True)
class Smile:
    T: float  # years
    x: np.ndarray  # log-moneyness grid: ln(K/F(T))
    w: np.ndarray  # total variance: T * iv^2

    def w_at(self, xq: ArrayLike) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        return np.interp(xq, self.x, self.w, left=self.w[0], right=self.w[-1])

    def iv_at(self, xq: ArrayLike) -> np.ndarray:
        wq = self.w_at(xq)
        return np.sqrt(np.maximum(wq / self.T, 0.0))


@dataclass(frozen=True, slots=True)
class VolSurface:
    expiries: np.ndarray
    smiles: tuple[Smile, ...]
    forward: Callable[[float], float]  # forward(T) -> float

    # -------------------------
    # Construction
    # -------------------------
    @classmethod
    def from_grid(
        cls,
        rows: Iterable[tuple[float, float, float]],  # (T, K, iv)
        forward,
        *,
        expiry_round_decimals: int = 10,
    ):
        buckets: dict[float, list[tuple[float, float]]] = {}

        for T, K, iv in rows:
            T_key = round(float(T), expiry_round_decimals)
            buckets.setdefault(T_key, []).append((float(K), float(iv)))

        expiries = np.array(sorted(buckets.keys()), dtype=float)
        smiles: list[Smile] = []

        for T in expiries:
            pts = buckets[T]
            pts.sort(key=lambda p: p[0])  # sort by strike

            K = np.array([p[0] for p in pts], dtype=float)
            iv = np.array([p[1] for p in pts], dtype=float)

            F = float(forward(T))
            x = np.log(K / F)  # <-- log-moneyness grid
            w = T * (iv**2)  # <-- total variance grid

            # important: x must be strictly increasing for np.interp
            if np.any(np.diff(x) <= 0):
                raise ValueError(
                    f"x grid not strictly increasing at T={T} (check strikes/forward)"
                )

            smiles.append(Smile(T=T, x=x, w=w))

        return cls(expiries=expiries, smiles=tuple(smiles), forward=forward)

    # -------------------------
    # Query
    # -------------------------

    def iv(self, K: ArrayLike, T: float) -> np.ndarray:
        T = float(T)
        if T <= 0:
            raise ValueError("T must be > 0")

        K_arr = np.asarray(K, dtype=float)
        xq = np.log(K_arr / float(self.forward(T)))  # <-- convert query strike to x

        # simplest (no time interp): pick nearest expiry or clamp
        if T <= self.expiries[0]:
            return self.smiles[0].iv_at(xq)
        if T >= self.expiries[-1]:
            return self.smiles[-1].iv_at(xq)

        # time interpolation in total variance at fixed x
        j = int(np.searchsorted(self.expiries, T))
        i = j - 1
        T0, T1 = self.expiries[i], self.expiries[j]
        s0, s1 = self.smiles[i], self.smiles[j]

        w0 = s0.w_at(xq)
        w1 = s1.w_at(xq)

        a = (T - T0) / (T1 - T0)
        w = (1.0 - a) * w0 + a * w1

        return np.sqrt(np.maximum(w / T, 0.0))
