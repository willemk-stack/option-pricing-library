from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ArrayLike = float | FloatArray


@dataclass(frozen=True, slots=True)
class Smile:
    T: float  # years
    x: FloatArray  # log-moneyness grid: ln(K/F(T))
    w: FloatArray  # total variance: T * iv^2

    def w_at(self, xq: ArrayLike) -> FloatArray:
        xq_arr = np.asarray(xq, dtype=np.float64)

        # np.interp is often typed as returning Any -> wrap with np.asarray to satisfy mypy
        out = np.interp(
            xq_arr,
            self.x,
            self.w,
            left=float(self.w[0]),
            right=float(self.w[-1]),
        )
        return np.asarray(out, dtype=np.float64)

    def iv_at(self, xq: ArrayLike) -> FloatArray:
        wq = self.w_at(xq)
        out = np.sqrt(np.maximum(wq / np.float64(self.T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class VolSurface:
    expiries: FloatArray
    smiles: tuple[Smile, ...]
    forward: Callable[[float], float]  # forward(T) -> float

    @classmethod
    def from_grid(
        cls,
        rows: Iterable[tuple[float, float, float]],  # (T, K, iv)
        forward: Callable[[float], float],
        *,
        expiry_round_decimals: int = 10,
    ) -> VolSurface:
        buckets: dict[float, list[tuple[float, float]]] = {}

        for T_raw, K_raw, iv_raw in rows:
            T_key = round(float(T_raw), expiry_round_decimals)
            buckets.setdefault(T_key, []).append((float(K_raw), float(iv_raw)))

        expiries = np.asarray(sorted(buckets.keys()), dtype=np.float64)
        smiles: list[Smile] = []

        for T_np in expiries:
            pts = buckets[float(T_np)]
            pts.sort(key=lambda p: p[0])  # sort by strike

            K = np.asarray([p[0] for p in pts], dtype=np.float64)
            iv = np.asarray([p[1] for p in pts], dtype=np.float64)

            F = float(forward(float(T_np)))
            if F <= 0.0:
                raise ValueError(
                    f"forward(T) must be > 0, got {F} at T_np={float(T_np)}"
                )
            if np.any(K <= 0.0):
                raise ValueError(
                    f"All strikes must be > 0 for log-moneyness at T={float(T_np)}"
                )

            x = np.log(K / np.float64(F)).astype(np.float64, copy=False)
            w = (np.float64(T_np) * (iv**2)).astype(np.float64, copy=False)

            if np.any(np.diff(x) <= 0):
                raise ValueError(
                    f"x grid not strictly increasing at T_np={float(T_np)}"
                )

            smiles.append(Smile(T=float(T_np), x=x, w=w))

        return cls(expiries=expiries, smiles=tuple(smiles), forward=forward)

    def iv(self, K: ArrayLike, T: float) -> FloatArray:
        T = float(T)
        if T <= 0:
            raise ValueError("T must be > 0")

        K_arr = np.asarray(K, dtype=np.float64)
        if np.any(K_arr <= 0.0):
            raise ValueError("Strikes must be > 0 for log-moneyness.")

        xq = np.log(K_arr / np.float64(self.forward(T))).astype(np.float64, copy=False)

        # Clamp outside known range
        if T <= float(self.expiries[0]):
            return self.smiles[0].iv_at(xq)
        if T >= float(self.expiries[-1]):
            return self.smiles[-1].iv_at(xq)

        # Find bracketing expiries
        j = int(np.searchsorted(self.expiries, np.float64(T)))
        i = j - 1

        T0 = float(
            self.expiries[i]
        )  # <- force scalar float (fixes your assignment errors)
        T1 = float(self.expiries[j])

        s0 = self.smiles[i]
        s1 = self.smiles[j]

        w0 = s0.w_at(xq)
        w1 = s1.w_at(xq)

        a = (T - T0) / (T1 - T0)
        w = np.float64(1.0 - a) * w0 + np.float64(a) * w1
        out = np.sqrt(np.maximum(w / np.float64(T), np.float64(0.0)))
        return np.asarray(out, dtype=np.float64)
