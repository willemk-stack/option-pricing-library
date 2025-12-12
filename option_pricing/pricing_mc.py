from dataclasses import dataclass, field
import numpy as np
from collections.abc import Callable
from .params import PricingInputs
from .processes import sim_gbm_terminal

@dataclass(frozen=True, slots=True)
class McGBMModel:
    S0: float
    r: float
    sigma: float
    T: float
    n_paths: int
    rng: np.random.Generator = field(
        default_factory=np.random.default_rng,
        repr=False,  # don't spam repr with generator
    )

    def simulate_terminal(self) -> np.ndarray:
        # Thin wrapper around processes.sim_gbm_terminal
        return sim_gbm_terminal(
            n_paths=self.n_paths,
            T=self.T,
            mu=self.r,
            sigma=self.sigma,
            S0=self.S0,
            rng=self.rng,   # if you implemented rng support
        )

    def price_european(self, payoff: Callable[[np.ndarray], np.ndarray]) -> tuple[float, float]:
        ST = self.simulate_terminal()
        payoff_vals = payoff(ST)
        disc = np.exp(-self.r * self.T)
        price = disc * payoff_vals.mean()
        std_err = disc * payoff_vals.std(ddof=1) / np.sqrt(self.n_paths)
        return price, std_err

    def price_call(self, K: float) -> tuple[float, float]:
        return self.price_european(lambda ST: np.maximum(ST - K, 0.0))

    def price_put(self, K: float) -> tuple[float, float]:
        return self.price_european(lambda ST: np.maximum(K - ST, 0.0))
    

def mc_call_from_inputs(p: PricingInputs, n_paths: int) -> tuple[float, float]:
    effective_T = p.T - p.t  # time to maturity
    model = McGBMModel(
        S0=p.S,
        r=p.r,
        sigma=p.sigma,
        T=effective_T,   # <-- T is now tau, time-to-maturity
        n_paths=n_paths,
    )
    return model.price_call(p.K)

def mc_put_from_inputs(p: PricingInputs, n_paths: int) -> tuple[float, float]:
    effective_T = p.T - p.t  # time to maturity
    model = McGBMModel(
        S0=p.S,
        r=p.r,
        sigma=p.sigma,
        T=effective_T,   # <-- T is now tau, time-to-maturity
        n_paths=n_paths,
    )
    return model.price_put(p.K)


