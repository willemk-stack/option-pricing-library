from __future__ import annotations # Recommended for forward references
from math import comb, exp, sqrt
from dataclasses import dataclass
from collections.abc import Callable
from ..types import PricingInputs

@dataclass(frozen=True, slots=True)
class BinomialModel:
    S0: float      # initial stock price
    u: float       # up factor
    d: float       # down factor
    r: float       # risk-free rate (per unit time)
    dt: float      # time step
    n_steps: int      # number of steps

    def __post_init__(self) -> None:
            if self.n_steps <= 0:
                raise ValueError("n_steps must be positive")

            if not (0 < self.d < self.u):
                raise ValueError("Need 0 < d < u")
            
    @classmethod
    def from_crr(cls, S0: float, r: float, sigma: float, T: float, n_steps: int) -> "BinomialModel":
        dt = T / n_steps
        u = exp(sigma * sqrt(dt))
        d = exp(-sigma * sqrt(dt))
        return cls(S0=S0, u=u, d=d, r=r, dt=dt, n_steps=n_steps)
    
    @property
    def T(self) -> float:
        return self.dt * self.n_steps

    @property
    def p_star(self) -> float:
        # risk-neutral probability (CRR with continuous compounding)
        return (exp(self.r * self.dt) - self.d) / (self.u - self.d)
    
    # --- generic helper
    def price_european(self, payoff: Callable[[float], float]) -> float:
        N = self.n_steps
        p = self.p_star
        disc_factor = exp(-self.r*self.T)
        sum = 0
        for j in range(N + 1):
            # Stock price at step N with j up moves
            S_T = self.S0 * (self.u ** j) * (self.d ** (N - j))
            sum += comb(N, j) * p**j * (1-p)**(N-j) * payoff(S_T)

        return disc_factor*sum
    
    # --- the "helpers" for call/put, using the generic method ---
    def price_european_call(self, K: float) -> float:
        return self.price_european(lambda S: max(S - K, 0.0))

    def price_european_put(self, K: float) -> float:
        return self.price_european(lambda S: max(K - S, 0.0))



@dataclass(slots=True)
class BinomialNode:
    step: int
    index: int
    stock_price: float
    option_price: float | None = None
    up: "BinomialNode" | None = None
    down: "BinomialNode" | None = None

@dataclass(slots=True)
class BinomialTree:
    model: BinomialModel
    root: BinomialNode
    nodes: list[list[BinomialNode]]   # <- 2D structure [step][index]

    @classmethod
    def from_model(cls, model: BinomialModel) -> "BinomialTree":
        nodes = cls._build_nodes(model)
        root = nodes[0][0]
        return cls(model=model, root=root, nodes=nodes)

    @staticmethod
    def _build_nodes(model: BinomialModel) -> list[list[BinomialNode]]:
        nodes: list[list[BinomialNode]] = []
        
        #Cycle n step
        for step in range(model.n_steps+1):
            #Initialize list containing all nodes at step n
            level: list[BinomialNode] = []

            #Cycle indices of step n
            for index in range(step + 1):
                #Calc stock price
                stock_price = model.S0 * (model.u ** index) * (model.d ** (step - index))

                #Define node as instance of BinomialNode
                node = BinomialNode(
                    step = step,
                    index = index,
                    stock_price = stock_price,
                )
                level.append(node)
            nodes.append(level)

        # Configure up and down pointers execpt at last step
        for step in range(model.n_steps):
            for index in range(step + 1):
                current = nodes[step][index]
                current.up = nodes[step + 1][index + 1] # Move 1 index up
                current.down = nodes[step + 1][index] # stay at same index

        return nodes
    
def binom_call_from_inputs(p: PricingInputs, n_steps: int) -> float:
    effective_T = p.T - p.t

    model = BinomialModel.from_crr(
        S0=p.S,
        r=p.r,
        sigma=p.sigma,
        T=effective_T,
        n_steps=n_steps,
    )
    return model.price_european_call(p.K)

def binom_put_from_inputs(p: PricingInputs, n_steps: int) -> float:
    effective_T = p.T - p.t

    model = BinomialModel.from_crr(
        S0=p.S,
        r=p.r,
        sigma=p.sigma,
        T=effective_T,
        n_steps=n_steps,
    )
    return model.price_european_put(p.K)