from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from math import comb, exp
from typing import Literal, overload

from ..market.curves import avg_carry_from_forward, avg_rate_from_df
from ..models.binomial_crr import BinomialModel
from ..types import OptionType, PricingContext, PricingInputs

# ----------------------------
# Payoff helpers (ruff-friendly)
# ----------------------------


def _call_payoff(S: float, *, K: float) -> float:
    return max(S - K, 0.0)


def _put_payoff(S: float, *, K: float) -> float:
    return max(K - S, 0.0)


# ----------------------------
# Tree structures
# ----------------------------


@dataclass(slots=True)
class BinomialNode:
    step: int
    index: int
    stock_price: float
    option_price: float | None = None
    up: BinomialNode | None = None
    down: BinomialNode | None = None


@dataclass(slots=True)
class BinomialTree:
    model: BinomialModel
    root: BinomialNode
    nodes: list[list[BinomialNode]]  # nodes[step][index]

    @classmethod
    def from_model(cls, model: BinomialModel) -> BinomialTree:
        nodes = cls._build_nodes(model)
        return cls(model=model, root=nodes[0][0], nodes=nodes)

    @staticmethod
    def _build_nodes(model: BinomialModel) -> list[list[BinomialNode]]:
        nodes: list[list[BinomialNode]] = []

        for step in range(model.n_steps + 1):
            level: list[BinomialNode] = []
            for index in range(step + 1):
                stock_price = model.S0 * (model.u**index) * (model.d ** (step - index))
                level.append(
                    BinomialNode(step=step, index=index, stock_price=stock_price)
                )
            nodes.append(level)

        for step in range(model.n_steps):
            for index in range(step + 1):
                cur = nodes[step][index]
                cur.up = nodes[step + 1][index + 1]
                cur.down = nodes[step + 1][index]

        return nodes


# ----------------------------
# Pricing engines (with overloads)
# ----------------------------


@overload
def price_european_tree(
    model: BinomialModel,
    payoff: Callable[[float], float],
    *,
    return_tree: Literal[False] = False,
) -> float: ...


@overload
def price_european_tree(
    model: BinomialModel,
    payoff: Callable[[float], float],
    *,
    return_tree: Literal[True],
) -> BinomialTree: ...


def price_european_tree(
    model: BinomialModel,
    payoff: Callable[[float], float],
    *,
    return_tree: bool = False,
) -> float | BinomialTree:
    """
    European pricing via backward induction on a recombining binomial tree.
    """
    tree = BinomialTree.from_model(model)

    for node in tree.nodes[-1]:
        node.option_price = payoff(node.stock_price)

    p = model.p_star
    disc = model.disc_step

    for step in range(model.n_steps - 1, -1, -1):
        for node in tree.nodes[step]:
            if node.up is None or node.down is None:
                raise RuntimeError("Tree node pointers not configured correctly.")
            if node.up.option_price is None or node.down.option_price is None:
                raise RuntimeError("Missing option prices during backward induction.")

            node.option_price = disc * (
                p * node.up.option_price + (1.0 - p) * node.down.option_price
            )

    if return_tree:
        return tree

    if tree.root.option_price is None:
        raise RuntimeError("Root option price was not computed.")
    return float(tree.root.option_price)


@overload
def price_american_tree(
    model: BinomialModel,
    payoff: Callable[[float], float],
    *,
    return_tree: Literal[False] = False,
) -> float: ...


@overload
def price_american_tree(
    model: BinomialModel,
    payoff: Callable[[float], float],
    *,
    return_tree: Literal[True],
) -> BinomialTree: ...


def price_american_tree(
    model: BinomialModel,
    payoff: Callable[[float], float],
    *,
    return_tree: bool = False,
) -> float | BinomialTree:
    """
    American pricing: backward induction with early exercise.
    """
    tree = BinomialTree.from_model(model)

    for node in tree.nodes[-1]:
        node.option_price = payoff(node.stock_price)

    p = model.p_star
    disc = model.disc_step

    for step in range(model.n_steps - 1, -1, -1):
        for node in tree.nodes[step]:
            if node.up is None or node.down is None:
                raise RuntimeError("Tree node pointers not configured correctly.")
            if node.up.option_price is None or node.down.option_price is None:
                raise RuntimeError("Missing option prices during backward induction.")

            cont = disc * (
                p * node.up.option_price + (1.0 - p) * node.down.option_price
            )
            ex = payoff(node.stock_price)
            node.option_price = max(cont, ex)

    if return_tree:
        return tree

    if tree.root.option_price is None:
        raise RuntimeError("Root option price was not computed.")
    return float(tree.root.option_price)


def price_european_closed_form(
    model: BinomialModel, payoff: Callable[[float], float]
) -> float:
    """
    European pricing via the binomial distribution (closed-form sum).
    """
    N = model.n_steps
    p = model.p_star
    disc = exp(-model.r * model.T)

    total = 0.0
    for j in range(N + 1):
        S_T = model.S0 * (model.u**j) * (model.d ** (N - j))
        total += comb(N, j) * (p**j) * ((1.0 - p) ** (N - j)) * payoff(S_T)

    return disc * total


# ----------------------------
# PricingInputs wrappers (generic dispatch uses kind)
# ----------------------------


def _model_from_inputs(p: PricingInputs, n_steps: int) -> BinomialModel:
    ctx = p.ctx
    df = ctx.df(p.tau)
    F = ctx.fwd(p.tau)
    r = avg_rate_from_df(df, p.tau)
    b = avg_carry_from_forward(ctx.spot, F, p.tau)
    q = r - b
    return BinomialModel.from_crr(
        S0=ctx.spot,
        r=r,
        q=q,
        sigma=p.sigma,
        T=p.tau,
        n_steps=n_steps,
    )


def _model_from_ctx(
    ctx: PricingContext, *, tau: float, sigma: float, n_steps: int
) -> BinomialModel:
    """Build a CRR binomial model from curves-first inputs."""
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    df = ctx.df(tau)
    F = ctx.fwd(tau)
    r = avg_rate_from_df(df, tau)
    b = avg_carry_from_forward(ctx.spot, F, tau)
    q = r - b
    return BinomialModel.from_crr(
        S0=ctx.spot,
        r=r,
        q=q,
        sigma=float(sigma),
        T=tau,
        n_steps=int(n_steps),
    )


def binom_price_from_ctx(
    *,
    kind: OptionType,
    strike: float,
    sigma: float,
    tau: float,
    ctx: PricingContext,
    n_steps: int,
    american: bool = False,
    method: Literal["tree", "closed_form"] = "tree",
) -> float:
    """Binomial (CRR) price from curves-first inputs.

    Parameters are expressed in **time-to-expiry** ``tau``. Discounting and forward
    information is supplied via ``ctx`` (discount and forward curves).
    """
    model = _model_from_ctx(
        ctx, tau=float(tau), sigma=float(sigma), n_steps=int(n_steps)
    )
    payoff: Callable[[float], float]
    if kind == OptionType.CALL:
        payoff = partial(_call_payoff, K=float(strike))
    elif kind == OptionType.PUT:
        payoff = partial(_put_payoff, K=float(strike))
    else:
        raise ValueError("Unsupported option kind")

    if american:
        if method != "tree":
            raise ValueError("American options require method='tree'.")
        return price_american_tree(model, payoff)

    if method == "tree":
        return price_european_tree(model, payoff)
    return price_european_closed_form(model, payoff)


def binom_price(
    p: PricingInputs,
    n_steps: int,
    *,
    american: bool = False,
    method: Literal["tree", "closed_form"] = "tree",
) -> float:
    """
    Generic binomial pricing using p.spec.kind (CALL/PUT).

    method:
      - "tree": backward induction (Euro or American)
      - "closed_form": Euro-only binomial sum (fast, no early exercise)
    """
    model = _model_from_inputs(p, n_steps)

    if p.spec.kind == OptionType.CALL:
        payoff: Callable[[float], float] = partial(_call_payoff, K=p.K)
    elif p.spec.kind == OptionType.PUT:
        payoff = partial(_put_payoff, K=p.K)
    else:
        raise ValueError(f"Unsupported option kind: {p.spec.kind}")

    if american:
        if method != "tree":
            raise ValueError("American options require method='tree'.")
        return price_american_tree(model, payoff)

    if method == "tree":
        return price_european_tree(model, payoff)
    return price_european_closed_form(model, payoff)


def binom_price_call(
    p: PricingInputs,
    n_steps: int,
    *,
    american: bool = False,
    method: Literal["tree", "closed_form"] = "tree",
) -> float:
    model = _model_from_inputs(p, n_steps)
    payoff: Callable[[float], float] = partial(_call_payoff, K=p.K)

    if american:
        if method != "tree":
            raise ValueError("American options require method='tree'.")
        return price_american_tree(model, payoff)

    if method == "tree":
        return price_european_tree(model, payoff)
    return price_european_closed_form(model, payoff)


def binom_price_put(
    p: PricingInputs,
    n_steps: int,
    *,
    american: bool = False,
    method: Literal["tree", "closed_form"] = "tree",
) -> float:
    model = _model_from_inputs(p, n_steps)
    payoff: Callable[[float], float] = partial(_put_payoff, K=p.K)

    if american:
        if method != "tree":
            raise ValueError("American options require method='tree'.")
        return price_american_tree(model, payoff)

    if method == "tree":
        return price_european_tree(model, payoff)
    return price_european_closed_form(model, payoff)


# Backwards-compatible names (if your old notebooks use these)
def binom_call_from_inputs(p: PricingInputs, n_steps: int) -> float:
    return binom_price_call(p, n_steps)


def binom_put_from_inputs(p: PricingInputs, n_steps: int) -> float:
    return binom_price_put(p, n_steps)
