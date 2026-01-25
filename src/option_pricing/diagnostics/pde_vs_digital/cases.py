from __future__ import annotations

from option_pricing.types import DigitalSpec, MarketData, OptionType, PricingInputs


def _mk(
    *,
    name: str,
    S: float,
    K: float,
    tau: float,
    sigma: float,
    payout: float = 1.0,
    r: float = 0.02,
    q: float = 0.00,
    kind: OptionType = OptionType.CALL,
    t: float = 0.0,
) -> tuple[str, PricingInputs[DigitalSpec]]:
    """Central constructor for a single digital option case."""

    expiry = float(t) + float(tau)
    spec = DigitalSpec(
        kind=kind,
        strike=float(K),
        expiry=float(expiry),
        payout=float(payout),
    )
    market = MarketData(
        spot=float(S),
        rate=float(r),
        dividend_yield=float(q),
    )
    p = PricingInputs(
        spec=spec,
        market=market,
        sigma=float(sigma),
        t=float(t),
    )
    return name, p


def digital_suite(
    *,
    S0: float = 100.0,
    payout: float = 1.0,
    r: float = 0.02,
    q: float = 0.00,
) -> list[tuple[str, PricingInputs[DigitalSpec]]]:
    """Compact digital suite across moneyness/tenor/vol for both call and put."""

    cases: list[tuple[str, PricingInputs[DigitalSpec]]] = []

    tenors = {
        "1w": 1.0 / 52.0,
        "1m": 1.0 / 12.0,
        "3m": 0.25,
        "1y": 1.0,
    }

    vols = {
        "low": 0.10,
        "mid": 0.20,
        "high": 0.50,
    }

    moneyness = {
        "OTM": 1.10,
        "ATM": 1.00,
        "ITM": 0.90,
    }

    for t_lbl, tau in tenors.items():
        for v_lbl, sigma in vols.items():
            for m_lbl, m in moneyness.items():
                K = m * S0

                cases.append(
                    _mk(
                        name=f"call_{m_lbl}_{t_lbl}_{v_lbl}",
                        S=S0,
                        K=K,
                        tau=tau,
                        sigma=sigma,
                        payout=payout,
                        r=r,
                        q=q,
                        kind=OptionType.CALL,
                    )
                )
                cases.append(
                    _mk(
                        name=f"put_{m_lbl}_{t_lbl}_{v_lbl}",
                        S=S0,
                        K=K,
                        tau=tau,
                        sigma=sigma,
                        payout=payout,
                        r=r,
                        q=q,
                        kind=OptionType.PUT,
                    )
                )

    return cases


def stress_suite(
    *,
    S0: float = 100.0,
    payout: float = 1.0,
    r: float = 0.02,
    q: float = 0.00,
) -> list[tuple[str, PricingInputs[DigitalSpec]]]:
    """Stress cases for digitals (discontinuous payoff)."""

    return [
        _mk(
            name="call_deep_otm_ultrashort",
            S=S0,
            K=1.50 * S0,
            tau=1.0 / 365.0,
            sigma=0.30,
            payout=payout,
            r=r,
            q=q,
            kind=OptionType.CALL,
        ),
        _mk(
            name="put_deep_itm_ultrashort",
            S=S0,
            K=1.50 * S0,
            tau=1.0 / 365.0,
            sigma=0.30,
            payout=payout,
            r=r,
            q=q,
            kind=OptionType.PUT,
        ),
        _mk(
            name="call_atm_veryhighvol_6m",
            S=S0,
            K=1.00 * S0,
            tau=0.5,
            sigma=1.20,
            payout=payout,
            r=r,
            q=q,
            kind=OptionType.CALL,
        ),
        _mk(
            name="put_atm_veryhighvol_6m",
            S=S0,
            K=1.00 * S0,
            tau=0.5,
            sigma=1.20,
            payout=payout,
            r=r,
            q=q,
            kind=OptionType.PUT,
        ),
        _mk(
            name="call_deep_itm_highvol_long",
            S=S0,
            K=0.50 * S0,
            tau=2.0,
            sigma=0.80,
            payout=payout,
            r=r,
            q=q,
            kind=OptionType.CALL,
        ),
        _mk(
            name="put_deep_otm_highvol_long",
            S=S0,
            K=0.50 * S0,
            tau=2.0,
            sigma=0.80,
            payout=payout,
            r=r,
            q=q,
            kind=OptionType.PUT,
        ),
    ]


def default_cases(
    *,
    S0: float = 100.0,
    payout: float = 1.0,
    r: float = 0.02,
    q: float = 0.00,
    include_stress: bool = True,
) -> list[tuple[str, PricingInputs[DigitalSpec]]]:
    """Recommended set for routine regression + decision diagnostics."""

    cases = digital_suite(S0=S0, payout=payout, r=r, q=q)
    if include_stress:
        cases.extend(stress_suite(S0=S0, payout=payout, r=r, q=q))
    return cases


__all__ = [
    "digital_suite",
    "stress_suite",
    "default_cases",
]
