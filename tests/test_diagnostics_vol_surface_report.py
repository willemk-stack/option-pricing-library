import numpy as np


def test_noarb_worst_points_and_run_surface_diagnostics():
    from option_pricing.diagnostics.vol_surface import (
        noarb_worst_points,
        run_surface_diagnostics,
    )
    from option_pricing.vol.arbitrage import check_surface_noarb
    from option_pricing.vol.surface import VolSurface

    def forward(T):
        return 100.0

    def df(T):
        return 1.0

    expiries = [0.5, 1.0, 2.0]
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    # Intentionally inconsistent quotes to trigger proxy no-arb violations.
    # (high vol on the right wing can raise far OTM call prices above nearer strikes)
    ivs = {
        0.5: np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
        1.0: np.array([0.3, 0.3, 0.3, 0.3, 3.0]),
        2.0: np.array([0.35, 0.35, 0.35, 0.35, 2.0]),
    }

    rows = []
    for T in expiries:
        for K, iv in zip(strikes, ivs[T], strict=True):
            rows.append((T, float(K), float(iv)))

    surface = VolSurface.from_grid(rows, forward=forward)
    rep = check_surface_noarb(surface, df=df)

    worst = noarb_worst_points(surface, rep, forward=forward, df=df, top_n=5)

    # We expect at least one of the proxy checks to flag.
    assert worst.summary["report_ok"] is False
    assert (len(worst.monotonicity) + len(worst.convexity) + len(worst.calendar)) > 0

    # Ensure the key columns exist (actionable tables).
    if not worst.monotonicity.empty:
        assert {"T", "K_left", "K_right", "dC"}.issubset(worst.monotonicity.columns)
    if not worst.convexity.empty:
        assert {"T", "K_mid", "violation"}.issubset(worst.convexity.columns)
    if not worst.calendar.empty:
        assert {"T0", "T1", "x", "dW", "violation"}.issubset(worst.calendar.columns)

    diag = run_surface_diagnostics(surface, forward=forward, df=df)
    assert "noarb_smiles" in diag.tables
    assert "noarb_worst_monotonicity" in diag.tables
    assert "noarb_worst_convexity" in diag.tables
    assert "noarb_worst_calendar" in diag.tables

    # JSON serialization should work (without arrays by default).
    s = diag.to_json()
    assert isinstance(s, str) and len(s) > 10
