"""Notebook helpers for the integration demo dashboard."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

from option_pricing.demos.capstone2 import run_capstone2

DisplayFn = Callable[..., None]
MarkdownFactory = Callable[[str], Any]


def _fallback_display(*objects: Any) -> None:
    for obj in objects:
        print(obj)


def _fallback_clear_output(*_args: Any, **_kwargs: Any) -> None:
    return None


def _fallback_markdown(text: str) -> str:
    return text


try:
    _ipython_display = import_module("IPython.display")
except Exception:
    Markdown: MarkdownFactory = _fallback_markdown
    clear_output: DisplayFn = _fallback_clear_output
    display: DisplayFn = _fallback_display
else:
    Markdown = cast(
        MarkdownFactory,
        getattr(_ipython_display, "Markdown", _fallback_markdown),
    )
    clear_output = cast(
        DisplayFn,
        getattr(_ipython_display, "clear_output", _fallback_clear_output),
    )
    display = cast(DisplayFn, getattr(_ipython_display, "display", _fallback_display))


@dataclass
class DemoDashboard:
    """Stateful dashboard helper for the integration demo notebook."""

    _demo_profile: str = "quick"
    _seed: int = 7
    _show_verbose_tables: bool = False
    _show_advanced: bool = False
    _overrides: dict[str, Any] | None = None
    art: Any | None = None
    _last_run_sig: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        default_profile = os.environ.get("OP_DEMO_PROFILE", "quick")
        default_seed = int(os.environ.get("OP_DEMO_SEED", "7"))
        default_verbose = os.environ.get("OP_SHOW_VERBOSE_TABLES", "").lower() in {
            "1",
            "true",
            "yes",
        }

        if default_profile in {"quick", "full"}:
            self._demo_profile = default_profile
        self._seed = int(default_seed)
        self._show_verbose_tables = bool(default_verbose)
        self.rebuild_overrides()

    @property
    def DEMO_PROFILE(self) -> str:  # noqa: N802 - keep notebook-friendly names
        return self._demo_profile

    @property
    def SEED(self) -> int:  # noqa: N802
        return int(self._seed)

    @property
    def SHOW_VERBOSE_TABLES(self) -> bool:  # noqa: N802
        return bool(self._show_verbose_tables)

    @property
    def SHOW_ADVANCED(self) -> bool:  # noqa: N802
        return bool(self._show_advanced)

    @property
    def OVERRIDES(self) -> dict[str, Any]:  # noqa: N802
        if self._overrides is None:
            self._overrides = {}
        return self._overrides

    def rebuild_overrides(self) -> None:
        """Recompute overrides from current profile flags."""

        self._show_advanced = self._demo_profile == "full"
        self._overrides = {
            "SYNTH_CFG": {"noise_level": 0.001, "outlier_prob": 0.0},
            "RUN_*": {
                "RUN_DUPIRE_VS_GATHERAL_COMPARE": self._show_advanced,
                "RUN_DIGITAL_PDE_BASELINE": self._show_advanced,
                "RUN_LOCALVOL_CONVERGENCE_SWEEP": self._show_advanced,
                "RUN_LOCALVOL_REPRICING": True,
                "RUN_EXPLICIT_SVI_REPAIR_DEMO": True,
            },
        }

    def _run_signature(self) -> tuple[Any, ...]:
        return (self._demo_profile, int(self._seed), repr(self.OVERRIDES))

    def ensure_art(self, *, force: bool = False):
        """Compute artifacts if needed; otherwise reuse cached artifacts."""

        self.rebuild_overrides()
        sig = self._run_signature()

        if force or (self.art is None) or (self._last_run_sig != sig):
            import time

            t0 = time.time()
            self.art = run_capstone2(
                profile=self._demo_profile,
                seed=int(self._seed),
                overrides=self.OVERRIDES,
            )
            self._last_run_sig = sig
            dt = time.time() - t0
            print(
                f"[run_capstone2] profile={self._demo_profile} seed={self._seed} done in {dt:.2f}s"
            )

        return self.art

    def show_table(self, title: str, df, *, max_rows: int = 12) -> None:
        """Display a captioned table with truncation by default."""

        display(Markdown(f"### {title}"))

        if df is None:
            display(Markdown("_No data._"))
            return

        try:
            import pandas as pd  # type: ignore

            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
        except Exception:
            pass

        try:
            n_rows = len(df)
        except Exception:
            display(df)
            return

        n = int(max_rows)
        if (not self._show_verbose_tables) and n_rows > n:
            display(df.head(n))
            display(
                Markdown(
                    f"_Showing first {n} of {n_rows} rows. Toggle `Verbose tables` to show full tables._"
                )
            )
        else:
            display(df)

    def display(self) -> None:
        """Render the ipywidgets dashboard (if available)."""

        try:
            import ipywidgets as widgets  # type: ignore[import-not-found]
        except Exception:
            self.rebuild_overrides()
            print("ipywidgets not available; using defaults/env only.")
            print("Install with: %pip install ipywidgets")
            return

        profile_toggle = widgets.ToggleButtons(
            options=[("Quick", "quick"), ("Full", "full")],
            value=self._demo_profile,
            description="Profile:",
        )
        seed_widget = widgets.BoundedIntText(
            value=int(self._seed), min=0, max=10_000_000, description="Seed:"
        )
        verbose_toggle = widgets.Checkbox(
            value=self._show_verbose_tables, description="Verbose tables"
        )

        apply_btn = widgets.Button(description="Apply", button_style="primary")
        run_btn = widgets.Button(description="Run pipeline", button_style="success")
        force_btn = widgets.Button(description="Force rerun", button_style="warning")

        status_out = widgets.Output()
        run_out = widgets.Output()

        running = {"value": False}

        def set_buttons_enabled(enabled: bool) -> None:
            apply_btn.disabled = not enabled
            run_btn.disabled = not enabled
            force_btn.disabled = not enabled

        def apply_state(*, quiet: bool = False) -> None:
            self._demo_profile = profile_toggle.value
            self._seed = int(seed_widget.value)
            self._show_verbose_tables = bool(verbose_toggle.value)
            self.rebuild_overrides()

            if not quiet:
                with status_out:
                    clear_output()
                    display(
                        Markdown(
                            "**Applied** → "
                            f"`DEMO_PROFILE='{self._demo_profile}'`, "
                            f"`SHOW_ADVANCED={self._show_advanced}`, "
                            f"`SEED={self._seed}`, "
                            f"`SHOW_VERBOSE_TABLES={self._show_verbose_tables}`"
                        )
                    )

        def run_pipeline(*, force: bool = False) -> None:
            if running["value"]:
                with status_out:
                    display(
                        Markdown(
                            "⏳ Pipeline is already running — ignoring extra click."
                        )
                    )
                return

            running["value"] = True
            set_buttons_enabled(False)

            apply_state(quiet=True)

            with run_out:
                clear_output(wait=True)
                display(
                    Markdown("⏳ Running pipeline… (buttons disabled until finished)")
                )

                try:
                    art = self.ensure_art(force=force)
                    clear_output(wait=True)
                    display(
                        Markdown("✅ **Pipeline ready** (`art` computed / cached).")
                    )

                    qs = (
                        getattr(art, "tables", {}).get("quote_summary") if art else None
                    )
                    if qs is not None:
                        self.show_table("Quote summary by expiry", qs, max_rows=12)
                except Exception:
                    clear_output()
                    import traceback

                    print("❌ Pipeline failed. Traceback:\n")
                    print(traceback.format_exc())
                finally:
                    running["value"] = False
                    set_buttons_enabled(True)
                    apply_state(quiet=False)

        def on_apply_clicked(_):
            apply_state(quiet=False)

        def on_run_clicked(_):
            run_pipeline(force=False)

        def on_force_clicked(_):
            run_pipeline(force=True)

        apply_btn.on_click(on_apply_clicked)
        run_btn.on_click(on_run_clicked)
        force_btn.on_click(on_force_clicked)

        def mark_pending(_change=None):
            with status_out:
                clear_output()
                display(
                    Markdown(
                        "⚙️ Settings changed — click **Run pipeline** (it will auto-apply)."
                    )
                )

        profile_toggle.observe(mark_pending, names="value")
        seed_widget.observe(mark_pending, names="value")
        verbose_toggle.observe(mark_pending, names="value")

        display(
            widgets.HBox(
                [
                    profile_toggle,
                    seed_widget,
                    verbose_toggle,
                    apply_btn,
                    run_btn,
                    force_btn,
                ]
            )
        )
        display(status_out)
        display(run_out)

        apply_state(quiet=False)


IntegrationDemoDashboard = DemoDashboard
