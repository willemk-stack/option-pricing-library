from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "build_provider_evidence_artifacts.py"
)
spec = importlib.util.spec_from_file_location(
    "build_provider_evidence_artifacts", MODULE_PATH
)
assert spec is not None
assert spec.loader is not None
build_provider_evidence_artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_provider_evidence_artifacts)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_build_provider_evidence_artifacts_writes_sanitized_public_bundle(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    provider_summary = tmp_path / "provider_public_summary.json"
    output_dir = tmp_path / "generated" / "provider_evidence"

    _write_json(
        provider_summary,
        {
            "underlying": "SPY",
            "asof": "2026-05-22T15:59:00Z",
            "run_id": "live-smoke",
            "provider_label": "Alpaca",
            "feed_label": "FRED",
            "rate_policy": "fred_dgs1",
            "dividend_policy": "flat_forward_dividend_yield",
            "raw_contract_count": 5,
            "normalized_contract_count": 5,
            "api_key": "must-not-leak",
            "latest_quote": {"bid": 1.0, "ask": 2.0},
        },
    )
    _write_json(
        bundle_root / "manifest.json",
        {
            "artifact_schema_version": "model_validation_bundle.v1",
            "run_id": "live-smoke",
            "underlying": "SPY",
            "valuation_date": "2026-05-22",
            "rows": {"cleaned_quotes": 4, "rejected_quotes": 1},
            "reason_counts": {"spread_too_wide": 1},
            "heston_smoke": {
                "status": "success",
                "objective_type": "iv_rmse",
                "best_cost": 0.012,
            },
        },
    )
    _write_json(
        bundle_root / "warnings.json",
        {"warnings": ["demo warning"], "data_quality": [], "heston_smoke": []},
    )

    pd.DataFrame(
        {
            "expiry": ["2026-06-30", "2026-06-30", "2026-09-30", "2026-09-30"],
            "expiry_years": [0.10, 0.10, 0.35, 0.35],
            "strike": [95, 100, 100, 105],
            "spot": [100, 100, 100, 100],
            "right": ["put", "call", "put", "call"],
            "mid": [1.2, 2.1, 4.3, 3.9],
            "iv": [0.20, 0.19, 0.22, 0.21],
        }
    ).to_csv(bundle_root / "cleaned_quotes.csv", index=False)
    pd.DataFrame({"rejection_reason": ["spread_too_wide"]}).to_csv(
        bundle_root / "rejected_quotes.csv", index=False
    )
    pd.DataFrame(
        {
            "expiry": ["2026-06-30", "2026-09-30"],
            "strike": [100, 100],
            "right": ["call", "put"],
            "iv": [0.19, 0.22],
        }
    ).to_csv(bundle_root / "heston_quotes.csv", index=False)
    pd.DataFrame(
        {
            "status": ["success"],
            "objective": ["iv_rmse"],
            "selected_quote_count": [2],
            "best_cost": [0.012],
            "kappa": [1.2],
            "vbar": [0.04],
            "eta": [0.8],
            "rho": [-0.5],
            "v": [0.05],
        }
    ).to_csv(bundle_root / "heston_fit_summary.csv", index=False)

    exit_code = build_provider_evidence_artifacts.main(
        [
            "--bundle-root",
            str(bundle_root),
            "--provider-summary-json",
            str(provider_summary),
            "--output-dir",
            str(output_dir),
            "--profile",
            "smoke",
        ]
    )

    assert exit_code == 0
    data_dir = output_dir / "data"
    manifest_path = data_dir / "provider_evidence_manifest.json"
    assert manifest_path.exists()

    expected_paths = [
        output_dir / "provider_evidence_summary_card.light.svg",
        output_dir / "provider_evidence_summary_card.dark.svg",
        output_dir / "provider_pipeline_flow.light.svg",
        output_dir / "provider_quote_cleaning_waterfall.light.png",
        output_dir / "provider_expiry_strike_coverage.light.png",
        output_dir / "provider_heston_fit_summary.light.png",
        data_dir / "provider_public_summary.json",
        data_dir / "provider_quote_cleaning_summary.csv",
        data_dir / "provider_rejection_reason_counts.csv",
        data_dir / "provider_expiry_coverage.csv",
        data_dir / "provider_model_ready_summary.json",
        data_dir / "provider_heston_fit_summary.csv",
        data_dir / "provider_heston_parameter_summary.csv",
        data_dir / "provider_warnings.json",
    ]
    for path in expected_paths:
        assert path.exists(), path

    public_summary_text = (
        (data_dir / "provider_public_summary.json").read_text(encoding="utf-8").lower()
    )
    assert "must-not-leak" not in public_summary_text
    assert "api_key" not in public_summary_text
    assert "latest_quote" not in public_summary_text
    assert "bid" not in public_summary_text
    assert "ask" not in public_summary_text

    public_summary_payload = json.loads(
        (data_dir / "provider_public_summary.json").read_text(encoding="utf-8")
    )
    assert public_summary_payload["underlying"] == "SPY"
    assert public_summary_payload["accepted_quote_count"] == 4
    assert public_summary_payload["rejected_quote_count"] == 1
    assert public_summary_payload["selected_quote_count"] == 2
    assert public_summary_payload["heston_fit_status"] == "success"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_run_id"] == "live-smoke"
    assert manifest["profile"] == "smoke"
    assert manifest["rebuild_command"]
    assert manifest["caveats"]
    artifact_names = {artifact["filename"] for artifact in manifest["artifacts"]}
    assert "provider_evidence_summary_card.light.svg" in artifact_names
    assert "provider_public_summary.json" in artifact_names
