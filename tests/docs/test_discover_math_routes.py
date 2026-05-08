from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "visual_audit"))

discover_math_routes = importlib.import_module("discover_math_routes")
normalize_route = discover_math_routes.normalize_route
select_math_routes = discover_math_routes.select_math_routes


def test_normalize_route_preserves_root_and_trailing_slash() -> None:
    assert normalize_route("/") == "/"
    assert normalize_route("performance") == "/performance/"
    assert normalize_route("/performance") == "/performance/"
    assert normalize_route("/performance/") == "/performance/"


def test_select_math_routes_returns_all_routes_without_filters() -> None:
    routes, message = select_math_routes(
        ["/", "/performance/", "/user_guides/surface_workflow/"],
        None,
    )

    assert routes == ["/", "/performance/", "/user_guides/surface_workflow/"]
    assert message is None


def test_select_math_routes_intersects_with_review_paths() -> None:
    routes, message = select_math_routes(
        ["/", "/performance/", "/user_guides/surface_workflow/"],
        ["/performance/", "/architecture/"],
    )

    assert routes == ["/performance/"]
    assert message is None


def test_select_math_routes_returns_skip_message_when_filter_has_no_math() -> None:
    routes, message = select_math_routes(
        ["/", "/performance/", "/user_guides/surface_workflow/"],
        ["/architecture/"],
    )

    assert routes == []
    assert message is not None
    assert "/architecture/" in message
