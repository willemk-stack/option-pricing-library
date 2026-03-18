from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "visual-state"
DOCS_VISUAL_CONFIG_PATH = ROOT / "scripts" / "visual_audit" / "docs_visual_config.json"
SVG_NS = {"svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}
HIGH_RISK_ASSETS = [
    Path("docs/assets/generated/showcase/reviewer_proof_panel.light.svg"),
    Path("docs/assets/generated/showcase/reviewer_proof_panel.dark.svg"),
    Path("docs/assets/generated/benchmarks/benchmark_overview.light.svg"),
    Path("docs/assets/generated/benchmarks/benchmark_overview.dark.svg"),
]


@dataclass
class Finding:
    severity: str
    category: str
    message: str
    file: str | None = None


@dataclass
class Report:
    summary: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def add(
        self, severity: str, category: str, message: str, file: Path | str | None = None
    ) -> None:
        self.findings.append(
            Finding(
                severity=severity,
                category=category,
                message=message,
                file=str(file) if file else None,
            )
        )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf8")


def parse_json(path: Path) -> dict:
    return json.loads(read_text(path))


def parse_simple_yaml_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text)
    if not match:
        return None
    value = match.group(1).strip()
    return value.strip("'\"")


def extract_default_docs_base_url(text: str) -> str | None:
    patterns = [
        r'DOCS_BASE_URL\s*\|\|\s*"([^"]+)"',
        r"DOCS_BASE_URL\s*=\s*\"([^\"]+)\"",
        r"\$env:DOCS_BASE_URL\s*=\s*\"([^\"]+)\"",
        r'DEFAULT_DOCS_BASE_URL\s*=\s*"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def screenshot_name(path: str, theme: str, project_name: str) -> str:
    safe_name = (
        "home"
        if path == "/"
        else re.sub(r"^-+|-+$", "", path.replace("/", "-").replace("_", "-"))
    )
    return f"{safe_name}-{theme}-{project_name}.png"


def expected_snapshot_names(
    pages: list[str], themes: list[str], widths: list[int]
) -> set[str]:
    names: set[str] = set()
    for path in pages:
        for theme in themes:
            for width in widths:
                names.add(screenshot_name(path, theme, f"chromium-{width}"))
    return names


def extract_href(element: ET.Element) -> str | None:
    return (
        element.get("{http://www.w3.org/1999/xlink}href")
        or element.get("href")
        or element.get("{http://www.w3.org/2000/svg}href")
    )


def scan_high_risk_svg(path: Path, report: Report) -> None:
    if not path.exists():
        report.add("high", "assets", "Expected high-risk SVG is missing", path)
        return

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        report.add("high", "assets", f"SVG parse error: {exc}", path)
        return

    root = tree.getroot()
    images = root.findall(".//svg:image", SVG_NS)
    local_linked = []
    external = []
    missing = []
    for image in images:
        href = extract_href(image)
        if not href:
            continue
        if href.startswith("data:"):
            continue
        if href.startswith(("http://", "https://")):
            external.append(href)
            continue
        target = (path.parent / href).resolve()
        if not target.exists():
            missing.append(href)
        else:
            local_linked.append(href)

    if missing:
        report.add("high", "assets", f"Missing linked sub-assets: {missing}", path)
    if external:
        report.add(
            "medium", "assets", f"External linked images inside SVG: {external}", path
        )
    if local_linked:
        report.add(
            "medium",
            "assets",
            f"SVG contains linked raster/image subpanels that need browser validation: {local_linked}",
            path,
        )

    if path.name.startswith("reviewer_proof_panel") and len(local_linked) < 3:
        report.add(
            "medium",
            "assets",
            f"Expected ~3 linked proof thumbnails, found {len(local_linked)}",
            path,
        )
    if path.name.startswith("benchmark_overview") and len(local_linked) < 3:
        report.add(
            "medium",
            "assets",
            f"Expected ~3 linked benchmark panels, found {len(local_linked)}",
            path,
        )


def workflow_has_visual_ci() -> bool:
    workflows = list((ROOT / ".github" / "workflows").glob("*.y*ml"))
    needles = [
        "playwright test",
        "check_svg_assets.py",
        "run_visual_audit.ps1",
        "run_visual_scan.ps1",
    ]
    for workflow in workflows:
        text = read_text(workflow)
        if any(needle in text for needle in needles):
            return True
    return False


def find_command_refs(paths: Iterable[Path], regex: str) -> list[tuple[str, int, str]]:
    pattern = re.compile(regex)
    matches: list[tuple[str, int, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        for i, line in enumerate(read_text(path).splitlines(), start=1):
            if pattern.search(line):
                matches.append((str(path.relative_to(ROOT)), i, line.strip()))
    return matches


def main() -> int:
    report = Report()

    review_targets_path = ROOT / "scripts" / "visual_audit" / "review_targets.json"
    mkdocs_path = ROOT / "mkdocs.yml"
    playwright_config = ROOT / "tests" / "visual" / "playwright.config.ts"
    pages_spec = ROOT / "tests" / "visual" / "pages.spec.ts"
    a11y_spec = ROOT / "tests" / "visual" / "a11y.spec.ts"
    serve_script = ROOT / "scripts" / "serve-docs.ps1"
    serve_script_py = ROOT / "scripts" / "serve_docs.py"
    run_script = ROOT / "scripts" / "run_visual_audit.ps1"
    snapshots_dir = ROOT / "tests" / "visual" / "pages.spec.ts-snapshots"

    targets = parse_json(review_targets_path)
    docs_visual_config = parse_json(DOCS_VISUAL_CONFIG_PATH)
    configured_base_url = docs_visual_config.get("docs_base_url")
    pages = targets.get("pages", [])
    themes = targets.get("themes", [])
    widths = targets.get("widths", [])

    expected = expected_snapshot_names(pages, themes, widths)
    actual = {path.name for path in snapshots_dir.glob("*.png")}
    missing = sorted(expected - actual)
    stale = sorted(actual - expected)

    mkdocs_text = read_text(mkdocs_path)
    playwright_text = read_text(playwright_config)
    serve_text = read_text(serve_script)
    run_text = read_text(run_script)
    pages_text = read_text(pages_spec)
    a11y_text = read_text(a11y_spec)

    site_url = parse_simple_yaml_value(mkdocs_text, "site_url")
    playwright_base = (
        extract_default_docs_base_url(playwright_text) or configured_base_url
    )
    serve_base = extract_default_docs_base_url(serve_text) or configured_base_url
    serve_py_text = read_text(serve_script_py)
    serve_py_base = extract_default_docs_base_url(serve_py_text) or configured_base_url
    run_base = extract_default_docs_base_url(run_text) or configured_base_url

    site_path = urlparse(site_url).path if site_url else None
    playwright_path = urlparse(playwright_base).path if playwright_base else None
    serve_path = urlparse(serve_base).path if serve_base else None
    serve_py_path = urlparse(serve_py_base).path if serve_py_base else None
    run_path = urlparse(run_base).path if run_base else None

    if site_path and playwright_path and site_path != playwright_path:
        report.add(
            "high",
            "config",
            f"mkdocs site_url path {site_path!r} does not match Playwright default path {playwright_path!r}",
            playwright_config,
        )
    if site_path and serve_path and site_path != serve_path:
        report.add(
            "high",
            "config",
            f"mkdocs site_url path {site_path!r} does not match serve-docs DOCS_BASE_URL path {serve_path!r}",
            serve_script,
        )
    if site_path and serve_py_path and site_path != serve_py_path:
        report.add(
            "high",
            "config",
            f"mkdocs site_url path {site_path!r} does not match serve_docs.py DOCS_BASE_URL path {serve_py_path!r}",
            serve_script_py,
        )
    if site_path and run_path and site_path != run_path:
        report.add(
            "high",
            "config",
            f"mkdocs site_url path {site_path!r} does not match run_visual_audit DOCS_BASE_URL path {run_path!r}",
            run_script,
        )

    if 'from "./targets"' in pages_text and 'from "./targets"' in a11y_text:
        report.add(
            "info",
            "targets",
            "pages.spec.ts and a11y.spec.ts both read targets from tests/visual/targets.ts",
        )
    else:
        report.add(
            "medium",
            "targets",
            "One or more visual specs may not be using the shared targets module",
        )

    if missing:
        report.add(
            "high",
            "snapshots",
            f"Missing snapshot baselines: {len(missing)}",
            snapshots_dir,
        )
    else:
        report.add(
            "info",
            "snapshots",
            f"Snapshot matrix is complete: {len(actual)} files",
            snapshots_dir,
        )

    if stale:
        report.add(
            "medium",
            "snapshots",
            f"Stale/unexpected snapshot files: {len(stale)}",
            snapshots_dir,
        )
    else:
        report.add(
            "info", "snapshots", "No stale snapshot filenames detected", snapshots_dir
        )

    for asset in HIGH_RISK_ASSETS:
        scan_high_risk_svg(ROOT / asset, report)

    if workflow_has_visual_ci():
        report.add(
            "info",
            "ci",
            "Detected at least one workflow that appears to run a visual-related gate",
        )
    else:
        report.add(
            "medium",
            "ci",
            "No obvious CI workflow currently runs Playwright or the visual audit scripts",
        )

    ref_paths = [
        ROOT / "README.md",
        ROOT / "docs" / "visual-audit.md",
        serve_script,
        serve_script_py,
        run_script,
    ]
    build_refs = find_command_refs(ref_paths, r"build_visual_artifacts\.py")
    if build_refs:
        report.add(
            "info",
            "docs",
            f"Found {len(build_refs)} references to build_visual_artifacts.py",
        )

    report.summary = {
        "pages": len(pages),
        "themes": len(themes),
        "widths": len(widths),
        "expected_snapshots": len(expected),
        "actual_snapshots": len(actual),
        "missing_snapshots": len(missing),
        "stale_snapshots": len(stale),
        "site_url": site_url,
        "playwright_base_url": playwright_base,
        "serve_docs_base_url": serve_base,
        "run_visual_audit_base_url": run_base,
        "visual_ci_present": workflow_has_visual_ci(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "report.json"
    md_path = OUT_DIR / "report.md"
    json_path.write_text(
        json.dumps(
            {
                "summary": report.summary,
                "findings": [asdict(f) for f in report.findings],
            },
            indent=2,
        ),
        encoding="utf8",
    )

    by_severity = {
        key: [f for f in report.findings if f.severity == key]
        for key in ["high", "medium", "info"]
    }
    lines = [
        "# Visual state report",
        "",
        "## Summary",
        "",
        *[f"- **{key}**: `{value}`" for key, value in report.summary.items()],
        "",
    ]
    for severity in ["high", "medium", "info"]:
        lines.append(f"## {severity.title()} findings")
        lines.append("")
        items = by_severity[severity]
        if not items:
            lines.append("- None")
        else:
            for finding in items:
                suffix = f" (`{finding.file}`)" if finding.file else ""
                lines.append(f"- **{finding.category}**: {finding.message}{suffix}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")

    return 1 if by_severity["high"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
