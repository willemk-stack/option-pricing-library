from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "visual-state"
SPECS_DIR = ROOT / "design" / "page_specs"


@dataclass(frozen=True, slots=True)
class PageScore:
    page_key: str
    archetype: str
    page_path: str
    structure_score: int
    evidence_score: int
    navigation_score: int
    theme_readiness_score: int
    total_score: int
    suggestions: list[str]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf8")


def load_specs() -> list[dict]:
    specs: list[dict] = []
    for path in sorted(SPECS_DIR.glob("*.yml")):
        specs.append(yaml.safe_load(read_text(path)))
    return specs


def count_headings(text: str) -> int:
    return len(re.findall(r"(?m)^#", text))


def count_tables(text: str) -> int:
    return len(re.findall(r"(?m)^\| .* \|$", text))


def count_diagrams(text: str) -> int:
    return text.count("<figure") + text.count('class="diagram"')


def has_cta_row(text: str) -> bool:
    return "cta-row" in text or "md-button" in text


def has_snapshot_grid(text: str) -> bool:
    return "snapshot-grid" in text


def has_theme_pair(text: str) -> bool:
    return ".diagram-light" in text and ".diagram-dark" in text


def score_spec(spec: dict) -> PageScore:
    page_path = ROOT / spec["page_path"]
    text = read_text(page_path)
    archetype = str(spec.get("archetype", "unknown"))
    suggestions: list[str] = []

    heading_count = count_headings(text)
    table_count = count_tables(text)
    diagram_count = count_diagrams(text)
    cta_present = has_cta_row(text)
    snapshot_present = has_snapshot_grid(text)
    theme_pair_present = has_theme_pair(text)

    structure_score = 25
    if heading_count < 3:
        structure_score -= 10
        suggestions.append(
            "Strengthen heading rhythm so the page scan path is clearer."
        )
    if text.count("\n# ") != 1 and not text.startswith("# "):
        structure_score -= 5
        suggestions.append(
            "Keep the page anchored by a single clear top-level heading."
        )

    evidence_score = 25
    if archetype in {
        "homepage",
        "long-form guide",
        "visual/report page",
        "visual-report",
        "long-form-guide",
    }:
        if diagram_count == 0:
            evidence_score -= 15
            suggestions.append(
                "Add or strengthen a primary visual evidence block near the top of the page."
            )
    if (
        archetype in {"homepage", "visual/report page", "visual-report"}
        and table_count == 0
    ):
        evidence_score -= 5
        suggestions.append(
            "Add a compact summary table or metric block to sharpen the proof narrative."
        )
    if (
        archetype in {"homepage", "long-form guide", "long-form-guide"}
        and not snapshot_present
    ):
        evidence_score -= 5
        suggestions.append(
            "Use grouped evidence blocks so figures feel curated rather than isolated."
        )

    navigation_score = 25
    if (
        archetype
        in {
            "homepage",
            "section landing page",
            "section-landing",
            "long-form guide",
            "long-form-guide",
        }
        and not cta_present
    ):
        navigation_score -= 10
        suggestions.append(
            "Add or strengthen CTA placement so the next click is obvious."
        )

    theme_readiness_score = 25
    if diagram_count > 0 and not theme_pair_present:
        theme_readiness_score -= 15
        suggestions.append(
            "Ensure visual blocks have equivalent light and dark treatment."
        )

    total_score = (
        structure_score + evidence_score + navigation_score + theme_readiness_score
    )

    if total_score >= 90:
        suggestions.append(
            "Page is in strong shape; focus on bounded polish rather than structural change."
        )
    elif total_score >= 75:
        suggestions.append(
            "Page is structurally solid; the next gains are in evidence presentation and spacing polish."
        )
    else:
        suggestions.append(
            "Page needs a bounded improvement pass before it should be treated as a quality reference page."
        )

    deduped_suggestions: list[str] = []
    for suggestion in suggestions:
        if suggestion not in deduped_suggestions:
            deduped_suggestions.append(suggestion)

    return PageScore(
        page_key=str(spec["page_key"]),
        archetype=archetype,
        page_path=str(spec["page_path"]),
        structure_score=structure_score,
        evidence_score=evidence_score,
        navigation_score=navigation_score,
        theme_readiness_score=theme_readiness_score,
        total_score=total_score,
        suggestions=deduped_suggestions[:3],
    )


def render_markdown(scores: list[PageScore]) -> str:
    lines = [
        "# Improvement report",
        "",
        "This report is a non-blocking quality pass derived from the current page specs under `design/page_specs/`.",
        "",
        "| Page | Archetype | Score | Top suggestions |",
        "| --- | --- | ---: | --- |",
    ]

    for score in scores:
        suggestions = "<br>".join(score.suggestions)
        lines.append(
            f"| `{score.page_path}` | `{score.archetype}` | `{score.total_score}` | {suggestions} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = [score_spec(spec) for spec in load_specs()]
    payload = {
        "scores": [asdict(score) for score in scores],
    }
    (OUT_DIR / "improvement-report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf8"
    )
    (OUT_DIR / "improvement-report.md").write_text(
        render_markdown(scores), encoding="utf8"
    )
    print(
        f"Wrote {(OUT_DIR / 'improvement-report.md')} and {(OUT_DIR / 'improvement-report.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
