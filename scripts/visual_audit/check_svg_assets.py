from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

SVG_NS = {"svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}


def extract_href(element: ET.Element) -> str | None:
    return (
        element.get("{http://www.w3.org/1999/xlink}href")
        or element.get("href")
        or element.get("{http://www.w3.org/2000/svg}href")
    )


def check_svg(path: Path) -> list[str]:
    issues: list[str] = []

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return [f"{path}: XML parse error: {exc}"]

    root = tree.getroot()
    images = root.findall(".//svg:image", SVG_NS)

    for image in images:
        href = extract_href(image)
        if not href:
            issues.append(f"{path}: <image> missing href")
            continue

        if href.startswith("data:"):
            continue

        if href.startswith(("http://", "https://")):
            issues.append(f"{path}: external linked image: {href}")
            continue

        target = (path.parent / href).resolve()
        if not target.exists():
            issues.append(f"{path}: missing linked asset: {href}")

    return issues


def main() -> int:
    svg_paths = sorted(Path("docs/assets/generated").rglob("*.svg"))
    if not svg_paths:
        print("No generated SVGs found under docs/assets/generated")
        return 0

    all_issues: list[str] = []
    for svg_path in svg_paths:
        all_issues.extend(check_svg(svg_path))

    if all_issues:
        print("SVG audit issues found:")
        for issue in all_issues:
            print(f"- {issue}")
        return 1

    print(f"Checked {len(svg_paths)} SVG files: no linked-asset issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
