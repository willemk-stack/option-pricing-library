from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

SVG_NS = {"svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}
GRAPHIC_TAGS = {
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "text",
    "image",
    "use",
}
ROOT = Path(__file__).resolve().parents[2]


def extract_href(element: ET.Element) -> str | None:
    return (
        element.get("{http://www.w3.org/1999/xlink}href")
        or element.get("href")
        or element.get("{http://www.w3.org/2000/svg}href")
    )


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_dimension(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip().removesuffix("px")
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_priority_assets() -> list[Path]:
    review_targets = ROOT / "scripts" / "visual_audit" / "review_targets.json"
    config = json.loads(review_targets.read_text(encoding="utf8"))
    assets: set[Path] = set()
    for pattern in config.get("priority_asset_globs", []):
        for match in glob(str(ROOT / pattern), recursive=True):
            path = Path(match)
            if path.is_file() and path.suffix.lower() in {".svg", ".png"}:
                assets.add(path)

    if assets:
        return sorted(assets)

    fallback = ROOT / "docs" / "assets" / "generated"
    return sorted(
        path for path in fallback.rglob("*") if path.suffix.lower() in {".svg", ".png"}
    )


def check_theme_pair(path: Path) -> list[str]:
    issues: list[str] = []
    if ".light." in path.name:
        dark = path.with_name(path.name.replace(".light.", ".dark."))
        if not dark.exists():
            issues.append(f"{path}: missing dark-theme pair")
    if ".dark." in path.name:
        light = path.with_name(path.name.replace(".dark.", ".light."))
        if not light.exists():
            issues.append(f"{path}: missing light-theme pair")
    return issues


def check_svg(path: Path) -> list[str]:
    issues: list[str] = []

    issues.extend(check_theme_pair(path))

    if path.stat().st_size < 1024:
        issues.append(
            f"{path}: suspiciously small SVG file ({path.stat().st_size} bytes)"
        )

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        return [f"{path}: XML parse error: {exc}"]

    root = tree.getroot()
    images = root.findall(".//svg:image", SVG_NS)

    width = parse_dimension(root.get("width"))
    height = parse_dimension(root.get("height"))
    view_box = root.get("viewBox")

    if (width is not None and width <= 0) or (height is not None and height <= 0):
        issues.append(f"{path}: non-positive SVG dimensions")
    if width is None and height is None and not view_box:
        issues.append(f"{path}: missing width/height and viewBox")

    graphic_count = sum(
        1 for element in root.iter() if local_name(element.tag) in GRAPHIC_TAGS
    )
    if graphic_count == 0:
        issues.append(f"{path}: no visible graphic elements found")

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

    if path.name.startswith(("reviewer_proof_panel", "benchmark_overview")):
        if not images:
            issues.append(f"{path}: expected embedded raster panels, found none")
        contain_images = [
            image for image in images if image.get("data-fit") == "contain"
        ]
        if len(contain_images) != len(images):
            issues.append(
                f'{path}: expected all embedded raster panels to declare data-fit="contain"'
            )
        for image in images:
            preserve = image.get("preserveAspectRatio", "")
            if "slice" in preserve:
                issues.append(
                    f"{path}: embedded raster panel still uses cropped preserveAspectRatio={preserve!r}"
                )

    return issues


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError("invalid PNG signature")
        handle.read(4)  # chunk length
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR":
            raise ValueError("missing IHDR chunk")
        width = int.from_bytes(handle.read(4), "big")
        height = int.from_bytes(handle.read(4), "big")
        return width, height


def check_png(path: Path) -> list[str]:
    issues = check_theme_pair(path)
    size_bytes = path.stat().st_size
    if size_bytes < 2048:
        issues.append(f"{path}: suspiciously small PNG file ({size_bytes} bytes)")

    try:
        width, height = read_png_size(path)
    except ValueError as exc:
        return [*issues, f"{path}: PNG parse error: {exc}"]

    if width < 64 or height < 64:
        issues.append(f"{path}: suspiciously small PNG dimensions ({width}x{height})")

    return issues


def main() -> int:
    asset_paths = load_priority_assets()
    if not asset_paths:
        print("No generated SVG or PNG assets found under docs/assets/generated")
        return 0

    all_issues: list[str] = []
    svg_count = 0
    png_count = 0
    for path in asset_paths:
        if path.suffix.lower() == ".svg":
            svg_count += 1
            all_issues.extend(check_svg(path))
        elif path.suffix.lower() == ".png":
            png_count += 1
            all_issues.extend(check_png(path))

    if all_issues:
        print("SVG audit issues found:")
        for issue in all_issues:
            print(f"- {issue}")
        return 1

    print(
        f"Checked {svg_count} SVG files and {png_count} PNG files: no asset-integrity issues found."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
