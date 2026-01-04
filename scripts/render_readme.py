from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_block(path: Path, start_marker: str, end_marker: str) -> str:
    text = path.read_text(encoding="utf-8")

    start = text.find(start_marker)
    if start == -1:
        raise ValueError(f"Could not find start marker in {path}: {start_marker}")

    start_content = start + len(start_marker)
    end = text.find(end_marker, start_content)
    if end == -1 or end <= start_content:
        raise ValueError(f"Could not find end marker in {path}: {end_marker}")

    block = text[start_content:end]
    block = block.strip("\n")
    block = textwrap.dedent(block).strip("\n")
    return block


def render(template_path: Path, out_path: Path, *, check: bool) -> int:
    tpl = template_path.read_text(encoding="utf-8")

    quick = extract_block(
        ROOT / "examples" / "quickstart.py",
        start_marker="# [START README_QUICKSTART]",
        end_marker="# [END README_QUICKSTART]",
    )

    curves_first = extract_block(
        ROOT / "examples" / "curves_first.py",
        start_marker="# [START README_CURVES_FIRST]",
        end_marker="# [END README_CURVES_FIRST]",
    )

    iv = extract_block(
        ROOT / "examples" / "implied_vol.py",
        start_marker="# [START README_IMPLIED_VOL]",
        end_marker="# [END README_IMPLIED_VOL]",
    )

    rendered = (
        tpl.replace("{{ QUICKSTART }}", quick)
        .replace("{{ CURVES_FIRST }}", curves_first)
        .replace("{{ IMPLIED_VOL }}", iv)
    )

    if check:
        current = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        if current != rendered:
            sys.stderr.write(
                "README.md is out of date. Run: python scripts/render_readme.py\n"
            )
            return 1
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Render README.md from template + examples snippets."
    )
    p.add_argument(
        "--check", action="store_true", help="Fail if README.md is not up to date."
    )
    args = p.parse_args()

    return render(
        template_path=ROOT / "README.template.md",
        out_path=ROOT / "README.md",
        check=args.check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
