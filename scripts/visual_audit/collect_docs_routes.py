from __future__ import annotations

from pathlib import Path


def main() -> None:
    docs_dir = Path("docs")
    pages: list[str] = []

    for path in sorted(docs_dir.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        if path.name.lower() == "visual-audit.md":
            continue

        rel = path.relative_to(docs_dir)
        if rel.name == "index.md":
            route = "/" if rel.parent == Path(".") else f"/{rel.parent.as_posix()}/"
        else:
            route = f"/{rel.with_suffix('').as_posix()}/"

        pages.append(route)

    print("Discovered docs routes:")
    for route in pages:
        print(route)


if __name__ == "__main__":
    main()
