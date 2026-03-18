from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_VISUAL_CONFIG_PATH = ROOT / "scripts" / "visual_audit" / "docs_visual_config.json"
DEFAULT_ARTIFACT_NAME = "docs-built-site"
DEFAULT_SITE_DIR = "site"
DEFAULT_SITE_ENTRYPOINT = "index.html"


@dataclass(frozen=True, slots=True)
class DocsSiteContract:
    artifact_name: str
    site_dir_name: str
    entrypoint_name: str

    @property
    def site_dir(self) -> Path:
        return ROOT / self.site_dir_name

    @property
    def site_entrypoint(self) -> str:
        return f"{self.site_dir_name}/{self.entrypoint_name}"

    @property
    def entrypoint_path(self) -> Path:
        return self.site_dir / self.entrypoint_name

    @property
    def root_entrypoint_path(self) -> Path:
        return ROOT / self.entrypoint_name


def load_docs_visual_config() -> dict[str, str]:
    return json.loads(DOCS_VISUAL_CONFIG_PATH.read_text(encoding="utf8"))


def load_docs_site_contract() -> DocsSiteContract:
    config = load_docs_visual_config()
    return DocsSiteContract(
        artifact_name=str(
            config.get("built_site_artifact_name", DEFAULT_ARTIFACT_NAME)
        ),
        site_dir_name=str(config.get("built_site_dir", DEFAULT_SITE_DIR)),
        entrypoint_name=str(
            config.get("built_site_entrypoint", DEFAULT_SITE_ENTRYPOINT)
        ),
    )


def contract_lines(contract: DocsSiteContract) -> list[str]:
    return [
        f"Built site artifact: {contract.artifact_name}",
        f"Expected site directory: {contract.site_dir_name} ({contract.site_dir})",
        f"Expected entrypoint: {contract.site_entrypoint} ({contract.entrypoint_path})",
    ]


def verify_prebuilt_site(contract: DocsSiteContract) -> None:
    problems: list[str] = []
    if not contract.site_dir.is_dir():
        problems.append(
            f"Missing expected site directory: {contract.site_dir_name} ({contract.site_dir})"
        )
    if not contract.entrypoint_path.is_file():
        problems.append(
            "Missing expected site entrypoint: "
            f"{contract.site_entrypoint} ({contract.entrypoint_path})"
        )

    if not problems:
        return

    lines = ["Docs built-site contract verification failed.", *contract_lines(contract)]
    lines.extend(f"- {problem}" for problem in problems)

    if not contract.site_dir.is_dir() and contract.root_entrypoint_path.is_file():
        lines.append(
            "Found the site entrypoint in the workspace root instead of under the "
            f"expected directory: {contract.root_entrypoint_path}"
        )
        lines.append(
            "This usually means the built-site artifact was extracted into the "
            "workspace root instead of the configured site directory."
        )

    if contract.site_dir.is_dir():
        sample_children = sorted(path.name for path in contract.site_dir.iterdir())[:12]
        if sample_children:
            lines.append(
                "Sample contents under the expected site directory: "
                + ", ".join(sample_children)
            )

    raise ValueError("\n".join(lines))


def write_github_outputs(contract: DocsSiteContract) -> None:
    output_lines = [
        f"artifact_name={contract.artifact_name}",
        f"site_dir={contract.site_dir_name}",
        f"site_entrypoint={contract.site_entrypoint}",
    ]
    output_path = os.environ.get("GITHUB_OUTPUT")

    if output_path:
        with Path(output_path).open("a", encoding="utf8") as handle:
            for line in output_lines:
                handle.write(f"{line}\n")
    else:
        for line in output_lines:
            print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Describe, export, and verify the shared docs built-site artifact contract."
        )
    )
    parser.add_argument(
        "command",
        choices=("describe", "github-outputs", "verify"),
        help="Action to perform.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_docs_site_contract()

    if args.command == "describe":
        print("\n".join(contract_lines(contract)), flush=True)
        return 0

    if args.command == "github-outputs":
        print("Loaded docs built-site contract:", flush=True)
        print("\n".join(contract_lines(contract)), flush=True)
        write_github_outputs(contract)
        return 0

    try:
        verify_prebuilt_site(contract)
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    print("Docs built-site contract verified.", flush=True)
    print("\n".join(contract_lines(contract)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
