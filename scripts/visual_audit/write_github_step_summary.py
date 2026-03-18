from __future__ import annotations

import json
import os
from pathlib import Path


def item_classification(item: dict[str, str]) -> str:
    return item.get("classification") or item.get("failure_class") or "unknown"


def load_required_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    raise SystemExit(f"Missing required environment variable: {name}")


def main() -> int:
    summary_path = Path(load_required_env("GITHUB_STEP_SUMMARY"))
    summary_title = load_required_env("WORKFLOW_SUMMARY_TITLE")
    raw_items = load_required_env("WORKFLOW_SUMMARY_ITEMS_JSON")
    items = json.loads(raw_items)

    failed_items = [
        item for item in items if os.environ.get(item["env_key"], "") == "failure"
    ]
    warning_items = [
        item for item in items if os.environ.get(item["env_key"], "") == "warning"
    ]

    with summary_path.open("a", encoding="utf8") as handle:
        handle.write(f"## {summary_title}\n")
        if not failed_items and not warning_items:
            handle.write("Status: success\n")
            handle.write(
                "No classified stage failures or warnings were recorded in this job.\n"
            )
            return 0

        handle.write(f"Status: {'failure' if failed_items else 'warning'}\n")

        if failed_items:
            handle.write("### Failures\n")
            handle.write("| Step | Failure class | Likely layer | Next step |\n")
            handle.write("| --- | --- | --- | --- |\n")
            for item in failed_items:
                handle.write(
                    "| {label} | {classification} | {layer} | {next_step} |\n".format(
                        label=item["label"],
                        classification=item_classification(item),
                        layer=item["layer"],
                        next_step=item["next_step"],
                    )
                )
            handle.write("\n")

        if warning_items:
            handle.write("### Warnings\n")
            handle.write("| Step | Warning class | Likely layer | Next step |\n")
            handle.write("| --- | --- | --- | --- |\n")
            for item in warning_items:
                handle.write(
                    "| {label} | {classification} | {layer} | {next_step} |\n".format(
                        label=item["label"],
                        classification=item_classification(item),
                        layer=item["layer"],
                        next_step=item["next_step"],
                    )
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
