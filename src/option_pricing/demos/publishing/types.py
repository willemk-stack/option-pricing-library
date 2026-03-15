from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

StoryName = Literal["static", "dupire", "numerics"]

MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    name: str
    relative_path: str
    story: StoryName
    rectangular_grid: bool
    axis_columns: dict[str, str]
    default_scalar_fields: tuple[str, ...]
    source_object_type: str
    aliases: tuple[str, ...] = ()
    description: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relative_path": self.relative_path,
            "story": self.story,
            "rectangular_grid": self.rectangular_grid,
            "axis_columns": dict(self.axis_columns),
            "default_scalar_fields": list(self.default_scalar_fields),
            "source_object_type": self.source_object_type,
            "aliases": list(self.aliases),
            "description": self.description,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DatasetManifest:
        return cls(
            name=str(payload["name"]),
            relative_path=str(payload["relative_path"]),
            story=payload["story"],
            rectangular_grid=bool(payload["rectangular_grid"]),
            axis_columns=dict(payload.get("axis_columns", {})),
            default_scalar_fields=tuple(payload.get("default_scalar_fields", [])),
            source_object_type=str(payload["source_object_type"]),
            aliases=tuple(payload.get("aliases", [])),
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class BundleManifest:
    version: int
    profile: str
    workflow_profile: str
    seed: int
    bundle_root: Path
    datasets: tuple[DatasetManifest, ...]

    def manifest_path(self) -> Path:
        return self.bundle_root / "meta" / "manifest.json"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "profile": self.profile,
            "workflow_profile": self.workflow_profile,
            "seed": int(self.seed),
            "bundle_root": str(self.bundle_root),
            "datasets": [dataset.to_json_dict() for dataset in self.datasets],
        }

    def write(self) -> Path:
        path = self.manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        return path

    def resolve_dataset(self, query: str) -> DatasetManifest:
        token = str(query).strip()
        if not token:
            raise ValueError("dataset query must be non-empty")

        exact: list[DatasetManifest] = []
        for dataset in self.datasets:
            names = {dataset.name, *dataset.aliases}
            if token in names:
                exact.append(dataset)
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ValueError(f"Dataset query {token!r} is ambiguous.")

        normalized = token.removesuffix(".csv").replace("\\", "/")
        matches = [
            dataset
            for dataset in self.datasets
            if normalized in {dataset.name, *dataset.aliases}
        ]
        if len(matches) == 1:
            return matches[0]

        basename = Path(normalized).name
        aliases = [
            dataset
            for dataset in self.datasets
            if basename in {dataset.name, *dataset.aliases}
        ]
        if len(aliases) == 1:
            return aliases[0]
        if len(aliases) > 1:
            raise ValueError(
                f"Dataset basename {basename!r} is ambiguous. Use a qualified name."
            )
        raise KeyError(f"Unknown dataset: {query}")

    @classmethod
    def load(cls, manifest_path: str | Path) -> BundleManifest:
        path = Path(manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=int(payload["version"]),
            profile=str(payload["profile"]),
            workflow_profile=str(payload["workflow_profile"]),
            seed=int(payload["seed"]),
            bundle_root=Path(payload["bundle_root"]),
            datasets=tuple(
                DatasetManifest.from_json_dict(item)
                for item in payload.get("datasets", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class PlotSpec:
    preset: str
    filename: str
    renderer: str
    datasets: tuple[str, ...]
    title: str
    kwargs: dict[str, Any]
