from __future__ import annotations

import ast
import subprocess
from collections import deque
from functools import cache
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "option_pricing"
BENCHMARK_CONFTEST_PATH = "benchmarks/conftest.py"
BENCHMARK_BUILDER_PATH = "scripts/build_benchmark_artifacts.py"
AUTHORITATIVE_BENCHMARK_TEST_PATHS = (
    "benchmarks/test_bench_iv.py",
    "benchmarks/test_bench_localvol.py",
    "benchmarks/test_bench_macro.py",
    "benchmarks/test_bench_pde.py",
    "benchmarks/test_bench_tree.py",
)
BENCHMARK_SCOPE_DEFINITION_PATHS = ("scripts/benchmark_source_scope.py",)


def normalize_repo_path(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")


def _path_parts(module_name: str) -> tuple[str, ...]:
    return tuple(module_name.split("."))


def _module_name_from_repo_path(repo_path: str) -> str | None:
    parts = Path(repo_path).parts
    if len(parts) < 3 or parts[0] != "src" or parts[1] != PACKAGE_NAME:
        return None
    if parts[-1] == "__init__.py":
        return ".".join(parts[1:-1])
    if not parts[-1].endswith(".py"):
        return None
    return ".".join((*parts[1:-1], Path(parts[-1]).stem))


def _package_context_for_repo_path(repo_path: str) -> str | None:
    module_name = _module_name_from_repo_path(repo_path)
    if module_name is None:
        return None
    if Path(repo_path).name == "__init__.py":
        return module_name
    package_name, _, _ = module_name.rpartition(".")
    return package_name or module_name


def _path_for_module_file(root: Path, module_name: str) -> str:
    path = root.joinpath("src", *_path_parts(module_name)).with_suffix(".py")
    return normalize_repo_path(path.relative_to(root))


def _path_for_package_init(root: Path, module_name: str) -> str:
    path = root.joinpath("src", *_path_parts(module_name), "__init__.py")
    return normalize_repo_path(path.relative_to(root))


def _iter_parent_package_init_paths(
    *,
    root: Path,
    module_name: str,
    tracked_paths: frozenset[str],
) -> set[str]:
    paths: set[str] = set()
    parts = _path_parts(module_name)
    for end in range(1, len(parts) + 1):
        init_path = _path_for_package_init(root, ".".join(parts[:end]))
        if init_path in tracked_paths and (root / init_path).is_file():
            paths.add(init_path)
    return paths


def _resolve_module_reference(
    *,
    root: Path,
    module_name: str,
    tracked_paths: frozenset[str],
) -> set[str]:
    if not module_name.startswith(PACKAGE_NAME):
        return set()

    paths = _iter_parent_package_init_paths(
        root=root,
        module_name=module_name,
        tracked_paths=tracked_paths,
    )

    module_file = _path_for_module_file(root, module_name)
    if module_file in tracked_paths and (root / module_file).is_file():
        paths.add(module_file)
    return paths


def _resolve_import_from_module(
    *,
    importer_path: str,
    module: str | None,
    level: int,
) -> str | None:
    if level == 0:
        return module

    package_context = _package_context_for_repo_path(importer_path)
    if package_context is None:
        return None

    relative_name = "." * level + (module or "")
    try:
        return resolve_name(relative_name, package_context)
    except ImportError:
        return None


def _iter_repo_local_import_targets(
    *,
    root: Path,
    repo_path: str,
    tracked_paths: frozenset[str],
) -> set[str]:
    tree = ast.parse((root / repo_path).read_text(encoding="utf-8-sig"))
    discovered: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                discovered.update(
                    _resolve_module_reference(
                        root=root,
                        module_name=alias.name,
                        tracked_paths=tracked_paths,
                    )
                )
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        base_module = _resolve_import_from_module(
            importer_path=repo_path,
            module=node.module,
            level=node.level,
        )
        if base_module is None:
            continue

        discovered.update(
            _resolve_module_reference(
                root=root,
                module_name=base_module,
                tracked_paths=tracked_paths,
            )
        )

        package_init = _path_for_package_init(root, base_module)
        if package_init not in tracked_paths or not (root / package_init).is_file():
            continue

        for alias in node.names:
            if alias.name == "*":
                continue
            discovered.update(
                _resolve_module_reference(
                    root=root,
                    module_name=f"{base_module}.{alias.name}",
                    tracked_paths=tracked_paths,
                )
            )

    return discovered


@cache
def _git_tracked_paths(root_str: str) -> frozenset[str]:
    root = Path(root_str)
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace")
        raise RuntimeError(
            "Failed to list git-tracked files for benchmark source scope: "
            f"{detail.strip() or 'unknown git error'}"
        )
    return frozenset(
        normalize_repo_path(path)
        for path in result.stdout.decode("utf-8", errors="replace").split("\0")
        if path
    )


def _benchmark_test_seed_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        path for path in AUTHORITATIVE_BENCHMARK_TEST_PATHS if (root / path).is_file()
    )


def benchmark_source_seed_paths(root: Path = ROOT) -> tuple[str, ...]:
    return (
        BENCHMARK_CONFTEST_PATH,
        *_benchmark_test_seed_paths(root),
        BENCHMARK_BUILDER_PATH,
    )


def benchmark_scope_definition_paths(root: Path = ROOT) -> tuple[str, ...]:
    del root
    return BENCHMARK_SCOPE_DEFINITION_PATHS


def is_benchmark_seed_path(path: str | Path) -> bool:
    normalized = normalize_repo_path(path)
    if normalized in {BENCHMARK_CONFTEST_PATH, BENCHMARK_BUILDER_PATH}:
        return True
    return normalized.startswith("benchmarks/test_bench_") and normalized.endswith(
        ".py"
    )


@cache
def _benchmark_source_paths(root_str: str) -> tuple[str, ...]:
    root = Path(root_str)
    tracked_paths = _git_tracked_paths(root_str)
    queue = deque(
        path for path in benchmark_source_seed_paths(root) if path in tracked_paths
    )
    resolved: set[str] = set()

    while queue:
        repo_path = queue.popleft()
        if repo_path in resolved or repo_path not in tracked_paths:
            continue
        if not (root / repo_path).is_file():
            continue

        resolved.add(repo_path)
        for discovered in sorted(
            _iter_repo_local_import_targets(
                root=root,
                repo_path=repo_path,
                tracked_paths=tracked_paths,
            )
        ):
            if discovered not in resolved:
                queue.append(discovered)

    return tuple(sorted(resolved))


def benchmark_source_paths(root: Path = ROOT) -> tuple[str, ...]:
    return _benchmark_source_paths(str(root))


def iter_benchmark_source_files(root: Path = ROOT) -> tuple[Path, ...]:
    return tuple(root / repo_path for repo_path in benchmark_source_paths(root))


def is_benchmark_freshness_input(path: str | Path, root: Path = ROOT) -> bool:
    normalized = normalize_repo_path(path)
    return (
        normalized in benchmark_source_paths(root)
        or is_benchmark_seed_path(normalized)
        or normalized in benchmark_scope_definition_paths(root)
    )
