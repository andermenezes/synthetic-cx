"""Emit reproducible license evidence for the direct runtime dependencies."""

from __future__ import annotations

import hashlib
import json
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

DIRECT_LICENSES = {
    "langgraph": "MIT",
    "openai": "Apache-2.0",
    "opentelemetry-api": "Apache-2.0",
    "opentelemetry-sdk": "Apache-2.0",
    "playwright": "Apache-2.0",
    "pydantic": "MIT",
    "pydantic-settings": "MIT",
    "pyyaml": "MIT",
    "tinytroupe": "MIT",
    "typer": "MIT",
}


def lock_versions() -> dict[str, set[str]]:
    lock = Path("uv.lock")
    document = tomllib.loads(lock.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for package in document["package"]:
        result.setdefault(package["name"], set()).add(package["version"])
    return result


def declared_dependencies() -> set[str]:
    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return {
        canonicalize_name(Requirement(dependency).name)
        for dependency in document["project"]["dependencies"]
    }


def license_file(distribution: metadata.Distribution) -> Path:
    candidates = [
        Path(str(distribution.locate_file(path)))
        for path in distribution.files or []
        if "license" in str(path).lower() or "copying" in str(path).lower()
    ]
    if not candidates:
        raise ValueError(f"{distribution.metadata['Name']} has no installed license file")
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def observed_license(distribution: metadata.Distribution, path: Path) -> str:
    declared = str(
        distribution.metadata["License-Expression"] or distribution.metadata["License"] or ""
    )
    if declared:
        return declared
    content = path.read_text(encoding="utf-8", errors="replace")
    if "MIT License" in content:
        return "MIT"
    raise ValueError(f"{distribution.metadata['Name']} has no recognizable license declaration")


def main() -> None:
    locked = lock_versions()
    declared = declared_dependencies()
    reviewed = {canonicalize_name(name) for name in DIRECT_LICENSES}
    if declared != reviewed:
        missing = sorted(declared - reviewed)
        unexpected = sorted(reviewed - declared)
        raise ValueError(
            "license inventory does not match project dependencies: "
            f"missing={missing}, unexpected={unexpected}"
        )
    records: list[dict[str, str]] = []
    for name, expected_license in DIRECT_LICENSES.items():
        distribution = metadata.distribution(name)
        if distribution.version not in locked.get(name, set()):
            raise ValueError(f"{name} {distribution.version} is not in uv.lock")
        path = license_file(distribution)
        license_value = observed_license(distribution, path)
        if license_value != expected_license:
            raise ValueError(f"{name} license {license_value!r} differs from {expected_license!r}")
        records.append(
            {
                "name": name,
                "version": distribution.version,
                "license": license_value,
                "license_path": str(path),
                "license_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
