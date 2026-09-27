"""Inspect the built archives and smoke-test two independent installed wheels."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

MAX_SDIST_MEMBERS = 10_000
MAX_SDIST_MEMBER_BYTES = 50 * 1024 * 1024
MAX_SDIST_TOTAL_BYTES = 100 * 1024 * 1024


def one(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def check_members(names: list[str]) -> set[str]:
    paths = {name.rstrip("/") for name in names}
    for name in paths:
        parts = PurePosixPath(name).parts
        if (
            not parts
            or name.startswith("/")
            or any(
                part in {".", "..", ".venv", ".runs", "__pycache__", ".git", "docs"} or "\\" in part
                for part in parts
            )
            or any(part.startswith(".env") for part in parts)
            or name.endswith((".log", ".pyc", ".pypirc"))
        ):
            raise ValueError(f"unexpected distribution member: {name}")
    return paths


def check_wheel(path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        members = check_members(archive.namelist())
        info = one_member(members, ".dist-info/METADATA")
        prefix = info.removesuffix("METADATA")
        allowed = {
            "syntheticcx",
            prefix.rstrip("/"),
            prefix + "WHEEL",
            prefix + "RECORD",
            prefix + "entry_points.txt",
            prefix + "METADATA",
            prefix + "licenses",
            prefix + "licenses/LICENSE",
        }
        unexpected = {
            name for name in members if not name.startswith("syntheticcx/") and name not in allowed
        }
        if unexpected:
            raise ValueError(f"wheel includes unapproved non-project members: {sorted(unexpected)}")
        required = {
            "syntheticcx/__init__.py",
            "syntheticcx/__main__.py",
            "syntheticcx/cli.py",
            prefix + "WHEEL",
            prefix + "RECORD",
            prefix + "entry_points.txt",
            prefix + "licenses/LICENSE",
        }
        if missing := required - members:
            raise ValueError(f"wheel is missing {sorted(missing)}")
        metadata = Parser().parsestr(archive.read(info).decode("utf-8"))
        if metadata["Name"] != "synthetic-cx" or metadata["License-Expression"] != "Apache-2.0":
            raise ValueError("wheel project metadata or license is missing")
        license_content = archive.read(prefix + "licenses/LICENSE")
        if "# synthetic-cx" not in metadata.get_payload() or not license_content.strip():
            raise ValueError("wheel README description or license content is missing")
        version = metadata["Version"]
        if not version:
            raise ValueError("wheel version is missing")
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(prefix + "entry_points.txt").decode("utf-8"))
        if (
            entry_points.get("console_scripts", "synthetic-cx", fallback="")
            != "syntheticcx.cli:app"
        ):
            raise ValueError("wheel CLI entry point is missing")
        return version, license_content


def one_member(members: set[str], suffix: str) -> str:
    matches = [name for name in members if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one archive member ending in {suffix}, found {len(matches)}")
    return matches[0]


def package_contents(wheel: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("syntheticcx/") and not name.endswith("/")
        }


def validate_sdist_members(members: list[tarfile.TarInfo]) -> str:
    if len(members) > MAX_SDIST_MEMBERS:
        raise ValueError(f"sdist has more than {MAX_SDIST_MEMBERS} members")
    names = check_members([item.name for item in members])
    roots = {PurePosixPath(name).parts[0] for name in names}
    if len(roots) != 1:
        raise ValueError(f"sdist must have one root, found {sorted(roots)}")
    root = roots.pop()
    allowed = {
        root,
        f"{root}/PKG-INFO",
        f"{root}/pyproject.toml",
        f"{root}/pyproject.toml.orig",
        f"{root}/LICENSE",
        f"{root}/README.md",
        f"{root}/src",
        f"{root}/src/syntheticcx",
    }
    unexpected = {
        name
        for name in names
        if not name.startswith(f"{root}/src/syntheticcx/") and name not in allowed
    }
    if unexpected:
        raise ValueError(f"sdist includes unapproved non-project members: {sorted(unexpected)}")
    required = {
        f"{root}/PKG-INFO",
        f"{root}/pyproject.toml",
        f"{root}/README.md",
        f"{root}/LICENSE",
        f"{root}/src/syntheticcx/__init__.py",
        f"{root}/src/syntheticcx/__main__.py",
        f"{root}/src/syntheticcx/cli.py",
    }
    if missing := required - names:
        raise ValueError(f"sdist is missing {sorted(missing)}")
    if any(item.issym() or item.islnk() or not (item.isfile() or item.isdir()) for item in members):
        raise ValueError("sdist contains a link or unsupported member type")
    files = [item for item in members if item.isfile()]
    if any(item.size > MAX_SDIST_MEMBER_BYTES for item in files):
        raise ValueError(f"sdist member exceeds {MAX_SDIST_MEMBER_BYTES} byte limit")
    if sum(item.size for item in files) > MAX_SDIST_TOTAL_BYTES:
        raise ValueError(f"sdist exceeds {MAX_SDIST_TOTAL_BYTES} byte extraction limit")
    return root


def check_sdist(path: Path, version: str, expected_license: bytes) -> str:
    with tarfile.open(path, "r:gz") as archive:
        root = validate_sdist_members(archive.getmembers())
        info = archive.extractfile(f"{root}/PKG-INFO")
        license_file = archive.extractfile(f"{root}/LICENSE")
        if info is None or license_file is None:
            raise ValueError("sdist metadata or license cannot be read")
        with info, license_file:
            metadata = Parser().parsestr(info.read().decode("utf-8"))
            if (
                metadata["Name"] != "synthetic-cx"
                or metadata["Version"] != version
                or metadata["License-Expression"] != "Apache-2.0"
                or "# synthetic-cx" not in metadata.get_payload()
                or license_file.read() != expected_license
            ):
                raise ValueError(
                    "sdist project metadata, README description, or license is invalid"
                )
        return root


def unpack_sdist(path: Path, destination: Path) -> None:
    """Extract only validated regular files into the isolated build directory."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        validate_sdist_members(members)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"sdist member cannot be read: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def run(*args: str, cwd: Path, environment: dict[str, str]) -> None:
    subprocess.run(args, cwd=cwd, env=environment, check=True)


def check_install(
    wheel: Path, destination: Path, uv: str, environment: dict[str, str], version: str
) -> None:
    environment_dir = destination / "env"
    run(
        uv,
        "venv",
        "--python",
        sys.executable,
        str(environment_dir),
        cwd=destination,
        environment=environment,
    )
    python = environment_dir / "bin" / "python"
    command = environment_dir / "bin" / "synthetic-cx"
    run(
        uv,
        "pip",
        "install",
        "--quiet",
        "--python",
        str(python),
        str(wheel),
        cwd=destination,
        environment=environment,
    )
    for invocation in ([str(command), "--help"], [str(python), "-m", "syntheticcx", "--help"]):
        help_text = subprocess.check_output(invocation, cwd=destination, env=environment, text=True)
        if "run" not in help_text:
            raise ValueError(f"installed CLI does not advertise run: {invocation[0]}")
    reported_version = subprocess.check_output(
        [str(command), "--version"], cwd=destination, env=environment, text=True
    ).strip()
    if reported_version != version:
        raise ValueError(f"installed CLI version {reported_version!r} differs from {version!r}")
    import_version = subprocess.check_output(
        [str(python), "-c", "import syntheticcx; print(syntheticcx.__version__)"],
        cwd=destination,
        env=environment,
        text=True,
    ).strip()
    if import_version != version:
        raise ValueError(f"installed package version {import_version!r} differs from {version!r}")
    location = subprocess.check_output(
        [str(python), "-c", "import syntheticcx; print(syntheticcx.__file__)"],
        cwd=destination,
        env=environment,
        text=True,
    ).strip()
    if not Path(location).resolve().is_relative_to(environment_dir.resolve()):
        raise ValueError(f"package imported from outside the clean environment: {location}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, help="directory containing exactly one wheel and sdist")
    arguments = parser.parse_args()
    dist = arguments.dist.resolve()
    wheel = one(dist, "*.whl")
    sdist = one(dist, "*.tar.gz")
    version, license_content = check_wheel(wheel)
    root = check_sdist(sdist, version, license_content)
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        environment.pop(key, None)
    uv = os.environ.get("UV", "uv")
    with tempfile.TemporaryDirectory(prefix="syntheticcx-distribution-") as temporary:
        scratch = Path(temporary).resolve()
        wheel_check = scratch / "wheel"
        sdist_check = scratch / "sdist"
        wheel_check.mkdir()
        sdist_check.mkdir()
        check_install(wheel, wheel_check, uv, environment, version)
        unpack_sdist(sdist, sdist_check)
        rebuilt_dist = scratch / "rebuilt"
        run(
            uv,
            "build",
            "--wheel",
            "--out-dir",
            str(rebuilt_dist),
            str(sdist_check / root),
            cwd=sdist_check,
            environment=environment,
        )
        rebuilt_wheel = one(rebuilt_dist, "*.whl")
        rebuilt_version, _ = check_wheel(rebuilt_wheel)
        if rebuilt_version != version:
            raise ValueError("sdist rebuilt to a different project version")
        if package_contents(rebuilt_wheel) != package_contents(wheel):
            raise ValueError("sdist wheel contains different package source files")
        check_install(rebuilt_wheel, sdist_check, uv, environment, version)
    for artifact in (wheel, sdist):
        print(f"sha256 {artifact.name} {hashlib.sha256(artifact.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
