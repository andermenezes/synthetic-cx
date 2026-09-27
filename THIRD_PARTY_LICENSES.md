# Third-party dependency and notice review

Reviewed on 2026-09-27 for the `synthetic-cx` 0.1.0 **wheel and sdist**, against
`uv.lock` SHA-256 `c570d9a581709d7d91ab3d7cc1076e9559f73b53ddadca6c2a51390b892bd687`.
The archives contain SyntheticCX source, metadata, README, and the project
Apache-2.0 `LICENSE`. They do not bundle the separately installed dependency
distributions, fixture, or generated schemas. Each installed dependency keeps
its own distribution and license files. This is an archive/notice disposition;
it does not describe a combined image or vendored dependency release.

## Direct runtime dependencies

The executable `scripts/license_inventory.py` is the reproducible evidence for
this table: it verifies each direct dependency's installed version is present
in `uv.lock`, locates the installed license file, checks the expected license
label (using the TinyTroupe MIT license text when metadata is absent), and
prints its exact installed path and SHA-256. The normal Linux CI workflow runs
that command after locked synchronization. The TinyTroupe wheel omits a
metadata license label; its installed `licenses/LICENSE` is the MIT License,
copyright Microsoft Corporation. The `telemetry` extra is not installed by
default.

| Dependency | Locked version | Observed license | Source of license evidence |
| --- | --- | --- | --- |
| langgraph | 1.2.11 | MIT | `license_inventory.py` metadata/path/SHA-256 output |
| openai | 2.54.0 | Apache-2.0 | `license_inventory.py` metadata/path/SHA-256 output |
| opentelemetry-api | 1.44.0 | Apache-2.0 | `license_inventory.py` metadata/path/SHA-256 output |
| opentelemetry-sdk | 1.44.0 | Apache-2.0 | `license_inventory.py` metadata/path/SHA-256 output |
| playwright | 1.62.0 | Apache-2.0 | `license_inventory.py` metadata/path/SHA-256 output; separate wheel carries driver notice |
| pydantic | 2.13.5 | MIT | `license_inventory.py` metadata/path/SHA-256 output |
| pydantic-settings | 2.15.0 | MIT | `license_inventory.py` metadata/path/SHA-256 output |
| PyYAML | 6.0.3 | MIT | `license_inventory.py` metadata/path/SHA-256 output |
| TinyTroupe | 0.7.0, Git commit `a6244b358a1fe1c71bf751f7ba0f8dfa368ec5a4` | MIT | `license_inventory.py` installed Git-built wheel path/SHA-256 output |
| typer | 0.27.2 | MIT | `license_inventory.py` metadata/path/SHA-256 output |

## Transitive graph and notice disposition

The lockfile dependency graph has **275 core dependencies**, **8 packages
reachable only through the opt-in `telemetry` extra**, and **13 packages
reachable only through the dev group**, excluding the root package. These are
cross-platform lockfile graph counts, not an assertion that every package is
installed on every machine. The extra-only packages are
`googleapis-common-protos`, `grpcio`, `opentelemetry-exporter-otlp`,
`opentelemetry-exporter-otlp-proto-common`,
`opentelemetry-exporter-otlp-proto-grpc`,
`opentelemetry-exporter-otlp-proto-http`, `opentelemetry-proto`, and `protobuf`.
The dev-only packages are `ast-serialize`, `cfgv`, `distlib`, `identify`,
`librt`, `mypy`, `nodeenv`, `pathspec`, `pre-commit`, `python-discovery`,
`ruff`, `types-pyyaml`, and `virtualenv`.

The following core transitive packages warrant particular attention when
distributing anything beyond the standalone SyntheticCX archives:

| Dependency | Locked version | Reviewed evidence / disposition |
| --- | --- | --- |
| llama-index-readers-web | 0.7.0 | Declares GPL-3.0-or-later; required by TinyTroupe, with license included in its installed wheel. |
| html2text | 2024.2.26 | Declares GNU GPL 3, installed `COPYING`; required by llama-index-readers-web. |
| requests | 2.34.2 | Apache-2.0; installed wheel includes its `NOTICE`. |
| selenium | 4.49.0 | Apache-2.0; installed wheel includes `NOTICE` and Selenium Manager third-party notices. |
| playwright | 1.62.0 | Apache-2.0; bundled driver within its *own* wheel carries a package `NOTICE`. |
| certifi | 2026.7.22 | MPL-2.0; license file accompanies its separate wheel. |
| torch | 2.14.0 | Multi-license metadata, with individual bundled third-party licenses in its separate distribution. |
| transformers | 5.17.0 | Apache-2.0 license in its separate wheel. |

`scripts/verify_distribution.py` explicitly allows only `syntheticcx/` package
members plus the project wheel metadata, entry point, and project Apache
license; the sdist allowlist likewise permits only project metadata, README,
license, build metadata, and `src/syntheticcx/`. It therefore rejects a
vendored dependency, extra third-party license, or third-party `NOTICE` in the
current archive format. The project archives have no project-authored `NOTICE`
requiring reproduction, so **no root `NOTICE` is added at this checkpoint**.
Dependency notices above remain in their respective installed distributions.
If binaries or dependency wheels are vendored, redistributed together, or
repackaged into a container/image, review their exact license and notice files
for that delivery format first; the standalone archive allowlist does not
approve such a format.

## Further evidence to collect for a bundled release

The local macOS environment used for this review contained versions different
from the committed lock for some transitives (for example `numpy`, `scipy`,
and `contourpy`) and omitted platform-specific Linux/Windows packages.
Several transitive distributions have no concise license field in package
metadata; their included license files, where present, are more authoritative
than guessing an SPDX expression. A future bundled Linux release needs a
fresh inventory from the **actually resolved and installed** Linux environment,
including optional extras if shipped, and a dependency-specific GPL/MPL/notice
review. The scheduled/manual OpenAI smoke test is a separate credentialed
integration gate and supplies no license evidence.
