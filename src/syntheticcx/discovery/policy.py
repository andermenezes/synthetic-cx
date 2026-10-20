"""Hard exclusions for discovery, independent of model choices and runtime permissions."""

import re
import unicodedata
from urllib.parse import unquote

from syntheticcx.errors import ConfigurationError

_PORTABILITY = re.compile(
    r"\b(?:portabilidade|portability|number porting|"
    r"port(?:ing)? (?:(?:my|your|the|phone|mobile) )*number|"
    r"(?:portar|trazer|traga|transferir|transferencia|manter|mantenha) "
    r"(?:(?:o|a|de|do|meu|seu|mesmo) )*(?:numero|telefone)|"
    r"(?:bring|transfer|keep|retain) (?:(?:my|your|the|same|phone|mobile) )*number)\b"
)


def portability_flow(*evidence: str) -> bool:
    text = unquote(" ".join(evidence))
    text = "".join(
        c
        for c in unicodedata.normalize("NFKD", text.casefold())
        if unicodedata.category(c) not in {"Cf", "Mn"}
    )
    return bool(_PORTABILITY.search(" ".join(re.sub(r"[^\w]+", " ", text).split())))


def require_discovery_probe(probe: str | None) -> None:
    if probe is not None and portability_flow(probe):
        raise ConfigurationError(
            "discover probe", "number-portability flows are forbidden in discovery"
        )


def require_discovery_target(url: str) -> None:
    if portability_flow(url):
        raise ConfigurationError(
            "discover URL", "number-portability flows are forbidden in discovery"
        )
