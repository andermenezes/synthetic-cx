"""Bounded, Pydantic-validated native discovery data (never browser authority).

Typed dictionaries retain the existing JSON-shaped observation API while making
the broker/controller seam explicit. Native SDK extraction stays adapter-private.
"""

from collections.abc import Callable
from typing import Annotated, Literal, NotRequired, TypeVar

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, with_config
from typing_extensions import TypedDict

from syntheticcx.config.models import JsonObject, LocatorConfig
from syntheticcx.errors import ChannelError

OpaqueId = Annotated[str, Field(min_length=1, max_length=128)]
CandidateKind = Literal[
    "opener", "scope", "frame", "input", "send", "assistant", "customer", "option"
]
WebsiteAction = Literal[
    "observe", "stop", "wait", "open", "hover", "follow_link", "scroll_down", "scroll_up", "back"
]


@with_config(ConfigDict(extra="forbid", strict=True))
class Candidate(TypedDict):
    id: OpaqueId
    kind: CandidateKind
    description: Annotated[str, Field(max_length=4000)]
    evidence: NotRequired[Annotated[str, Field(max_length=80)]]
    context: NotRequired[Annotated[str, Field(max_length=2000)]]
    actions: NotRequired[Annotated[list[WebsiteAction], Field(max_length=9)]]
    blocked_by: NotRequired[Annotated[str, Field(max_length=1000)]]
    tried_actions: NotRequired[Annotated[list[WebsiteAction], Field(max_length=9)]]


@with_config(ConfigDict(extra="forbid", strict=True))
class Catalog(TypedDict):
    candidates: Annotated[list[Candidate], Field(max_length=160)]
    revision: NotRequired[OpaqueId]
    truncated: NotRequired[bool]
    has_more: NotRequired[bool]


@with_config(ConfigDict(extra="forbid", strict=True))
class ObservedOption(TypedDict):
    id: OpaqueId
    label: Annotated[str, Field(min_length=1, max_length=4000)]


@with_config(ConfigDict(extra="forbid", strict=True))
class ObservedState(TypedDict):
    id: OpaqueId
    kind: Literal["composer", "menu", "mixed", "terminal"]
    option_labels: Annotated[list[str], Field(max_length=160)]


@with_config(ConfigDict(extra="forbid", strict=True))
class LearningObservation(TypedDict):
    revision: OpaqueId
    catalog: Catalog
    available_options: Annotated[list[ObservedOption], Field(max_length=160)]
    ownership_bound: NotRequired[bool]
    opener_opened: NotRequired[bool]
    reader_bound: NotRequired[bool]
    interaction_ready: NotRequired[bool]
    state: NotRequired[ObservedState | None]
    composer_id: NotRequired[OpaqueId | None]


@with_config(ConfigDict(extra="forbid", strict=True))
class WebsitePage(TypedDict):
    origin: str
    state: str
    scroll_position: int
    back_depth: int


@with_config(ConfigDict(extra="forbid", strict=True))
class WebsiteHistory(TypedDict):
    action: WebsiteAction
    state: str
    description: Annotated[str, Field(max_length=4000)]
    result: Annotated[str, Field(max_length=80)]


@with_config(ConfigDict(extra="forbid", strict=True))
class WebsiteObservation(TypedDict):
    terminal: NotRequired[Literal["webchat", "whatsapp"] | None]
    revision: NotRequired[OpaqueId]
    catalog: NotRequired[Catalog]
    available_actions: NotRequired[Annotated[list[WebsiteAction], Field(max_length=9)]]
    observed_channel: NotRequired[Literal["webchat", "whatsapp"] | None]
    page_context: NotRequired[Annotated[str, Field(max_length=10000)]]
    page: NotRequired[WebsitePage]
    feedback: NotRequired[Annotated[str, Field(max_length=256)]]
    path_length: NotRequired[int]
    history: NotRequired[Annotated[list[WebsiteHistory], Field(max_length=128)]]
    catalog_page: NotRequired[int]


@with_config(ConfigDict(extra="forbid", strict=True))
class ReaderBinding(TypedDict):
    assistant_message_locator: JsonObject
    customer_message_locator: JsonObject | None


T = TypeVar("T")


def _validate(validate: Callable[[object], T], value: object) -> T:
    try:
        return validate(value)
    except ValidationError:
        raise ChannelError("channel.observe", "native discovery observation is invalid") from None


_LEARNING = TypeAdapter(LearningObservation)
_WEBSITE = TypeAdapter(WebsiteObservation)
_READER = TypeAdapter(ReaderBinding)


def learning_observation(value: object) -> LearningObservation:
    observation: LearningObservation = _validate(_LEARNING.validate_python, value)
    if observation["catalog"].get("revision", observation["revision"]) != observation["revision"]:
        raise ChannelError("channel.observe", "native catalog revision is inconsistent")
    return observation


def website_observation(value: object) -> WebsiteObservation:
    observation: WebsiteObservation = _validate(_WEBSITE.validate_python, value)
    if not observation.get("terminal") and not all(
        key in observation for key in ("revision", "catalog", "available_actions")
    ):
        raise ChannelError("channel.observe", "native website observation is incomplete")
    return observation


def reader_binding(value: object) -> ReaderBinding:
    binding: ReaderBinding = _validate(_READER.validate_python, value)
    try:
        LocatorConfig.model_validate(binding["assistant_message_locator"])
        if binding["customer_message_locator"] is not None:
            LocatorConfig.model_validate(binding["customer_message_locator"])
    except ValidationError:
        raise ChannelError("channel.reader", "native reader binding is invalid") from None
    return binding
