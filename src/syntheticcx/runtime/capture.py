"""Optional owned failure-image capabilities and explicit capture permission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from syntheticcx.config.models import ProjectConfig
from syntheticcx.domain.models import ArtifactRecord
from syntheticcx.errors import ConfigurationError


@dataclass(frozen=True)
class ScreenshotPolicy:
    """A validated project's independent capture, storage, and target affirmations."""

    allow_capture: bool = False
    store_artifact: bool = False
    target_safe: bool = False

    def __post_init__(self) -> None:
        if self.store_artifact and not (self.allow_capture and self.target_safe):
            raise ConfigurationError(
                "screenshots", "storage requires capture permission and a safe-target affirmation"
            )

    @classmethod
    def from_config(cls, config: ProjectConfig) -> ScreenshotPolicy:
        return cls(
            allow_capture=config.privacy.allow_screenshots,
            store_artifact=config.storage.store_screenshots,
            target_safe=config.privacy.confirm_screenshot_target_safe,
        )

    @property
    def enabled(self) -> bool:
        return self.allow_capture and self.store_artifact and self.target_safe


@runtime_checkable
class FailureScreenshotSource(Protocol):
    async def capture_failure_screenshot(self) -> bytes: ...


@runtime_checkable
class ScreenshotArtifactWriter(Protocol):
    async def write_screenshot(
        self, run_id: str, relative_path: str, content: bytes
    ) -> ArtifactRecord: ...
