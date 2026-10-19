"""Manual provisioning of dedicated channel browser sessions."""

import fcntl
import os
import re
from collections.abc import Callable
from pathlib import Path

from playwright.sync_api import sync_playwright

from syntheticcx.adapters.channel.whatsapp import validate_browser_profile
from syntheticcx.errors import ChannelError, ConfigurationError


def create_browser_profile(path: Path, *, confirm: Callable[[], None]) -> Path:
    """Open WhatsApp for manual authentication; never inspect or export credentials."""
    try:
        profile = path.expanduser().resolve()
        if profile.name == "Default" or re.fullmatch(r"Profile [0-9]+", profile.name):
            raise ConfigurationError("profile", "use a dedicated browser profile directory")
        ancestor = profile
        while not ancestor.exists():
            ancestor = ancestor.parent
        validate_browser_profile(ancestor)
        profile.mkdir(parents=True, exist_ok=True, mode=0o700)
        profile = validate_browser_profile(profile)
    except OSError:
        raise ConfigurationError(
            "profile", "cannot create the dedicated profile directory"
        ) from None

    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                profile / ".syntheticcx-whatsapp.lock",
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ChannelError(
                "profile", "dedicated browser profile is unavailable or already in use"
            ) from None
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(profile, headless=False)
            try:
                page = context.new_page()
                page.goto("https://web.whatsapp.com/", wait_until="domcontentloaded")
                confirm()
            finally:
                context.close()
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return profile
