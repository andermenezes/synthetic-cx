import fcntl
import os
from pathlib import Path

import pytest

from syntheticcx.application.profile import create_browser_profile
from syntheticcx.errors import ChannelError, ConfigurationError


@pytest.mark.parametrize("name", ["Default", "Profile 1"])
def test_personal_subprofile_is_rejected_before_creation(tmp_path: Path, name: str) -> None:
    profile = tmp_path / name
    with pytest.raises(ConfigurationError):
        create_browser_profile(profile, confirm=lambda: pytest.fail("browser opened"))
    assert not profile.exists()


def test_active_profile_cannot_be_provisioned(tmp_path: Path) -> None:
    lock = tmp_path / ".syntheticcx-whatsapp.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ChannelError):
            create_browser_profile(tmp_path, confirm=lambda: pytest.fail("browser opened"))
        # The failed invocation must not release the existing owner's lock.
        contender = os.open(lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    finally:
        os.close(descriptor)
