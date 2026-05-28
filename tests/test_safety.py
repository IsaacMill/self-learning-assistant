from pathlib import Path

import pytest

from assistant.safety import (
    assert_not_self_modifying,
    contains_secret,
    contains_sensitive_personal_info,
    is_blocked_action_request,
)


def test_secret_detection() -> None:
    assert contains_secret("password=hunter2")


def test_sensitive_personal_info_detection() -> None:
    assert contains_sensitive_personal_info("My SSN is 123-45-6789")


def test_blocked_action_detection() -> None:
    assert is_blocked_action_request("Please run shell command ls")
    assert is_blocked_action_request("Delete this file")


def test_self_modifying_guard_blocks_assistant_source() -> None:
    with pytest.raises(PermissionError):
        assert_not_self_modifying(Path("assistant/main.py"))
