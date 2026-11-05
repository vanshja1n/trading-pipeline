"""Tests for the single-instance guard.

Motivated by a live incident: two bots ran against one paper account, one
bought 128 SPY and the other sold it 3 minutes later.
"""

from __future__ import annotations

import os

import pytest

from autotrade.engine.live import AlreadyRunning, InstanceLock


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "live.lock"


def test_acquire_writes_our_pid(lock_path):
    with InstanceLock(lock_path) as lock:
        assert lock_path.read_text() == str(os.getpid())
        assert lock.acquired
    assert not lock_path.exists()  # released on exit


# PID 1 is launchd/init: always alive, never this test process. Signalling it
# raises PermissionError, which the lock correctly reads as "held by someone else".
OTHER_LIVE_PID = "1"


def test_second_instance_is_refused(lock_path):
    lock_path.write_text(OTHER_LIVE_PID)
    with pytest.raises(AlreadyRunning, match="already running"):
        InstanceLock(lock_path).acquire()


def test_error_message_tells_you_how_to_fix_it(lock_path):
    lock_path.write_text(OTHER_LIVE_PID)
    with pytest.raises(AlreadyRunning) as exc:
        InstanceLock(lock_path).acquire()
    message = str(exc.value)
    assert f"kill {OTHER_LIVE_PID}" in message
    assert str(lock_path) in message


def test_refusal_does_not_overwrite_the_holders_lock(lock_path):
    lock_path.write_text(OTHER_LIVE_PID)
    with pytest.raises(AlreadyRunning):
        InstanceLock(lock_path).acquire()
    assert lock_path.read_text() == OTHER_LIVE_PID


def test_stale_lock_from_a_dead_process_is_reclaimed(lock_path):
    # PID 999999 is above the default max on macOS/Linux, so it cannot be live.
    lock_path.write_text("999999")
    with InstanceLock(lock_path):
        assert lock_path.read_text() == str(os.getpid())


def test_garbage_lockfile_is_reclaimed(lock_path):
    lock_path.write_text("not a pid")
    with InstanceLock(lock_path):
        assert lock_path.read_text() == str(os.getpid())


def test_release_leaves_another_processes_lock_alone(lock_path):
    """A crashed instance must never delete the live instance's lock."""
    lock = InstanceLock(lock_path)
    lock.acquire()
    lock_path.write_text("999998")  # someone else took over
    lock.release()
    assert lock_path.exists()


def test_reacquire_by_same_process_is_allowed(lock_path):
    lock = InstanceLock(lock_path)
    lock.acquire()
    lock.acquire()  # idempotent, not an error
    assert lock_path.read_text() == str(os.getpid())
    lock.release()


def test_lock_released_even_if_body_raises(lock_path):
    with pytest.raises(ValueError):
        with InstanceLock(lock_path):
            raise ValueError("boom")
    assert not lock_path.exists()
