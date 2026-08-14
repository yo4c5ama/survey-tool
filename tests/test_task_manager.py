from threading import Event
from time import monotonic, sleep

from vnn_survey.app.task_manager import TaskManager, raise_if_cancelled


def test_task_manager_keeps_one_background_task_per_project() -> None:
    manager = TaskManager(max_workers=1)
    release = Event()
    started = Event()

    def work() -> str:
        started.set()
        release.wait(timeout=5)
        return "done"

    try:
        assert manager.start("project", "discovery", work)
        assert started.wait(timeout=2)
        assert manager.is_running("project")
        assert not manager.start("project", "duplicate", work)

        snapshot = manager.snapshot("project")
        assert snapshot is not None
        assert snapshot.operation == "discovery"
        assert snapshot.running
    finally:
        release.set()
        manager.shutdown()

    snapshot = manager.snapshot("project")
    assert snapshot is not None
    assert not snapshot.running
    assert snapshot.error == ""


def test_task_manager_cancels_and_restarts_last_task() -> None:
    manager = TaskManager(max_workers=1)
    started = Event()
    calls = 0

    def work() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            while True:
                raise_if_cancelled()
                sleep(0.01)
        return "restarted"

    try:
        assert manager.start("project", "discovery", work)
        assert started.wait(timeout=2)
        assert manager.cancel("project")

        deadline = monotonic() + 2
        while manager.is_running("project") and monotonic() < deadline:
            sleep(0.01)

        cancelled = manager.snapshot("project")
        assert cancelled is not None
        assert not cancelled.running
        assert cancelled.cancelled
        assert cancelled.can_restart
        assert cancelled.error == ""

        assert manager.restart("project")
        deadline = monotonic() + 2
        while manager.is_running("project") and monotonic() < deadline:
            sleep(0.01)

        restarted = manager.snapshot("project")
        assert restarted is not None
        assert not restarted.running
        assert not restarted.cancelled
        assert restarted.error == ""
        assert calls == 2
    finally:
        manager.shutdown()
