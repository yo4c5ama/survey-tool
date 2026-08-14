from threading import Event

from vnn_survey.app.task_manager import TaskManager


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
