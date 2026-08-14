from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    operation: str
    started_at: str
    running: bool
    error: str


@dataclass(slots=True)
class _Task:
    operation: str
    started_at: str
    future: Future[Any]


class TaskManager:
    """Runs one background pipeline task per project."""

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="surveyflow",
        )
        self._tasks: dict[str, _Task] = {}
        self._lock = Lock()

    def start(
        self,
        project_slug: str,
        operation: str,
        target: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        with self._lock:
            current = self._tasks.get(project_slug)
            if current and not current.future.done():
                return False
            future = self._executor.submit(target, *args, **kwargs)
            self._tasks[project_slug] = _Task(
                operation=operation,
                started_at=datetime.now().isoformat(timespec="seconds"),
                future=future,
            )
            return True

    def is_running(self, project_slug: str) -> bool:
        with self._lock:
            task = self._tasks.get(project_slug)
            return bool(task and not task.future.done())

    def snapshot(self, project_slug: str) -> TaskSnapshot | None:
        with self._lock:
            task = self._tasks.get(project_slug)
            if task is None:
                return None
            running = not task.future.done()
            error = ""
            if not running:
                exception = task.future.exception()
                error = str(exception) if exception else ""
            return TaskSnapshot(
                operation=task.operation,
                started_at=task.started_at,
                running=running,
                error=error,
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
