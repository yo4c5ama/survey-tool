from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from threading import Event, Lock
from typing import Any


class TaskCancelled(RuntimeError):
    """Raised by cooperative pipeline checkpoints after a user cancellation."""


_CANCEL_EVENT: ContextVar[Event | None] = ContextVar(
    "surveyflow_cancel_event",
    default=None,
)


def cancellation_requested() -> bool:
    event = _CANCEL_EVENT.get()
    return bool(event and event.is_set())


def raise_if_cancelled() -> None:
    if cancellation_requested():
        raise TaskCancelled("Run stopped by user.")


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    operation: str
    started_at: str
    running: bool
    error: str
    cancel_requested: bool
    cancelled: bool
    can_restart: bool


@dataclass(slots=True)
class _Task:
    operation: str
    started_at: str
    future: Future[Any]
    cancel_event: Event
    target: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


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
            cancel_event = Event()
            future = self._executor.submit(
                self._run_task,
                cancel_event,
                target,
                args,
                kwargs,
            )
            self._tasks[project_slug] = _Task(
                operation=operation,
                started_at=datetime.now().isoformat(timespec="seconds"),
                future=future,
                cancel_event=cancel_event,
                target=target,
                args=args,
                kwargs=dict(kwargs),
            )
            return True

    @staticmethod
    def _run_task(
        cancel_event: Event,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        token = _CANCEL_EVENT.set(cancel_event)
        try:
            raise_if_cancelled()
            return target(*args, **kwargs)
        finally:
            _CANCEL_EVENT.reset(token)

    def cancel(self, project_slug: str) -> bool:
        """Request cancellation of the active task at its next checkpoint."""

        with self._lock:
            task = self._tasks.get(project_slug)
            if task is None or task.future.done():
                return False
            task.cancel_event.set()
            task.future.cancel()
            return True

    def restart(self, project_slug: str) -> bool:
        """Repeat the last finished invocation with a fresh cancellation signal."""

        with self._lock:
            task = self._tasks.get(project_slug)
            if task is None or not task.future.done():
                return False
            operation = task.operation
            target = task.target
            args = task.args
            kwargs = dict(task.kwargs)
        return self.start(project_slug, operation, target, *args, **kwargs)

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
            cancel_requested = task.cancel_event.is_set()
            cancelled = cancel_requested and not running
            error = ""
            if not running and not task.future.cancelled():
                exception = task.future.exception()
                if exception and not isinstance(exception, TaskCancelled):
                    error = str(exception)
            return TaskSnapshot(
                operation=task.operation,
                started_at=task.started_at,
                running=running,
                error=error,
                cancel_requested=cancel_requested,
                cancelled=cancelled,
                can_restart=not running,
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
