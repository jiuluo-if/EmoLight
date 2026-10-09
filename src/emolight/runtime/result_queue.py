from queue import Empty, Full, Queue
import threading
from typing import Generic, TypeVar


T = TypeVar("T")


class LatestOnlyQueue(Generic[T]):
    """Thread-safe one-slot queue that replaces stale results with the newest."""

    def __init__(self) -> None:
        self._queue: Queue[T] = Queue(maxsize=1)
        self._replaced_count = 0
        self._counter_lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return 1

    @property
    def replaced_count(self) -> int:
        with self._counter_lock:
            return self._replaced_count

    def publish(self, value: T) -> None:
        while True:
            try:
                self._queue.put_nowait(value)
                return
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    continue
                with self._counter_lock:
                    self._replaced_count += 1

    def take_latest(self) -> T | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None
