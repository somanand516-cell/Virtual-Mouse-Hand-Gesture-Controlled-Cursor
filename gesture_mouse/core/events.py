"""Hand-off points between threads.

Two very different kinds of data move between threads, and they need opposite policies:

* Frames and snapshots: only the newest one matters. If a consumer is slow, older items are
  worthless (processing them would only add lag), so `LatestSlot` keeps exactly one item and
  overwrites it. This is the "queue that always drops old frames".

* Discrete actions (button down/up, key presses): every single one matters. Dropping a
  "button up" would leave the mouse stuck down, so these go through an unbounded FIFO queue
  (`queue.SimpleQueue`) owned by the output controller.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class SlotClosed(Exception):
    """Raised by LatestSlot.get() after close(): the producer is gone."""


class LatestSlot(Generic[T]):
    """A single-item, overwrite-on-put hand-off. `get` returns the newest item not yet seen."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._seq = 0
        self._read_seq = 0
        self._closed = False
        self.dropped = 0  # items overwritten before anyone read them (for metrics)

    def put(self, item: T) -> None:
        with self._cond:
            if self._seq != self._read_seq:
                self.dropped += 1
            self._item = item
            self._seq += 1
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> T | None:
        """Wait for an item newer than the last one returned. None on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._read_seq or self._closed, timeout):
                return None
            if self._seq == self._read_seq:  # closed with nothing new
                raise SlotClosed
            self._read_seq = self._seq
            return self._item

    def peek(self) -> T | None:
        """The newest item, without waiting or marking it read."""
        with self._cond:
            return self._item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
