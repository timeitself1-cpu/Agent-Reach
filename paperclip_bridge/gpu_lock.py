"""Machine-wide GPU lock shared by every local-model worker.

An OS-level lock on a file: released automatically if the holder crashes,
works on Windows (``msvcrt``) and POSIX (``fcntl``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import IO

if os.name == "nt":
    import msvcrt

    def _try_lock(fh: IO[bytes]) -> bool:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fh: IO[bytes]) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(fh: IO[bytes]) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fh: IO[bytes]) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class GpuLock:
    """``with GpuLock(path, wait_s) as got:`` -> ``got`` is False if the wait ran out."""

    def __init__(self, path: Path, wait_s: float, poll_s: float = 0.5):
        self.path = Path(path)
        self.wait_s = wait_s
        self.poll_s = poll_s
        self._fh: IO[bytes] | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        deadline = time.monotonic() + max(0.0, self.wait_s)
        while True:
            if _try_lock(fh):
                self._fh = fh
                return True
            if time.monotonic() >= deadline:
                fh.close()
                return False
            time.sleep(self.poll_s)

    def release(self) -> None:
        if self._fh is not None:
            try:
                _unlock(self._fh)
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
