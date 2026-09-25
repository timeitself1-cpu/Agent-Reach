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


BACKEND = "msvcrt" if os.name == "nt" else "fcntl"


def hold(path: Path, hold_s: float) -> int:
    """Take the lock, print 'held', keep it for ``hold_s`` seconds. Used by ``selftest``."""
    with GpuLock(path, wait_s=10.0) as got:
        if not got:
            print("busy", flush=True)
            return 1
        print("held", flush=True)
        time.sleep(hold_s)
    return 0


def selftest(path: Path, hold_s: float = 2.0) -> list[str]:
    """Cross-process lock checks. Returns failure messages; empty means the lock works.

    1. A lock held by another process blocks a non-waiting acquire.
    2. A waiting acquire gets the lock once the holder releases it.
    3. A lock held by a killed process is freed by the OS.
    """
    import subprocess
    import sys

    def spawn(seconds: float) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "-m", "paperclip_bridge", "lock-hold", str(path), str(seconds)],
            stdout=subprocess.PIPE, text=True,
        )

    failures: list[str] = []
    holder = spawn(hold_s)
    try:
        if (holder.stdout.readline() if holder.stdout else "").strip() != "held":
            return [f"helper process could not take the lock at {path}"]
        with GpuLock(path, wait_s=0.0) as got:
            if got:
                failures.append("lock was acquired while another process held it")
        start = time.monotonic()
        with GpuLock(path, wait_s=hold_s + 15.0) as got:
            if not got:
                failures.append("lock was not handed over after the holder released it")
            elif time.monotonic() - start < hold_s * 0.5:
                failures.append("waiting acquire returned before the holder released the lock")
    finally:
        holder.wait(timeout=hold_s + 30.0)

    crashed = spawn(600.0)
    try:
        if (crashed.stdout.readline() if crashed.stdout else "").strip() != "held":
            failures.append("second helper process could not take the lock")
            return failures
        crashed.kill()
        crashed.wait(timeout=30.0)
    finally:
        if crashed.poll() is None:
            crashed.kill()
    with GpuLock(path, wait_s=10.0) as got:
        if not got:
            failures.append("lock stayed held after the holding process was killed")
    return failures
