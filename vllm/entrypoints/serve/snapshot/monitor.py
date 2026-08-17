# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading


class SnapshotMonitor:
    """Thread-safe snapshot lifecycle completion flags."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._suspend_done = False
        self._unlock_done = False
        self._resume_done = False

    @property
    def is_suspend_done(self) -> bool:
        with self._lock:
            return self._suspend_done

    @property
    def is_unlock_done(self) -> bool:
        with self._lock:
            return self._unlock_done

    @property
    def is_resume_done(self) -> bool:
        with self._lock:
            return self._resume_done

    def mark_suspend_done(self) -> None:
        with self._lock:
            self._suspend_done = True

    def mark_unlock_done(self) -> None:
        with self._lock:
            self._unlock_done = True

    def mark_resume_done(self) -> None:
        with self._lock:
            self._resume_done = True
