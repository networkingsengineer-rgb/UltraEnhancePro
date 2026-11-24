"""Run a native tool as a child process with progress, pause and cancel."""
from __future__ import annotations

import re
import threading
from collections import deque
from typing import Callable, List, Optional

from . import engines
from .config import log
from .job_queue import Cancelled, JobControl

_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)%")


def run_tool(cmd: List[str], ctl: Optional[JobControl] = None,
             on_line: Optional[Callable[[str], None]] = None, name: str = "tool") -> List[str]:
    log.info("Running %s: %s", name, " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    proc = engines.popen(cmd)
    if ctl:
        ctl.attach_process(proc)
    tail: deque[str] = deque(maxlen=40)

    def reader():
        buf = ""
        assert proc.stdout is not None
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in "\r\n":
                if buf.strip():
                    tail.append(buf.strip())
                    if on_line:
                        try:
                            on_line(buf.strip())
                        except Exception:
                            pass
                buf = ""
            else:
                buf += ch
        if buf.strip():
            tail.append(buf.strip())

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        while proc.poll() is None:
            t.join(0.25)
            if ctl and ctl.cancelled:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise Cancelled()
        t.join(2)
    finally:
        if ctl:
            ctl.detach_process()
    if ctl and ctl.cancelled:
        raise Cancelled()
    if proc.returncode != 0:
        detail = "\n".join(list(tail)[-8:])
        log.error("%s failed (%s):\n%s", name, proc.returncode, "\n".join(tail))
        raise RuntimeError(f"{name} stopped with an error.\n{detail}")
    return list(tail)


def percent_from_line(line: str) -> Optional[float]:
    m = _PCT.search(line)
    if not m:
        return None
    v = float(m.group(1))
    return v / 100 if 0 <= v <= 100 else None
