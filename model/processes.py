"""Worker processes that end with the process they work for."""

import ctypes
import os
import signal
import time


def end_with(parent):
    """Make the calling process's group end when `parent` is gone, however it went.

    The caller leads a session of its own. A parent that is killed runs no cleanup:
    left alone, a loader feeder would keep the trainer's GPU memory and the lock of
    its output, and a sampling process its engines."""
    leader = os.getpid()

    def stop(*_):
        if os.getpid() == leader:
            os.killpg(leader, signal.SIGKILL)
        # A worker inherits this handler and is terminated one at a time.
        os._exit(143)

    signal.signal(signal.SIGTERM, stop)
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG, not inherited by workers
    if os.getppid() != parent:
        stop()


def release_memory():
    """Return what the C heap has free to the system. A process that built and
    dropped something large otherwise keeps its pages for as long as it lives."""
    ctypes.CDLL(None).malloc_trim(0)


def end(processes, grace=5.0):
    """End group-leading processes and what they started: one that has not finished is
    asked to stop along with its process group, then killed if it does not."""
    for process in processes:
        if process.is_alive():
            _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            _signal_group(process.pid, signal.SIGKILL)
            process.join()
        else:
            # What a finished process started has exited with it; of a killed one it
            # may linger, and its group still names it.
            _signal_group(process.pid, signal.SIGKILL)


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
