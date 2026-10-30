"""Small ctypes bridge. Compile explicitly with python -m Deeplearning.model."""
import ctypes as ct
from pathlib import Path
import subprocess
import os
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "build" / "libbirdsong.so"
DT = 1e-7
CONTROL_NAMES = ("epsilon_l", "epsilon_r", "ps", "f0_l", "f0_r")
DEFAULT_CONTROLS = np.array([3.5e7, 3.5e7, 3e6, 4.9e4, 4.9e4])
LOW = np.array([2.5e7, 2.5e7, 2.5e6, 4.9e4, 4.9e4])
HIGH = np.array([5e7, 5e7, 4e6, 1e7, 1e7])


def build():
    LIBRARY.parent.mkdir(exist_ok=True)
    # Replace atomically so rebuilding never truncates a library mapped by training.
    with tempfile.TemporaryDirectory(dir=LIBRARY.parent) as directory:
        temporary = Path(directory)/"libbirdsong.so"
        subprocess.run(["g++", "-O3", "-std=c++11", "-fPIC", "-shared",
                        str(ROOT/"native/model.cpp"), "-o", str(temporary)], check=True)
        os.replace(temporary, LIBRARY)


def vector(x, size):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.shape != (size,) or not np.isfinite(x).all():
        raise ValueError(f"Expected {size} finite values")
    return x


def effective_controls(controls, condition):
    """Resolve constraints once for application, saving, plots and replay."""
    if condition not in ("A", "B", "C"):
        raise ValueError("Condition must be A, B or C")
    c = vector(controls, 5).copy()
    if np.any(c < LOW) or np.any(c > HIGH):
        raise ValueError("Controls outside documented physical ranges")
    if condition == "B":
        c[4] = 1e7
    return c


class BirdModel:
    def __init__(self, condition="A"):
        if condition not in ("A", "B", "C"):
            raise ValueError("Condition must be A, B or C")
        if not LIBRARY.exists():
            raise RuntimeError("Build first: python -m Deeplearning.model")
        self.condition = condition
        self.lib = ct.CDLL(str(LIBRARY))
        ptr = ct.POINTER(ct.c_double)
        self.lib.bird_create.restype = ct.c_void_p
        self.lib.bird_destroy.argtypes = [ct.c_void_p]
        self.lib.bird_history_size.argtypes = [ct.c_void_p]
        self.lib.bird_reset.argtypes = [ct.c_void_p, ptr, ptr, ptr, ct.c_int, ct.c_int]
        self.lib.bird_controls.argtypes = [ct.c_void_p, ptr]
        self.lib.bird_advance.argtypes = [ct.c_void_p, ct.c_int, ptr]
        self.lib.bird_advance_rms.argtypes = [ct.c_void_p, ct.c_int, ptr, ptr]
        self.lib.bird_advance_moments.argtypes = [ct.c_void_p, ct.c_int, ptr, ptr]
        self.lib.bird_state.argtypes = [ct.c_void_p, ptr]
        self.ptr = self.lib.bird_create()
        if not self.ptr:
            raise MemoryError("Unable to create native model")
        self.history_size = self.lib.bird_history_size(self.ptr)
        self.failed = False
        self.reset()

    @staticmethod
    def _pointer(x):
        return x.ctypes.data_as(ct.POINTER(ct.c_double))

    def _check_open(self):
        if not self.ptr:
            raise RuntimeError("Model is closed")

    def _controls(self, controls):
        return effective_controls(controls, self.condition)

    def reset(self, controls=DEFAULT_CONTROLS, *, state=(0, 0, 0, 0), history=None):
        self._check_open()
        c, state = self._controls(controls), vector(state, 4)
        history = np.zeros(self.history_size) if history is None else vector(history, self.history_size)
        rc = self.lib.bird_reset(self.ptr, self._pointer(c), self._pointer(state),
                                 self._pointer(history), self.history_size, "ABC".index(self.condition))
        if rc:
            raise RuntimeError("Native reset failed")
        self.controls, self.failed = c, False
        return self.state

    def set_controls(self, controls):
        self._check_open()
        c = self._controls(controls)
        self.lib.bird_controls(self.ptr, self._pointer(c))
        self.controls = c

    def advance(self, seconds, *, return_rms=False, return_moments=False):
        self._check_open()
        if self.failed:
            raise RuntimeError("Reset required after numerical failure")
        if return_rms and return_moments:
            raise ValueError("Choose RMS or moments, not both")
        if not np.isfinite(seconds) or not 0 < seconds <= 1:
            raise ValueError("Advance duration must be in (0, 1] seconds")
        n = round(seconds/DT)
        if n < 1 or abs(n*DT-seconds) > 1e-12:
            raise ValueError("Duration must be a positive integral multiple of dt")
        out = np.empty(n, dtype=np.float64)
        rms = np.empty(4, dtype=np.float64)
        moments = np.empty(8, dtype=np.float64)
        if return_moments:
            code = self.lib.bird_advance_moments(self.ptr, n, self._pointer(out), self._pointer(moments))
        elif return_rms:
            code = self.lib.bird_advance_rms(self.ptr, n, self._pointer(out), self._pointer(rms))
        else:
            code = self.lib.bird_advance(self.ptr, n, self._pointer(out))
        if code:
            self.failed = True
            raise FloatingPointError("Native integration diverged; reset required")
        if return_moments:
            means, stds = moments[:4], moments[4:]
            return out, dict(mean=means, std=stds, rms=np.hypot(means, stds))
        return (out, rms) if return_rms else out

    @property
    def state(self):
        self._check_open()
        out = np.empty(6)
        self.lib.bird_state(self.ptr, self._pointer(out))
        return out

    def close(self):
        if getattr(self, "ptr", None):
            self.lib.bird_destroy(self.ptr)
            self.ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    build()
    print("Built Deeplearning/build/libbirdsong.so")
