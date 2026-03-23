import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops
from .qwen2 import load_qwen2


def load_shared_library():
    lib_dir = Path(__file__).parent

    candidates = []
    if sys.platform.startswith("linux"):
        candidates = ["libllaisys.so"]
    elif sys.platform == "win32":
        candidates = ["llaisys.dll"]
    elif sys.platform == "darwin":
        candidates = ["libllaisys.dylib", "llaisys.dylib"]
    else:
        raise RuntimeError("Unsupported platform")

    for libname in candidates:
        lib_path = os.path.join(lib_dir, libname)
        if os.path.isfile(lib_path):
            return ctypes.CDLL(str(lib_path))

    raise FileNotFoundError(
        f"Shared library not found in {lib_dir}, tried: {', '.join(candidates)}"
    )


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2(LIB_LLAISYS)


__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
    "load_qwen2",
]
