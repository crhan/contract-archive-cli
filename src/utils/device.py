"""
设备选择策略：MPS → CUDA → CPU 自动降级。

未来切到 RTX 5080 时只需安装 cu128 PyTorch，本函数会自动选 cuda；
当前在 Mac 上会选 mps（M-series）或 cpu（Intel Mac）。
"""
from __future__ import annotations

import os
from typing import Literal

Device = Literal["mps", "cuda", "cpu"]


def select_device(prefer: str | None = None) -> Device:
    """
    按优先级返回当前可用的最佳设备。

    :param prefer: 用户偏好。"auto" 或 None 走自动；其它值（mps/cuda/cpu）强制使用。
    :return: 实际选定的设备名。
    """
    prefer = (prefer or os.getenv("COMPUTE_DEVICE", "auto")).lower()

    if prefer in {"mps", "cuda", "cpu"}:
        return prefer  # type: ignore[return-value]

    # auto 模式：依次尝试
    try:
        import torch  # 延迟导入避免无 torch 环境报错
    except ImportError:
        return "cpu"

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def describe_device(device: Device) -> str:
    """返回设备人类可读描述，用于日志/报告。"""
    if device == "cuda":
        try:
            import torch

            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    if device == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"
