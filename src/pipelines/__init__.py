"""
OCR Pipelines.

三路彼此独立可单独 import 与运行，互不耦合。
"""
from .base import BasePipeline


def get_pipeline(name: str, **kwargs) -> BasePipeline:
    """
    简单工厂——避免在三个模块顶层互相 import 引发依赖冲突。
    每个分支只在被选中时才 lazy import 对应模块（这样某一路依赖装不上也不影响其他路）。
    """
    name = name.lower()
    if name == "dashscope":
        from .dashscope_pipeline import DashScopePipeline

        return DashScopePipeline(**kwargs)
    if name == "paddleocr":
        from .paddleocr_pipeline import PaddleOCRPipeline

        return PaddleOCRPipeline(**kwargs)
    if name == "mineru":
        from .mineru_pipeline import MinerUPipeline

        return MinerUPipeline(**kwargs)
    raise ValueError(
        f"Unknown pipeline {name!r}; expected one of dashscope/paddleocr/mineru"
    )


__all__ = ["BasePipeline", "get_pipeline"]
