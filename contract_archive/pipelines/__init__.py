"""
OCR Pipelines.

当前只保留一个归一化入口：native PDF text layer + DashScope VL OCR。
"""
from .ocr_pipeline import OCRPipeline

__all__ = ["OCRPipeline"]
