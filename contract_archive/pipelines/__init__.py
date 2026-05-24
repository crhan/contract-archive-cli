"""
OCR Pipelines.

历史：原本有 dashscope/paddleocr/mineru 三路 + 简单工厂 get_pipeline。
重构后只保留 MinerU 一路，工厂消失，使用方直接 import MinerUPipeline。
"""
from .mineru_pipeline import MinerUPipeline

__all__ = ["MinerUPipeline"]
