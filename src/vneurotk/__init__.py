"""VneuroTK: 神经科学数据工具包

一个用于神经科学数据管理、分析和可视化的工具包。
"""

from __future__ import annotations

from vneurotk import utils
from vneurotk.io import read
from vneurotk.vision.extractor.extractor import VisionExtractor

__version__ = "0.1.0"
__author__ = "VneuroTK Contributors"

__all__ = ["__version__", "read", "utils", "VisionExtractor"]
