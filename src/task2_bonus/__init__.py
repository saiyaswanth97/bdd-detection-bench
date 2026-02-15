# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""
Task2b: Faster R-CNN object detection with PyTorch
"""

from typing import Any

__all__ = ["load_model", "BDDDataset", "calculate_ap_all_classes", "apply_nms"]


def __getattr__(name: str) -> Any:
    """Lazy import to avoid loading heavy dependencies until needed.

    Args:
        name (str): The name of the attribute to import.

    Returns:
        Any: The requested module attribute.

    Raises:
        AttributeError: If the attribute name is not recognized.
    """
    if name == "load_model":
        from .model import load_model

        return load_model
    elif name == "BDDDataset":
        from .dataset import BDDDataset

        return BDDDataset
    elif name == "calculate_ap_all_classes":
        from .metrics import calculate_ap_all_classes

        return calculate_ap_all_classes
    elif name == "apply_nms":
        from .metrics import apply_nms

        return apply_nms
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
