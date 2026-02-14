# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""BDD100K dataset integration with Detectron2 for object detection."""

from typing import Any

__all__ = [
    "make_coco_dicts",
    "get_transform",
    "TopKLossVisualizationHook",
    "apply_nms_to_predictions",
    "LossSample",
]


def __getattr__(name: str) -> Any:
    """Lazy import attributes to avoid circular import warnings when running with -m.

    Args:
        name (str): The attribute name to import.

    Returns:
        Any: The requested module attribute.

    Raises:
        AttributeError: If the attribute is not found in __all__.
    """
    if name in __all__:
        # Import the module directly to avoid circular imports
        if name in ["make_coco_dicts", "get_transform"]:
            from .dataset import make_coco_dicts, get_transform

            # Cache the imports in the module namespace
            globals().update(
                {
                    "make_coco_dicts": make_coco_dicts,
                    "get_transform": get_transform,
                }
            )
        elif name in [
            "TopKLossVisualizationHook",
            "apply_nms_to_predictions",
            "LossSample",
        ]:
            from .visualization import (
                TopKLossVisualizationHook,
                apply_nms_to_predictions,
                LossSample,
            )

            # Cache the imports in the module namespace
            globals().update(
                {
                    "TopKLossVisualizationHook": TopKLossVisualizationHook,
                    "apply_nms_to_predictions": apply_nms_to_predictions,
                    "LossSample": LossSample,
                }
            )

        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
