# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

"""Utility functions for BDD100K dataset processing."""

from typing import Any

__all__ = [
    "CLASSES",
    "Annotation",
    "BBox",
    "parse_annotations",
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
        from .parse_annotations import CLASSES, Annotation, BBox, parse_annotations

        # Cache the imports in the module namespace
        globals().update(
            {
                "CLASSES": CLASSES,
                "Annotation": Annotation,
                "BBox": BBox,
                "parse_annotations": parse_annotations,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
