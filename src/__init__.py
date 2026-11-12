"""vn-plate-ocr: Vietnamese license plate detection (YOLOv8) and recognition (EasyOCR)."""

__version__ = "0.1.0"

# Type aliases shared across modules.
# Box is (x1, y1, x2, y2) in absolute pixel coordinates of the source image.
Box = tuple[int, int, int, int]

__all__ = ["Box", "__version__"]
