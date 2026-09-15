"""Framework-isolated oil-spill segmentation adapters."""

from oilspill.adapters.detection.config import (
    DeepLabV3PlusSettings,
    DenseInputChannel,
    DenseSpatialTransform,
    DetectorInputChannel,
    UNetSettings,
    YOLOv8SegSettings,
)
from oilspill.adapters.detection.dense import DeepLabV3PlusAdapter, UNetAdapter
from oilspill.adapters.detection.errors import (
    DetectorArtifactError,
    DetectorDependencyError,
    DetectorError,
    DetectorInputError,
    DetectorOutputError,
    MissingModelCheckpointError,
)
from oilspill.adapters.detection.yolov8 import YOLOv8SegAdapter, create_yolov8_seg

__all__ = [
    "DetectorArtifactError",
    "DetectorDependencyError",
    "DetectorError",
    "DetectorInputChannel",
    "DetectorInputError",
    "DetectorOutputError",
    "MissingModelCheckpointError",
    "DeepLabV3PlusAdapter",
    "DeepLabV3PlusSettings",
    "DenseInputChannel",
    "DenseSpatialTransform",
    "UNetAdapter",
    "UNetSettings",
    "YOLOv8SegAdapter",
    "YOLOv8SegSettings",
    "create_yolov8_seg",
]
