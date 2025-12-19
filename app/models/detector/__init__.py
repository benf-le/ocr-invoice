"""Detector module"""
from .model import DetectionModel
from .inference import run_detection, run_detection_on_image

__all__ = ["DetectionModel", "run_detection", "run_detection_on_image"]
