import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "OCR Invoice API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    
    # Model paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    DETECTOR_MODEL_PATH: Path = WEIGHTS_DIR / "Model_det_small"
    RECOGNIZER_MODEL_PATH: Path = WEIGHTS_DIR / "Model_rec"
    
    # Detection settings
    DETECTION_RESIZE_LONG: int = 960
    DETECTION_THRESH: float = 0.3
    DETECTION_BOX_THRESH: float = 0.6
    
    # Recognition settings
    RECOGNITION_TARGET_H: int = 48
    RECOGNITION_TARGET_W: int = 320
    
    # Image processing
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png"}
    
    # Expand polygon settings
    EXPAND_RATIO_W: float = 0.085
    EXPAND_RATIO_H: float = 0.2
    MIN_PAD_H: int = 3
    MAX_PAD_H: int = 15
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
