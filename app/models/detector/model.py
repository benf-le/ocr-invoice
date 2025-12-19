import os
import paddle
from pathlib import Path
from app.core.logger import logger


class DetectionModel:
    """Detection model loader"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        
    def load_detection_model(self):
        """Load model detection"""
        try:
            model_file = self.model_path / "inference.json"
            params_file = self.model_path / "inference.pdiparams"
            
            if not model_file.exists() or not params_file.exists():
                logger.error(f"Model files not found at {self.model_path}")
                raise FileNotFoundError(f"Model files not found at {self.model_path}")
            
            logger.info(f"Loading detection model from {self.model_path}")
            logger.info(f"   - {model_file}")
            logger.info(f"   - {params_file}")
            
            self.model = paddle.jit.load(str(self.model_path / "inference"))
            logger.info("Detection model loaded successfully!")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading detection model: {e}")
            raise
    
    def get_model(self):
        """Get loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_detection_model() first.")
        return self.model
