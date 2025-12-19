from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logger import logger
from app.api.v1.router import api_router
from app.models.detector import DetectionModel
from app.models.recognizer import RecognitionModel
from app.services.ocr_service import OCRService
from app.api.v1.endpoints.ocr import set_ocr_service


# Global model holders
det_model_instance = None
rec_model_instance = None
det_model = None
rec_model = None
ocr_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    Models are loaded once at startup
    """
    global det_model_instance, rec_model_instance, det_model, rec_model, ocr_service
    
    # Startup: Load models
    logger.info("=" * 60)
    logger.info("Starting OCR API Server")
    logger.info("=" * 60)
    
    try:
        # Load Detection Model
        logger.info("Loading Detection Model...")
        det_model_instance = DetectionModel(str(settings.DETECTOR_MODEL_PATH))
        det_model = det_model_instance.load_detection_model()
        logger.info("✓ Detection model loaded successfully")
        
        # Load Recognition Model
        logger.info("Loading Recognition Model...")
        rec_model_instance = RecognitionModel(str(settings.RECOGNIZER_MODEL_PATH))
        rec_model = rec_model_instance.load_recognition_model()
        logger.info("✓ Recognition model loaded successfully")
        
        # Initialize OCR Service
        logger.info("Initializing OCR Service...")
        ocr_service = OCRService(det_model, rec_model)
        set_ocr_service(ocr_service)
        logger.info("✓ OCR Service initialized")
        
        logger.info("=" * 60)
        logger.info("Server startup completed successfully!")
        logger.info(f"API Documentation: http://localhost:8000/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down OCR API Server...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="OCR Invoice API - Extract text and fields from invoice images",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OCR Invoice API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": det_model is not None and rec_model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
