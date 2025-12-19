from fastapi import APIRouter
from app.api.v1.endpoints import ocr

api_router = APIRouter()

# Include OCR endpoints
api_router.include_router(
    ocr.router,
    prefix="/ocr",
    tags=["OCR"]
)
