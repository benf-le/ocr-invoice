from typing import Optional, List
from pydantic import BaseModel, Field


class InvoiceFieldsResponse(BaseModel):
    """Response schema for invoice field extraction"""
    supplier_name: Optional[str] = Field(None, description="Supplier/vendor name")
    total: Optional[str] = Field(None, description="Total amount")
    currency: Optional[str] = Field(None, description="Currency")


class BBoxResult(BaseModel):
    """Single bounding box result"""
    label: str = Field(..., description="Text label/content")
    text: str = Field(..., description="Recognized text")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")


class BBoxListResponse(BaseModel):
    """Response schema for bounding box list"""
    results: List[BBoxResult] = Field(..., description="List of detection results")


class MockResponse(BaseModel):
    """Mock response for testing"""
    supplier_name: str = Field(..., description="Mock supplier name")
    total: str = Field(..., description="Mock total amount")
    currency: str = Field(..., description="Mock currency")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
