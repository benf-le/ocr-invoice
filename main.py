# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os
import time
from datetime import datetime

# Import các module của chúng ta
from detection import load_db_predictor, run_detection
from recognition import load_recognition_model, recognize_text
from extract_amount import extract_grand_total
from config import DET_DIR, REC_DIR, DICT_PATH

app = FastAPI(title="OCR Invoice Amount Extractor")

# Global variables để load model một lần
det_predictor = None
rec_model = None

@app.on_event("startup")
async def load_models():
    """Load models khi server khởi động"""
    global det_predictor, rec_model
    
    print("🔄 Đang load models...")
    
    det_predictor = load_db_predictor(DET_DIR)
    if det_predictor is None:
        raise Exception("Không thể load Detection model!")
    
    rec_model = load_recognition_model(REC_DIR)
    if rec_model is None:
        raise Exception("Không thể load Recognition model!")
    
    print("✅ Tất cả models đã sẵn sàng!")

@app.get("/")
async def root():
    return {"message": "OCR Invoice Amount Extractor API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "detection_model_loaded": det_predictor is not None,
        "recognition_model_loaded": rec_model is not None
    }

@app.post("/webhook/process-image")
async def process_image(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    start_iso = datetime.utcnow().isoformat() + "Z"
    print(f"[PROCESS] start={start_iso}")

    try:
        contents = await file.read()

        # Bước 1: đọc ảnh
        t_read = time.perf_counter()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Không thể đọc ảnh. Vui lòng kiểm tra định dạng file.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t_after_decode = time.perf_counter()
        print(f"[STEP] decode_image_ms={(t_after_decode - t_read)*1000:.1f}")

        # Bước 2: Detection
        print("🔄 Đang chạy Detection...")
        t_det = time.perf_counter()
        boxes = run_detection(det_predictor, img_rgb)
        t_after_det = time.perf_counter()
        print(f"[STEP] detection_ms={(t_after_det - t_det)*1000:.1f}, boxes={len(boxes)}")

        if len(boxes) == 0:
            t_end = time.perf_counter()
            end_iso = datetime.utcnow().isoformat() + "Z"
            print(f"[PROCESS] end={end_iso}, total_ms={(t_end - t0)*1000:.1f}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "amount": "0",
                    "message": "Không tìm thấy text nào trong ảnh",
                    "boxes_detected": 0,
                    "start_time": start_iso,
                    "end_time": end_iso,
                    "total_ms": round((t_end - t0)*1000, 1)
                }
            )

        # Bước 3: Recognition
        print("🔄 Đang chạy Recognition...")
        t_rec = time.perf_counter()
        ocr_results = recognize_text(rec_model, img_rgb, boxes, DICT_PATH)
        t_after_rec = time.perf_counter()
        print(f"[STEP] recognition_ms={(t_after_rec - t_rec)*1000:.1f}, texts={len(ocr_results)}")

        if len(ocr_results) == 0:
            t_end = time.perf_counter()
            end_iso = datetime.utcnow().isoformat() + "Z"
            print(f"[PROCESS] end={end_iso}, total_ms={(t_end - t0)*1000:.1f}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "amount": "0",
                    "message": "Không nhận diện được text nào",
                    "texts_detected": 0,
                    "start_time": start_iso,
                    "end_time": end_iso,
                    "total_ms": round((t_end - t0)*1000, 1)
                }
            )

        # Bước 4: Extract Amount
        print("🔄 Đang trích xuất số tiền...")
        t_ext = time.perf_counter()
        amount = extract_grand_total(ocr_results)
        t_after_ext = time.perf_counter()
        print(f"[STEP] extract_amount_ms={(t_after_ext - t_ext)*1000:.1f}")

        # Tổng kết
        t_end = time.perf_counter()
        end_iso = datetime.utcnow().isoformat() + "Z"
        print(f"[PROCESS] end={end_iso}, total_ms={(t_end - t0)*1000:.1f}")

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "amount": amount,
                "boxes_detected": len(boxes),
                "texts_recognized": len(ocr_results),
                "total_ms": round((t_end - t0)*1000, 1),
                "decode_ms": round((t_after_decode - t_read)*1000, 1),
                "detection_ms": round((t_after_det - t_det)*1000, 1),
                "recognition_ms": round((t_after_rec - t_rec)*1000, 1),
                "extract_ms": round((t_after_ext - t_ext)*1000, 1)
            }
        )

    except Exception as e:
        t_end = time.perf_counter()
        end_iso = datetime.utcnow().isoformat() + "Z"
        print(f"[PROCESS] error end={end_iso}, total_ms={(t_end - t0)*1000:.1f}, err={e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)