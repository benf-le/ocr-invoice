# 🚀 OCR Invoice API Server

Server API FastAPI production-ready để xử lý OCR hóa đơn với khả năng phát hiện và nhận diện văn bản.

## ✨ Tính năng

- 🔍 Phát hiện văn bản sử dụng PaddleOCR DBNet
- 📝 Nhận diện văn bản với CTC decoder
- 🏷️ Trích xuất thông tin hóa đơn (nhà cung cấp, tổng tiền, đơn vị tiền tệ)
- 🎨 Hiển thị kết quả với bounding boxes
- 🚀 FastAPI với hỗ trợ async
- 🐳 Hỗ trợ Docker
- 🧪 Test script tích hợp sẵn

## 📁 Cấu trúc dự án và nhiệm vụ từng file

```
ocr_server/
├── app/
│   ├── main.py                 # 🎯 Điểm khởi đầu FastAPI
│   │                           # - Khởi tạo ứng dụng FastAPI
│   │                           # - Load models khi server start
│   │                           # - Cấu hình CORS, routes
│   │                           # - Health check endpoint
│   │
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   └── ocr.py          # 📡 4 API endpoints chính
│   │   │                       # - POST /invoice: Trích xuất fields
│   │   │                       # - POST /invoice/visualize: Vẽ bbox
│   │   │                       # - POST /invoice/bboxes: Raw bbox
│   │   │                       # - GET /mock: Mock data test
│   │   │
│   │   └── router.py           # 🔀 Router tổng hợp các endpoints
│   │
│   ├── core/
│   │   ├── config.py           # ⚙️ Cấu hình toàn hệ thống
│   │   │                       # - Đường dẫn models
│   │   │                       # - Tham số detection/recognition
│   │   │                       # - Biến môi trường
│   │   │
│   │   └── logger.py           # 📝 Logging system
│   │                           # - Setup logger cho toàn app
│   │                           # - Format log messages
│   │
│   ├── models/
│   │   ├── detector/
│   │   │   ├── model.py        # 🔍 Load model detection
│   │   │   │                   # - Class DetectionModel
│   │   │   │                   # - Load PaddlePaddle model
│   │   │   │
│   │   │   └── inference.py   # 🎯 Inference detection
│   │   │                       # - Tiền xử lý ảnh (resize, pad, normalize)
│   │   │                       # - Chạy model detection
│   │   │                       # - Trả về heatmap
│   │   │
│   │   └── recognizer/
│   │       ├── model.py        # 📖 Load model recognition
│   │       │                   # - Class RecognitionModel
│   │       │                   # - Load PaddlePaddle model
│   │       │
│   │       └── inference.py   # 🔤 Inference recognition
│   │                           # - Crop và resize text regions
│   │                           # - Chạy model recognition
│   │                           # - CTC decode thành text
│   │
│   ├── services/
│   │   └── ocr_service.py      # 🔄 OCR Pipeline chính
│   │                           # - Class OCRService
│   │                           # - Kết hợp detection + recognition
│   │                           # - Trích xuất invoice fields
│   │                           # - Logic xử lý supplier, total, currency
│   │
│   ├── schemas/
│   │   └── ocr.py              # 📋 Pydantic schemas
│   │                           # - InvoiceFieldsResponse
│   │                           # - BBoxListResponse
│   │                           # - MockResponse
│   │                           # - Validation dữ liệu
│   │
│   └── utils/
│       ├── image.py            # 🖼️ Xử lý ảnh
│       │                       # - Expand polygon (nới rộng bbox)
│       │                       # - Extract bboxes từ heatmap
│       │                       # - Visualize bboxes
│       │
│       └── pdf.py              # 📄 Xử lý PDF (placeholder)
│
├── weights/
│   ├── Model_det_small/        # 🎯 Model phát hiện văn bản
│   │   ├── inference.json      # - Kiến trúc model
│   │   └── inference.pdiparams # - Trọng số model
│   │
│   └── Model_rec/              # 🔤 Model nhận diện văn bản
│       ├── inference.json      # - Kiến trúc model
│       └── inference.pdiparams # - Trọng số model
│
├── requirements.txt            # 📦 Danh sách thư viện
├── Dockerfile                  # 🐳 Docker configuration
├── run.sh                      # ▶️ Script chạy server
├── setup_models.sh             # 📥 Script copy models
├── test_api.py                 # 🧪 Script test APIs
└── README.md                   # 📖 File này
```

## 🔧 Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- PaddlePaddle
- OpenCV
- Virtual environment (khuyến nghị)

### Hướng dẫn cài đặt từng bước

#### Bước 1: Copy model weights

Kiểm tra models đã copy đúng chưa:
```bash
ls -la weights/Model_det_small/
ls -la weights/Model_rec/
```

#### Bước 2: Kích hoạt virtual environment

```bash
# Nếu đã có venv sẵn ở thư mục Kyanon
source /Users/macbook/Desktop/Kyanon/myenv/bin/activate

# Hoặc tạo venv mới
python3 -m venv venv
source venv/bin/activate
```

#### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

#### Bước 4: Cấu hình (tùy chọn)

```bash
# Copy file cấu hình env vào
cp .env.example .env
```

## 🚀 Chạy server

### Cách 1: Dùng uvicorn (Linh hoạt)

```bash
# Development mode (tự động reload khi sửa code)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode (nhiều workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Cách 3: Dùng FastAPI CLI (Mới nhất)

```bash
# Development
fastapi dev app/main.py

# Production
fastapi run app/main.py
```

### Cách 4: Docker

```bash
# Build image
docker build -t ocr-api .

# Chạy container
docker run -p 8000:8000 ocr-api
```

### Khi server chạy thành công

Truy cập:
- **Swagger UI**: http://localhost:8000/docs (Giao diện test API)
- **ReDoc**: http://localhost:8000/redoc (Tài liệu API)
- **Health Check**: http://localhost:8000/health

## 📡 Các API Endpoints

### API 1️⃣: Trích xuất thông tin hóa đơn

**POST** `/api/v1/ocr/invoice`

Trích xuất các trường thông tin từ ảnh hóa đơn.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file ảnh (jpg/png)

**Response:**
```json
{
  "supplier_name": "Công ty ACME",
  "total": "12500000",
  "currency": "VND"
}
```

**Ví dụ:**
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/invoice" \
  -F "file=@test1.png"
```

---

### API 2️⃣: OCR với hình ảnh trực quan

**POST** `/api/v1/ocr/invoice/visualize`

Trả về ảnh với các bounding boxes và text đã nhận diện được vẽ lên.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file ảnh

**Response:**
- PNG image với bounding boxes màu xanh và text labels

**Ví dụ:**
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/invoice/visualize" \
  -F "file=@test1.png" \
  --output result.png
```

---

### API 3️⃣: Lấy raw bounding boxes

**POST** `/api/v1/ocr/invoice/bboxes`

Lấy tất cả các text boxes đã phát hiện với tọa độ và nội dung.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file ảnh

**Response:**
```json
{
  "results": [
    {
      "label": "Công ty ACME",
      "text": "Công ty ACME",
      "bbox": [100, 50, 300, 80]
    },
    {
      "label": "12500000",
      "text": "12500000",
      "bbox": [400, 200, 550, 230]
    }
  ]
}
```

**Ví dụ:**
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/invoice/bboxes" \
  -F "file=@test1.png"
```

---

### API 4️⃣: Mock data (cho testing)

**GET** `/api/v1/ocr/mock`

Trả về dữ liệu mẫu không cần model, dùng để test integration.

**Request:**
- Method: GET
- Không cần params

**Response:**
```json
{
  "supplier_name": "ACME Corporation",
  "total": "12500000",
  "currency": "VND"
}
```

**Ví dụ:**
```bash
curl http://localhost:8000/api/v1/ocr/mock
```

## 🧪 Test API

### Cách 1: Dùng test script Python (Khuyến nghị)

```bash
# Test với 1 ảnh cụ thể (chỉ cần tên file)
python test_api.py test1.png

# Test với đường dẫn đầy đủ
python test_api.py /Users/macbook/Desktop/Kyanon/test2.jpg

# Test tất cả ảnh có sẵn
python test_api.py --all

# Chế độ tương tác (chọn ảnh)
python test_api.py
```

**Kết quả test script:**
- ✅ Hiển thị đẹp với màu sắc
- ✅ Tự động tìm ảnh trong thư mục Kyanon
- ✅ Test cả 4 APIs
- ✅ Lưu ảnh visualization
- ✅ Hiển thị JSON response

### Cách 2: Dùng Swagger UI (Trực quan nhất)

1. Mở trình duyệt: http://localhost:8000/docs
2. Chọn endpoint muốn test
3. Click "Try it out"
4. Upload file ảnh
5. Click "Execute"
6. Xem kết quả

### Cách 3: Dùng cURL

```bash
# Test health check
curl http://localhost:8000/health

# Test mock API
curl http://localhost:8000/api/v1/ocr/mock

# Test extract invoice
curl -X POST "http://localhost:8000/api/v1/ocr/invoice" \
  -F "file=@test1.png"

# Test visualization
curl -X POST "http://localhost:8000/api/v1/ocr/invoice/visualize" \
  -F "file=@test1.png" \
  --output result.png

# Test bboxes
curl -X POST "http://localhost:8000/api/v1/ocr/invoice/bboxes" \
  -F "file=@test1.png"
```

### Cách 4: Dùng Python script tùy chỉnh

```python
import requests

# Test extract invoice fields
with open('test1.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/ocr/invoice',
        files={'file': f}
    )
    print(response.json())

# Test bounding boxes
with open('test1.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/ocr/invoice/bboxes',
        files={'file': f}
    )
    print(response.json())

# Test visualization
with open('test1.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/ocr/invoice/visualize',
        files={'file': f}
    )
    with open('result.png', 'wb') as out:
        out.write(response.content)
```

## ⚙️ Cấu hình

### File `.env`

Copy và chỉnh sửa file cấu hình:
```bash
cp .env.example .env
```

### Các tham số cấu hình

```bash
# Detection (Phát hiện văn bản)
DETECTION_RESIZE_LONG=960      # Kích thước max của ảnh (càng lớn càng chính xác nhưng chậm)
DETECTION_THRESH=0.3           # Ngưỡng phát hiện (0.0-1.0, thấp = nhiều box hơn)
DETECTION_BOX_THRESH=0.6       # Ngưỡng lọc box (0.0-1.0)

# Recognition (Nhận diện văn bản)
RECOGNITION_TARGET_H=48        # Chiều cao chuẩn của text crop
RECOGNITION_TARGET_W=320       # Chiều rộng chuẩn của text crop

# Image Processing (Xử lý ảnh)
EXPAND_RATIO_W=0.085          # Tỷ lệ nới rộng bbox theo chiều ngang
EXPAND_RATIO_H=0.2            # Tỷ lệ nới rộng bbox theo chiều dọc
MIN_PAD_H=3                   # Padding tối thiểu chiều dọc (px)
MAX_PAD_H=15                  # Padding tối đa chiều dọc (px)

# Server
DEBUG=False                    # Bật/tắt debug mode
```

### Tùy chỉnh theo nhu cầu

**Muốn phát hiện nhiều text hơn:**
```bash
DETECTION_THRESH=0.2           # Giảm ngưỡng xuống
```

**Muốn bbox rộng hơn (crop nhiều hơn):**
```bash
EXPAND_RATIO_W=0.15
EXPAND_RATIO_H=0.3
```

**Xử lý ảnh lớn:**
```bash
DETECTION_RESIZE_LONG=1280     # Tăng kích thước max
```

## 📊 Thông tin Models

### Detection Model (Model_det_small)
- **Kiến trúc**: DBNet (PaddleOCR)
- **Input**: Ảnh RGB, được resize và pad thành bội số của 32
- **Output**: Heatmap segmentation (ma trận xác suất vùng có text)
- **Kích thước**: ~8-10 MB
- **Chức năng**: Tìm vị trí các vùng có văn bản trong ảnh

### Recognition Model (Model_rec)
- **Kiến trúc**: CTC-based CRNN
- **Input**: Ảnh text đã crop, 48x320 pixels
- **Output**: Chuỗi ký tự
- **Character set**: 
  - Chữ số: 0-9
  - Chữ cái: a-z, A-Z
  - Ký tự đặc biệt: space, . , ! ? - _ / : ( ) @ + = % $
- **Kích thước**: ~10-12 MB
- **Chức năng**: Nhận diện text từ các vùng đã crop

### Pipeline OCR hoạt động như thế nào?

```
1. [Ảnh đầu vào] 
   ↓
2. [Detection Model] → Tìm vị trí text boxes
   ↓
3. [Extract Bboxes] → Tính toán tọa độ chính xác
   ↓
4. [Crop từng box] → Cắt từng vùng text
   ↓
5. [Recognition Model] → Nhận diện text mỗi box
   ↓
6. [Field Extraction] → Trích xuất supplier, total, currency
   ↓
7. [Kết quả JSON]
```

## 💡 Tips tối ưu hiệu suất

### 1. CPU Optimization
PaddlePaddle chạy trên CPU mặc định. Để tăng tốc:
```bash
# Dùng nhiều workers cho production
uvicorn app.main:app --workers 4
```

### 2. Xử lý batch
Nếu có nhiều ảnh, xử lý song song:
```python
import asyncio
import aiohttp

async def process_images(image_paths):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for path in image_paths:
            tasks.append(upload_image(session, path))
        return await asyncio.gather(*tasks)
```

### 3. Tiền xử lý ảnh
Resize ảnh lớn trước khi gửi:
```python
from PIL import Image

img = Image.open('large_image.jpg')
if max(img.size) > 2000:
    img.thumbnail((2000, 2000))
    img.save('resized.jpg')
```

### 4. Cache kết quả
Cache response cho ảnh đã xử lý:
```python
# Trong production, dùng Redis hoặc Memcached
from functools import lru_cache
```

## 🐛 Xử lý lỗi thường gặp

### Lỗi: Model không load được

**Triệu chứng:**
```
FileNotFoundError: Model files not found
```

**Giải pháp:**
```bash
# Kiểm tra models có đúng vị trí không
ls -la weights/Model_det_small/
ls -la weights/Model_rec/

# Copy lại nếu thiếu
./setup_models.sh
```

### Lỗi: Import errors

**Triệu chứng:**
```
ModuleNotFoundError: No module named 'paddle'
```

**Giải pháp:**
```bash
# Cài lại dependencies
pip install --force-reinstall -r requirements.txt

# Hoặc cài từng package
pip install paddlepaddle
pip install opencv-python
```

### Lỗi: Out of Memory

**Triệu chứng:**
Server bị crash khi xử lý ảnh lớn.

**Giải pháp:**
```bash
# Giảm kích thước resize trong .env
DETECTION_RESIZE_LONG=640
```

### Lỗi: Port đã được sử dụng

**Triệu chứng:**
```
ERROR: [Errno 48] Address already in use
```

**Giải pháp:**
```bash
# Đổi port
uvicorn app.main:app --port 8001

# Hoặc kill process cũ
lsof -ti:8000 | xargs kill -9
```

### Lỗi: Cannot connect to server

**Triệu chứng:**
Test script báo "Cannot connect to server"

**Giải pháp:**
```bash
# 1. Kiểm tra server có chạy không
curl http://localhost:8000/health

# 2. Nếu không, chạy server
uvicorn app.main:app --reload

# 3. Đảm bảo đúng port
# Test script mặc định dùng port 8000
```

## 🎯 Use Cases thực tế

### 1. Xử lý hóa đơn hàng loạt
```python
import os
import requests

invoice_dir = '/path/to/invoices'
results = []

for filename in os.listdir(invoice_dir):
    if filename.endswith(('.jpg', '.png')):
        with open(os.path.join(invoice_dir, filename), 'rb') as f:
            response = requests.post(
                'http://localhost:8000/api/v1/ocr/invoice',
                files={'file': f}
            )
            results.append({
                'filename': filename,
                'data': response.json()
            })

print(f"Processed {len(results)} invoices")
```

### 2. Tích hợp vào web app
```javascript
// Frontend upload
async function uploadInvoice(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/api/v1/ocr/invoice', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    console.log('Supplier:', data.supplier_name);
    console.log('Total:', data.total);
    console.log('Currency:', data.currency);
}
```

### 3. Validation và post-processing
```python
def validate_invoice_data(data):
    """Validate và clean dữ liệu sau OCR"""
    
    # Clean total amount
    if data['total']:
        # Remove non-numeric characters
        data['total'] = ''.join(filter(str.isdigit, data['total']))
    
    # Validate currency
    valid_currencies = ['VND', 'USD', 'EUR']
    if data['currency'] not in valid_currencies:
        data['currency'] = 'VND'  # Default
    
    return data
```

## 📞 Hỗ trợ và Liên hệ

### Tài liệu bổ sung
- `QUICKSTART.md` - Hướng dẫn nhanh
- `SETUP.md` - Hướng dẫn setup chi tiết
- `PROJECT_SUMMARY.md` - Tổng quan dự án
- `CHECKLIST.md` - Checklist kiểm tra

### Báo lỗi
Nếu gặp lỗi, hãy cung cấp:
1. Log output của server
2. File ảnh test (nếu có thể)
3. Các bước tái hiện lỗi
4. Môi trường (OS, Python version)

### Đóng góp
Contributions are welcome! Vui lòng:
1. Fork repository
2. Tạo branch mới
3. Commit changes
4. Tạo Pull Request

---

**Phiên bản**: 1.0.0  
**Ngày cập nhật**: December 2025  
**Tác giả**: [Your Name]  
**License**: [Your License]
