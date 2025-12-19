import cv2
import numpy as np
import paddle
from typing import List, Tuple
from app.core.logger import logger


def expand_polygon_adaptive(
    box,
    expand_ratio_w=0.25,
    expand_ratio_h=0.25,
    min_pad_w=5, max_pad_w=30,
    min_pad_h=4, max_pad_h=20
):
    """
    Expand polygon adaptively based on box size
    
    Args:
        box: Polygon points [[x1,y1], [x2,y2], ...]
        expand_ratio_w: Width expansion ratio
        expand_ratio_h: Height expansion ratio
        min_pad_w: Minimum width padding
        max_pad_w: Maximum width padding
        min_pad_h: Minimum height padding
        max_pad_h: Maximum height padding
        
    Returns:
        Expanded polygon points
    """
    box = box.astype(np.float32)
    
    # Calculate center
    cx = np.mean(box[:, 0])
    cy = np.mean(box[:, 1])
    center = np.array([cx, cy])
    
    # Calculate current width and height
    w_curr = np.max(box[:, 0]) - np.min(box[:, 0])
    h_curr = np.max(box[:, 1]) - np.min(box[:, 1])
    
    # Avoid division by zero
    w_curr = max(w_curr, 1.0)
    h_curr = max(h_curr, 1.0)
    
    # Calculate padding
    pad_w_pixel = w_curr * expand_ratio_w
    pad_w_pixel = np.clip(pad_w_pixel, min_pad_w, max_pad_w)
    
    pad_h_pixel = h_curr * expand_ratio_h
    pad_h_pixel = np.clip(pad_h_pixel, min_pad_h, max_pad_h)
    
    # Calculate scale
    scale_x = (w_curr + 2 * pad_w_pixel) / w_curr
    scale_y = (h_curr + 2 * pad_h_pixel) / h_curr
    
    # Expand
    vectors = box - center
    expanded = np.zeros_like(vectors)
    expanded[:, 0] = center[0] + vectors[:, 0] * scale_x
    expanded[:, 1] = center[1] + vectors[:, 1] * scale_y
    
    return expanded.astype(np.int32)


def extract_bboxes_from_output(
    output,
    original_img: np.ndarray,
    conf_threshold: float = 0.2,
    expand_ratio_w: float = 0.085,
    expand_ratio_h: float = 0.2,
    min_pad_h: int = 3,
    max_pad_h: int = 15
) -> List[List]:
    """
    Extract bounding boxes from detection output
    
    Args:
        output: Model output
        original_img: Original image
        conf_threshold: Confidence threshold
        expand_ratio_w: Width expansion ratio
        expand_ratio_h: Height expansion ratio
        min_pad_h: Minimum height padding
        max_pad_h: Maximum height padding
        
    Returns:
        List of bboxes [x1, y1, x2, y2, confidence]
    """
    h, w = original_img.shape[:2]
    bboxes = []
    
    # Convert paddle.Tensor to numpy
    if isinstance(output, paddle.Tensor):
        output = output.numpy()
    
    # If output is tuple (PPOCR style)
    if isinstance(output, (tuple, list)):
        output = output[0]
    
    logger.info(f"Output shape: {output.shape}")
    
    # CASE 1: Segmentation format (DBNet / PPOCR)
    if len(output.shape) in [3, 4] and output.shape[1] <= 3:
        logger.info("Format: Segmentation (DBNet-style)")
        
        heatmap = output.squeeze()  # (H, W)
        
        # Threshold
        binary = (heatmap > conf_threshold).astype(np.uint8) * 255
        
        # Resize to original image
        binary_resized = cv2.resize(binary, (w, h))
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        contours, _ = cv2.findContours(binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Found {len(contours)} contours")
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 80:
                continue
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = expand_polygon_adaptive(
                box,
                expand_ratio_w=expand_ratio_w,
                expand_ratio_h=expand_ratio_h,
                min_pad_h=min_pad_h,
                max_pad_h=max_pad_h
            )
            
            xs = box[:, 0]
            ys = box[:, 1]
            
            x1 = max(int(xs.min()), 0)
            y1 = max(int(ys.min()), 0)
            x2 = min(int(xs.max()), w)
            y2 = min(int(ys.max()), h)
            
            # Calculate confidence
            region = heatmap_resized[y1:y2, x1:x2]
            conf = float(np.mean(region)) if region.size > 0 else 0.0
            conf = max(min(conf, 1.0), 0.0)
            
            bboxes.append([x1, y1, x2, y2, conf])
    
    # CASE 2: YOLO-like format (nx5 or nx6)
    elif len(output.shape) == 3 and output.shape[2] >= 5:
        logger.info("Format: YOLO-like")
        
        detections = output[0]
        for det in detections:
            conf = det[4]
            if conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = det[:4]
            
            # Normalize to pixel coordinates
            if x2 <= 1.0:
                x1 = int(x1 * w)
                x2 = int(x2 * w)
                y1 = int(y1 * h)
                y2 = int(y2 * h)
            else:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Add padding
            pad = 4
            x1 = max(x1 - pad, 0)
            y1 = max(y1 - pad, 0)
            x2 = min(x2 + pad, w)
            y2 = min(y2 + pad, h)
            
            bboxes.append([x1, y1, x2, y2, conf])
    
    else:
        logger.warning(f"Unknown output format: {output.shape}")
    
    logger.info(f"Total BBOX before filtering: {len(bboxes)}")
    
    # Filter by confidence
    bboxes = [b for b in bboxes if b[4] >= conf_threshold]
    logger.info(f"Remaining after conf filter {conf_threshold}: {len(bboxes)}")
    
    return bboxes


def visualize_bboxes(image: np.ndarray, bboxes: List[List]) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        bboxes: List of bboxes [x1, y1, x2, y2, conf]
        
    Returns:
        Image with drawn bboxes
    """
    img_draw = image.copy()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, conf = bbox
        
        # Draw rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{i+1}: {conf:.2f}"
        cv2.putText(
            img_draw, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    
    return img_draw


def visualize_ocr_results(image: np.ndarray, results: List[dict]) -> np.ndarray:
    """
    Draw OCR results on image
    
    Args:
        image: Input image
        results: List of OCR results with 'bbox', 'text', 'confidence'
        
    Returns:
        Image with drawn results
    """
    img_result = image.copy()
    
    for i, result in enumerate(results):
        x1, y1, x2, y2 = result['bbox']
        text = result['text']
        conf = result.get('confidence', 0.0)
        
        # Draw rectangle
        cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw text
        label = f"{text}"
        cv2.putText(
            img_result, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    
    return img_result
