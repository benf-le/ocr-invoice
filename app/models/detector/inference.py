import cv2
import numpy as np
import paddle
from typing import Tuple, List, Optional
from app.core.logger import logger


def run_detection(
    det_model,
    image_path: str,
    resize_long: int = 960,
    thresh: float = 0.3,
    box_thresh: float = 0.6
) -> Tuple[np.ndarray, paddle.Tensor]:
    """
    Run detection on image
    
    Args:
        det_model: Loaded detection model
        image_path: Path to input image
        resize_long: Maximum dimension for resizing
        thresh: Detection threshold
        box_thresh: Bounding box threshold
        
    Returns:
        Tuple of (original_image, model_output)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Cannot load image: {image_path}")
        raise ValueError(f"Cannot load image: {image_path}")
    
    h, w = img.shape[:2]
    logger.info(f"Original image size: {w}x{h}")
    
    # Resize keeping aspect ratio
    if max(h, w) > resize_long:
        if h > w:
            new_h = resize_long
            new_w = int(w * resize_long / h)
        else:
            new_w = resize_long
            new_h = int(h * resize_long / w)
    else:
        new_h, new_w = h, w
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Pad to multiple of 32
    pad_h = ((new_h + 31) // 32) * 32
    pad_w = ((new_w + 31) // 32) * 32
    
    pad_top = (pad_h - new_h) // 2
    pad_bottom = pad_h - new_h - pad_top
    pad_left = (pad_w - new_w) // 2
    pad_right = pad_w - new_w - pad_left
    
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    
    logger.info(f"Resized: {new_w}x{new_h}, Padded: {pad_w}x{pad_h}")
    
    # Normalize
    img_norm = img_padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_norm = (img_norm - mean) / std
    
    # CHW + batch
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    
    logger.info(f"Input shape: {img_input.shape}")
    
    # Inference
    img_tensor = paddle.to_tensor(img_input, dtype='float32')
    with paddle.no_grad():
        output = det_model(img_tensor)
    
    logger.info("Detection completed!")
    
    return img, output


def run_detection_on_image(
    det_model,
    image: np.ndarray,
    resize_long: int = 960,
    thresh: float = 0.3,
    box_thresh: float = 0.6
) -> paddle.Tensor:
    """
    Run detection on numpy image array
    
    Args:
        det_model: Loaded detection model
        image: Input image as numpy array
        resize_long: Maximum dimension for resizing
        thresh: Detection threshold
        box_thresh: Bounding box threshold
        
    Returns:
        Model output tensor
    """
    h, w = image.shape[:2]
    logger.info(f"Image size: {w}x{h}")
    
    # Resize keeping aspect ratio
    if max(h, w) > resize_long:
        if h > w:
            new_h = resize_long
            new_w = int(w * resize_long / h)
        else:
            new_w = resize_long
            new_h = int(h * resize_long / w)
    else:
        new_h, new_w = h, w
    
    img_resized = cv2.resize(image, (new_w, new_h))
    
    # Pad to multiple of 32
    pad_h = ((new_h + 31) // 32) * 32
    pad_w = ((new_w + 31) // 32) * 32
    
    pad_top = (pad_h - new_h) // 2
    pad_bottom = pad_h - new_h - pad_top
    pad_left = (pad_w - new_w) // 2
    pad_right = pad_w - new_w - pad_left
    
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    
    # Normalize
    img_norm = img_padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_norm = (img_norm - mean) / std
    
    # CHW + batch
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    
    # Inference
    img_tensor = paddle.to_tensor(img_input, dtype='float32')
    with paddle.no_grad():
        output = det_model(img_tensor)
    
    return output
