import cv2
import numpy as np
import paddle
from typing import List
from app.core.logger import logger


def resize_keep_ratio(img, target_h=48, target_w=320):
    """Resize image keeping aspect ratio and pad"""
    h, w = img.shape[:2]
    ratio = target_h / h
    new_w = int(w * ratio)
    if new_w > target_w:
        new_w = target_w
    
    resized = cv2.resize(img, (new_w, target_h))
    
    # Padding right side
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[:, :new_w] = resized
    return padded


def preprocess_for_recognition(image, bbox):
    """Crop and preprocess bbox for recognition"""
    x1, y1, x2, y2 = bbox[:4]
    
    # Crop region
    cropped = image[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return None
    
    # Resize to standard size for recognition (48x320)
    cropped = resize_keep_ratio(cropped)
    
    # Convert BGR to RGB and normalize
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped = cropped.astype(np.float32) / 255.0
    
    # Convert to [C, H, W]
    cropped = np.transpose(cropped, (2, 0, 1))
    
    # Add batch dimension [N, C, H, W]
    cropped = np.expand_dims(cropped, axis=0)
    
    return cropped


def create_character_dict():
    """Create character dictionary"""
    chars = ['<blank>']  # CTC blank token
    chars.extend(list('0123456789'))
    chars.extend(list('abcdefghijklmnopqrstuvwxyz'))
    chars.extend(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    chars.extend([' ', '.', ',', '!', '?', '-', '_', '/', ':', '(', ')', '@', '+', '=', '%', '$'])
    chars.extend(['+', '-', '.', ',', ' '])
    
    return chars


def ctc_decode(logits, char_dict):
    """CTC decode"""
    logits = logits.squeeze(0)
    pred_indices = paddle.argmax(logits, axis=1)
    pred_indices = pred_indices.numpy()
    
    decoded = []
    prev_idx = -1
    
    for idx in pred_indices:
        if idx != prev_idx and idx != 0:
            if idx < len(char_dict):
                decoded.append(char_dict[idx])
        prev_idx = idx
    
    return ''.join(decoded)


def run_recognition_on_bbox(rec_model, image, bbox):
    """Run recognition on one bbox"""
    # Crop and preprocess
    img_data = preprocess_for_recognition(image, bbox)
    if img_data is None:
        return ""
    
    try:
        # Inference
        img_tensor = paddle.to_tensor(img_data, dtype='float32')
        with paddle.no_grad():
            output = rec_model(img_tensor)
        
        # Decode
        char_dict = create_character_dict()
        text = ctc_decode(output, char_dict)
        
        return text
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return ""
