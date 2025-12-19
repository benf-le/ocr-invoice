import re
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.core.logger import logger
from app.models.detector.inference import run_detection_on_image
from app.models.recognizer.inference import run_recognition_on_bbox
from app.utils.image import extract_bboxes_from_output, visualize_ocr_results
from app.core.config import settings


class OCRService:
    """OCR Service - handles detection and recognition pipeline"""
    
    def __init__(self, det_model, rec_model):
        self.det_model = det_model
        self.rec_model = rec_model
    
    def process_image(self, image: np.ndarray) -> List[Dict]:
        """
        Process image through detection and recognition pipeline
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of OCR results with 'bbox', 'text', 'confidence'
        """
        logger.info("Starting OCR pipeline...")
        
        # Step 1: Run detection
        logger.info("Step 1: Running detection...")
        output = run_detection_on_image(
            self.det_model,
            image,
            resize_long=settings.DETECTION_RESIZE_LONG,
            thresh=settings.DETECTION_THRESH,
            box_thresh=settings.DETECTION_BOX_THRESH
        )
        
        # Step 2: Extract bounding boxes
        logger.info("Step 2: Extracting bounding boxes...")
        bboxes = extract_bboxes_from_output(
            output,
            image,
            conf_threshold=settings.DETECTION_THRESH,
            expand_ratio_w=settings.EXPAND_RATIO_W,
            expand_ratio_h=settings.EXPAND_RATIO_H,
            min_pad_h=settings.MIN_PAD_H,
            max_pad_h=settings.MAX_PAD_H
        )
        
        logger.info(f"Found {len(bboxes)} bounding boxes")
        
        # Step 3: Run recognition on each bbox
        logger.info("Step 3: Running recognition...")
        results = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2, conf = bbox
            
            # Run recognition
            text = run_recognition_on_bbox(self.rec_model, image, bbox)
            
            results.append({
                'bbox': [x1, y1, x2, y2],
                'text': text,
                'confidence': conf
            })
            
            logger.debug(f"  {i+1}/{len(bboxes)}: '{text}' (conf: {conf:.3f})")
        
        logger.info(f"OCR pipeline completed. Processed {len(results)} text boxes.")
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Visualize OCR results on image
        
        Args:
            image: Input image
            results: OCR results
            
        Returns:
            Image with drawn bounding boxes and text
        """
        return visualize_ocr_results(image, results)
    
    def extract_invoice_fields(self, results: List[Dict]) -> Dict[str, Optional[str]]:
        """
        Extract invoice fields from OCR results
        
        Args:
            results: OCR results from process_image
            
        Returns:
            Dictionary with supplier_name, total, currency
        """
        logger.info("Extracting invoice fields...")
        
        # Standardize data format
        standardized_data = []
        for item in results:
            text = str(item.get('text', '')).lower()
            raw_text = str(item.get('text', ''))
            conf = item.get('confidence', 0.0)
            
            bbox = item.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            standardized_data.append({
                'text': text,
                'text_raw': raw_text,
                'bbox': bbox,  # [x1, y1, x2, y2]
                'conf': conf
            })
        
        # Extract fields
        supplier_name = self._extract_supplier_name(standardized_data)
        total = self._extract_grand_total(standardized_data)
        currency = self._extract_currency(standardized_data, total)
        
        return {
            'supplier_name': supplier_name,
            'total': total,
            'currency': currency
        }
    
    def _extract_supplier_name(self, data: List[Dict]) -> Optional[str]:
        """Extract supplier name from OCR data"""
        # Look for supplier-related keywords
        supplier_keywords = ['supplier', 'vendor', 'from', 'company', 'seller', 'nhà cung cấp', 'công ty']
        
        for i, item in enumerate(data):
            text = item['text']
            
            # Check if this line contains supplier keywords
            if any(kw in text for kw in supplier_keywords):
                # Try to find the actual name in the same line or next line
                if len(item['text_raw'].split()) > 2:
                    # Name is in the same line
                    parts = item['text_raw'].split(':', 1)
                    if len(parts) > 1:
                        return parts[1].strip()
                    return item['text_raw']
                
                # Check next line
                if i + 1 < len(data):
                    next_item = data[i + 1]
                    # Check if next item is close vertically
                    if self._calculate_y_overlap(item['bbox'], next_item['bbox']) > 0.3:
                        return next_item['text_raw']
        
        # Fallback: Use text from top-left corner (usually has company name)
        if data:
            # Find items in top 20% of image
            max_y = max([d['bbox'][3] for d in data])
            top_items = [d for d in data if d['bbox'][1] < max_y * 0.2]
            
            if top_items:
                # Sort by y position (top to bottom)
                top_items.sort(key=lambda x: x['bbox'][1])
                # Return first substantial text (more than 3 characters)
                for item in top_items[:3]:
                    if len(item['text_raw']) > 3:
                        return item['text_raw']
        
        return None
    
    def _extract_grand_total(self, data: List[Dict]) -> Optional[str]:
        """Extract grand total from OCR data"""
        # Priority keywords
        priority_keywords = ['grand total', 'amount due', 'total due', 'amount to pay', 
                            'thanh toan', 'tong tien', 'cong tien', 'phai thu']
        generic_keywords = ['total', 'tổng', 'cộng']
        exclude_keywords = ['sub', 'net', 'tax', 'vat', 'trước thuế', 'discount', 'khuyến mãi', 'qty', 'sl']
        
        candidates = []
        
        for item in data:
            text = item['text']
            bbox = item['bbox']
            
            if any(ex in text for ex in exclude_keywords):
                continue
            
            score = 0
            if any(pk in text for pk in priority_keywords):
                score = 2
            elif any(gk in text for gk in generic_keywords):
                score = 1
            
            if score > 0:
                candidates.append({'item': item, 'score': score, 'bottom_y': bbox[3]})
        
        # Check candidates
        if candidates:
            # Sort by score (high to low) and bottom_y (low to high)
            candidates.sort(key=lambda x: (x['score'], x['bottom_y']), reverse=True)
            top_labels = candidates[:3]
            
            for cand in top_labels:
                label_item = cand['item']
                label_bbox = label_item['bbox']
                
                # Check 1: Number in same line (inline)
                inline_nums = re.findall(r'[\d.,]+', label_item['text_raw'])
                valid_inline = [n for n in inline_nums if len(re.sub(r'[^\d]', '', n)) >= 3]
                if valid_inline:
                    return self._clean_money_string(valid_inline[-1])
                
                # Check 2: Find value on the right
                possible_values = []
                for item in data:
                    if item == label_item:
                        continue
                    
                    val_text = item['text_raw']
                    val_bbox = item['bbox']
                    
                    # Must contain number
                    if not re.search(r'\d', val_text):
                        continue
                    if len(re.sub(r'[^\d]', '', val_text)) < 2:
                        continue
                    
                    # Must be on the right
                    if val_bbox[0] < label_bbox[0]:
                        continue
                    
                    # Check vertical overlap
                    overlap = self._calculate_y_overlap(label_bbox, val_bbox)
                    
                    if overlap > 0.3:
                        possible_values.append({
                            'text': val_text,
                            'overlap': overlap
                        })
                
                if possible_values:
                    possible_values.sort(key=lambda x: x['overlap'], reverse=True)
                    return self._clean_money_string(possible_values[0]['text'])
        
        # Fallback: Find number in bottom-right corner
        if not data:
            return None
        
        max_w = max([i['bbox'][2] for i in data])
        max_h = max([i['bbox'][3] for i in data])
        
        region_x = max_w * 0.4
        region_y = max_h * 0.55
        
        bottom_right_nums = []
        for item in data:
            bx = item['bbox']
            if bx[0] > region_x and bx[1] > region_y:
                txt = self._clean_money_string(item['text_raw'])
                if len(txt) >= 3:
                    bottom_right_nums.append({
                        'text': txt,
                        'y': bx[1],
                        'x': bx[0]
                    })
        
        if bottom_right_nums:
            bottom_right_nums.sort(key=lambda k: (k['y'], k['x']), reverse=True)
            return bottom_right_nums[0]['text']
        
        return None
    
    def _extract_currency(self, data: List[Dict], total: Optional[str]) -> Optional[str]:
        """Extract currency from OCR data"""
        currency_keywords = {
            'VND': ['vnd', 'vnđ', 'đ', 'dong', 'việt nam'],
            'USD': ['usd', '$', 'dollar'],
            'EUR': ['eur', '€', 'euro'],
            'THB': ['thb', '฿', 'baht']
        }
        
        # Check all text for currency symbols or keywords
        for item in data:
            text = item['text'].lower()
            for currency, keywords in currency_keywords.items():
                if any(kw in text for kw in keywords):
                    return currency
        
        # Check total string for currency symbols
        if total:
            for currency, keywords in currency_keywords.items():
                if any(kw in total.lower() for kw in keywords if len(kw) > 1):
                    return currency
        
        # Default to VND (common for Vietnamese invoices)
        return "VND"
    
    def _calculate_y_overlap(self, box1: List[int], box2: List[int]) -> float:
        """Calculate vertical overlap ratio between two boxes"""
        y1_a, y2_a = box1[1], box1[3]
        y1_b, y2_b = box2[1], box2[3]
        
        intersect_start = max(y1_a, y1_b)
        intersect_end = min(y2_a, y2_b)
        
        if intersect_end <= intersect_start:
            return 0.0
        
        overlap_height = intersect_end - intersect_start
        min_height = min(y2_a - y1_a, y2_b - y1_b)
        
        if min_height == 0:
            return 0.0
        
        return overlap_height / min_height
    
    def _clean_money_string(self, text: str) -> str:
        """Clean money string by removing non-numeric characters"""
        if not text:
            return ""
        cleaned = re.sub(r'[^\d.,]', '', text)
        cleaned = cleaned.rstrip('.,')
        return cleaned
