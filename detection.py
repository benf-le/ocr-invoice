# detection.py
import os
import cv2
import numpy as np
import paddle
from paddle.inference import Config, create_predictor

class DBConfig:
    def __init__(self):
        self.thresh = 0.3
        self.box_thresh = 0.5
        self.max_candidates = 1000
        self.unclip_ratio = 1.6
        self.min_size = 3

def py_clipper_expand(box, distance):
    box = np.array(box, dtype=np.float32)
    center = np.mean(box, axis=0)
    vectors = box - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1
    unit_vectors = vectors / lengths
    expanded_box = box + unit_vectors * distance
    return expanded_box.astype(np.int32)

def unclip(box, unclip_ratio=1.6):
    poly = cv2.convexHull(box)
    area = cv2.contourArea(poly)
    length = cv2.arcLength(poly, True)
    if length <= 0: return None
    distance = area * unclip_ratio / length
    return py_clipper_expand(box, distance)

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def boxes_from_bitmap(pred, bitmap, dest_width, dest_height, config):
    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []

    for contour in contours[:config.max_candidates]:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))

        if len(points) < 4: continue
        score = box_score_fast(pred, points)
        if score < config.box_thresh: continue

        box = cv2.minAreaRect(points)
        box_points = cv2.boxPoints(box)
        if min(box[1]) < config.min_size: continue

        expanded_box = unclip(box_points, config.unclip_ratio)
        if expanded_box is None: continue

        rect = cv2.minAreaRect(expanded_box)
        final_box = cv2.boxPoints(rect)

        final_box[:, 0] = np.clip(final_box[:, 0], 0, dest_width)
        final_box[:, 1] = np.clip(final_box[:, 1], 0, dest_height)

        boxes.append(final_box.astype(np.int32))
        scores.append(score)

    return boxes, scores

def load_db_predictor(model_dir):
    try:
        params_file = os.path.join(model_dir, "inference.pdiparams")
        if not os.path.exists(params_file):
            print(f"❌ Không tìm thấy file params: {params_file}")
            return None

        model_file = os.path.join(model_dir, "inference.pdmodel")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "inference.json")

        if not os.path.exists(model_file):
            print(f"❌ Không tìm thấy file model tại: {model_dir}")
            return None

        config = Config(model_file, params_file)

        if paddle.device.is_compiled_with_cuda():
            config.enable_use_gpu(100, 0)
            print("⚡ Running on GPU")
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(4)
            print("🐢 Running on CPU")

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        predictor = create_predictor(config)
        print("✅ Load Detection Predictor thành công!")
        return predictor

    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo Predictor: {e}")
        return None

def preprocess_image(image_array, limit_side=960):
    """Nhận numpy array (RGB) thay vì đường dẫn file"""
    img = image_array.copy()
    h, w = img.shape[:2]

    ratio = 1.0
    if max(h, w) > limit_side:
        if h > w:
            ratio = limit_side / h
        else:
            ratio = limit_side / w

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)

    img_resized = cv2.resize(img, (resize_w, resize_h))

    img_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype('float32')
    img_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype('float32')

    img_normalized = (img_resized.astype('float32') / 255.0 - img_mean) / img_std

    img_tensor = img_normalized.transpose((2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, 0)

    return img, img_tensor, (ratio, resize_h, resize_w)

def run_detection(predictor, image_array):
    """Nhận numpy array và trả về list boxes"""
    original_img, img_tensor, (ratio, rh, rw) = preprocess_image(image_array, limit_side=960)
    
    if original_img is None:
        return []

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.copy_from_cpu(img_tensor)

    predictor.run()

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])
    output_data = output_tensor.copy_to_cpu()

    pred = output_data[0, 0, :, :]

    config = DBConfig()
    segmentation = pred > config.thresh

    boxes, scores = boxes_from_bitmap(pred, segmentation, rw, rh, config)

    final_boxes = []
    scale_x = original_img.shape[1] / rw
    scale_y = original_img.shape[0] / rh

    for box in boxes:
        box = box.astype(np.float32)
        box[:, 0] *= scale_x
        box[:, 1] *= scale_y
        final_boxes.append(box.astype(np.int32))

    return final_boxes