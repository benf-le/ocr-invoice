# recognition.py
import cv2
import numpy as np
import paddle
import os

def sorted_boxes(dt_boxes):
    num_boxes = len(dt_boxes)
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def get_char_dict(dict_path):
    if not os.path.exists(dict_path):
        print(f"⚠️ Không tìm thấy file dict tại '{dict_path}'")
        default_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        return ['blank'] + list(default_chars)

    try:
        chars = []
        with open(dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            char = line.strip('\n').strip('\r')
            chars.append(char)

        if ' ' not in chars: chars.append(' ')
        return ['blank'] + chars

    except Exception as e:
        print(f"❌ Lỗi đọc file dict: {e}")
        return ['blank']

def decode_output(output_tensor, char_dict):
    preds_idx = paddle.argmax(output_tensor, axis=2)
    preds_prob = paddle.max(output_tensor, axis=2)

    preds_idx = preds_idx.numpy()[0]
    preds_prob = preds_prob.numpy()[0]

    text = ""
    conf_list = []

    last_index = 0
    for i, index in enumerate(preds_idx):
        if index != last_index and index != 0 and index < len(char_dict):
            text += char_dict[index]
            conf_list.append(preds_prob[i])
        last_index = index

    avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0.0
    return text, avg_conf

def load_recognition_model(model_dir):
    try:
        model_file = f"{model_dir}/inference.json"
        params_file = f"{model_dir}/inference.pdiparams"

        if os.path.exists(model_file) and os.path.exists(params_file):
            model = paddle.jit.load(model_dir + "/inference")
            print("✅ [RECOGNITION] Load model thành công!")
            return model
        else:
            print("❌ [RECOGNITION] Không tìm thấy file model")
            return None
    except Exception as e:
        print(f"❌ [RECOGNITION] Lỗi: {e}")
        return None

def recognize_text(rec_model, image, polygons, dict_path):
    results = []
    h_img, w_img = image.shape[:2]
    
    char_dict = get_char_dict(dict_path)
    sorted_polys = sorted_boxes(polygons)

    for i, poly in enumerate(sorted_polys):
        x, y, w, h = cv2.boundingRect(poly)

        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + 2*pad)
        h = min(h_img - y, h + 2*pad)

        cropped_img = image[y:y+h, x:x+w]

        if cropped_img.size == 0 or w < 5 or h < 5:
            continue

        rec_h = 48
        rec_w = 320
        resized_img = cv2.resize(cropped_img, (rec_w, rec_h))

        norm_img = resized_img.astype(np.float32) / 255.0
        norm_img = (norm_img - 0.5) / 0.5
        norm_img = norm_img.transpose((2, 0, 1))
        input_tensor = np.expand_dims(norm_img, axis=0)

        try:
            x_tensor = paddle.to_tensor(input_tensor)

            with paddle.no_grad():
                output = rec_model(x_tensor)

            text, conf = decode_output(output, char_dict)

            if conf > 0.5:
                results.append({
                    'poly': poly,
                    'text': text,
                    'conf': conf,
                    'rect': (x, y, w, h)
                })

        except Exception as e:
            print(f"  ❌ Lỗi tại box {i}: {e}")
            continue

    return results