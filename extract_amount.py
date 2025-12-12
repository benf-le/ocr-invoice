# extract_amount.py
import re

def calculate_y_overlap(box1, box2):
    y1_a, y2_a = box1[1], box1[3]
    y1_b, y2_b = box2[1], box2[3]

    intersect_start = max(y1_a, y1_b)
    intersect_end = min(y2_a, y2_b)

    if intersect_end <= intersect_start: return 0.0

    overlap_height = intersect_end - intersect_start
    min_height = min(y2_a - y1_a, y2_b - y1_b)

    if min_height == 0: return 0.0
    return overlap_height / min_height

def clean_money_string(text):
    if not text: return ""
    cleaned = re.sub(r'[^\d.,]', '', text)
    cleaned = cleaned.rstrip('.,')
    return cleaned

def extract_grand_total(ocr_results):
    standardized_data = []

    for item in ocr_results:
        text = str(item.get('text', '')).lower()
        raw_text = str(item.get('text', ''))
        conf = item.get('conf', 0.0)

        if 'rect' in item:
            x, y, w, h = item['rect']
            bbox_xyxy = [x, y, x + w, y + h]
        elif 'bbox' in item:
            box = item['bbox']
            if len(box) == 4:
                bbox_xyxy = box
            else:
                continue
        else:
            continue

        standardized_data.append({
            'text': text,
            'text_raw': raw_text,
            'bbox': bbox_xyxy,
            'conf': conf
        })

    priority_keywords = ['grand total', 'amount due', 'total due', 'amount to pay', 'thanh toan', 'tong tien', 'cong tien', 'phai thu']
    generic_keywords = ['total', 'tổng', 'cộng']
    exclude_keywords = ['sub', 'net', 'tax', 'vat', 'trước thuế', 'discount', 'khuyến mãi', 'qty', 'sl']

    candidates = []

    for item in standardized_data:
        text = item['text']
        bbox = item['bbox']

        if any(ex in text for ex in exclude_keywords): continue

        score = 0
        if any(pk in text for pk in priority_keywords): score = 2
        elif any(gk in text for gk in generic_keywords): score = 1

        if score > 0:
            candidates.append({'item': item, 'score': score, 'bottom_y': bbox[3]})

    if candidates:
        candidates.sort(key=lambda x: (x['score'], x['bottom_y']), reverse=True)
        top_labels = candidates[:3]

        for cand in top_labels:
            label_item = cand['item']
            label_bbox = label_item['bbox']

            inline_nums = re.findall(r'[\d.,]+', label_item['text_raw'])
            valid_inline = [n for n in inline_nums if len(re.sub(r'[^\d]', '', n)) >= 3]
            if valid_inline:
                return clean_money_string(valid_inline[-1])

            possible_values = []
            for item in standardized_data:
                if item == label_item: continue

                val_text = item['text_raw']
                val_bbox = item['bbox']

                if not re.search(r'\d', val_text): continue
                if len(re.sub(r'[^\d]', '', val_text)) < 2: continue

                if val_bbox[0] < label_bbox[0]: continue

                overlap = calculate_y_overlap(label_bbox, val_bbox)

                if overlap > 0.3:
                    possible_values.append({
                        'text': val_text,
                        'overlap': overlap
                    })

            if possible_values:
                possible_values.sort(key=lambda x: x['overlap'], reverse=True)
                return clean_money_string(possible_values[0]['text'])

    # if not standardized_data: return "0"

    # max_w = max([i['bbox'][2] for i in standardized_data])
    # max_h = max([i['bbox'][3] for i in standardized_data])

    # region_x = max_w * 0.4
    # region_y = max_h * 0.55

    # bottom_right_nums = []
    # for item in standardized_data:
    #     bx = item['bbox']
    #     if bx[0] > region_x and bx[1] > region_y:
    #         txt = clean_money_string(item['text_raw'])
    #         if len(txt) >= 3:
    #             bottom_right_nums.append({
    #                 'text': txt,
    #                 'y': bx[1],
    #                 'x': bx[0]
    #             })

    # if bottom_right_nums:
    #     bottom_right_nums.sort(key=lambda k: (k['y'], k['x']), reverse=True)
    #     return bottom_right_nums[0]['text']

    return "Không tìm thấy"