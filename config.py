# config.py
import os

# Đường dẫn model - cần điều chỉnh theo môi trường của bạn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn model detection
DET_DIR = os.path.join(BASE_DIR, "PP-OCRv5_server_det_infer")
# Hoặc nếu model ở cùng thư mục:
# DET_DIR = BASE_DIR

# Đường dẫn model recognition
REC_DIR = os.path.join(BASE_DIR, "inference_model")
# Hoặc nếu model ở cùng thư mục:
# REC_DIR = BASE_DIR

# Đường dẫn dictionary
DICT_PATH = os.path.join(REC_DIR, "en_dict.txt")
# Hoặc nếu dict ở cùng thư mục:
# DICT_PATH = os.path.join(BASE_DIR, "en_dict.txt")