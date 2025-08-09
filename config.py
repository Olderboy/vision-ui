"""
Vision UI 配置文件
定义系统运行所需的各种配置参数
"""

# CPU线程数配置
# 设置为CPU核心数可获得最佳性能
OP_NUM_THREADS = 4

# 模型文件路径配置
# UI目标检测模型路径（YOLOX ONNX格式）
IMAGE_INFER_MODEL_PATH = "capture/local_models/ui_det_v2.onnx"

# CLIP语义搜索模型路径（ONNX格式）
CLIP_MODEL_PATH = "capture/local_models/clip_vit32_feat.onnx"

# 临时文件目录
# 用于存储下载的图片和临时处理文件
IMAGE_TEMP_DIR = 'capture/temp'

