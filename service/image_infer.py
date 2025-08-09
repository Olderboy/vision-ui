"""
UI目标检测服务
基于YOLOX模型的UI元素检测，能够识别图标、图片等UI组件
适用于移动应用和Web界面的UI元素定位
"""

import os.path
import re
import cv2
import numpy as np
import onnxruntime
import time
from config import IMAGE_INFER_MODEL_PATH, OP_NUM_THREADS
from service.image_utils import yolox_preprocess, yolox_postprocess, multiclass_nms, img_show


class ImageInfer(object):
    """
    UI目标检测器
    使用YOLOX模型进行UI元素检测
    """
    def __init__(self, model_path):
        """
        初始化检测器
        :param model_path: ONNX模型文件路径
        """
        self.UI_CLASSES = ("bg", "icon", "pic")  # UI类别：背景、图标、图片
        self.input_shape = [640, 640]  # 模型输入尺寸
        self.cls_thresh = 0.5  # 分类阈值
        self.nms_thresh = 0.2  # NMS阈值
        self.model_path = model_path
        
        # 配置ONNX运行时
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = OP_NUM_THREADS
        self.model_session = onnxruntime.InferenceSession(self.model_path, sess_options=so,
                                                          providers=['CPUExecutionProvider'])

    def ui_infer(self, image):
        """
        执行UI目标检测推理
        :param image: 输入图像路径或numpy数组
        :return: 检测结果数组 [x1, y1, x2, y2, score, class_id]
        """
        # 读取图像
        origin_img = cv2.imread(image) if isinstance(image, str) else image
        
        # 图像预处理
        img, ratio = yolox_preprocess(origin_img, self.input_shape)
        
        # 模型推理
        ort_inputs = {self.model_session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.model_session.run(None, ort_inputs)
        
        # 后处理
        predictions = yolox_postprocess(output[0], self.input_shape)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        
        # 转换为xyxy格式
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.  # x1 = center_x - width/2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.  # y1 = center_y - height/2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.  # x2 = center_x + width/2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.  # y2 = center_y + height/2
        boxes_xyxy /= ratio  # 还原到原始图像尺寸
        
        # 执行NMS
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nms_thresh, score_thr=self.cls_thresh)
        
        if dets is not None:
            # 兼容不同版本模型返回结果中UI classes index起始位置
            offset = 0
            match_obj = re.match(r'.*o(\d+)\.onnx$', self.model_path)
            if match_obj:
                offset = int(match_obj.group(1))
            dets[:, 5] += offset
        return dets

    def show_infer(self, dets, origin_img, infer_result_path):
        """
        可视化检测结果
        :param dets: 检测结果
        :param origin_img: 原始图像
        :param infer_result_path: 结果保存路径
        """
        if dets is not None:
            boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = img_show(origin_img, boxes, scores, cls_inds, conf=self.cls_thresh,
                                  class_names=self.UI_CLASSES)
        cv2.imwrite(infer_result_path, origin_img)


# 创建全局检测器实例
image_infer = ImageInfer(IMAGE_INFER_MODEL_PATH)


def get_ui_infer(image, cls_thresh):
    """
    UI目标检测接口函数
    :param image: 输入图像路径
    :param cls_thresh: 分类阈值
    :return: 检测结果列表，包含元素类型、位置和置信度
    """
    data = []
    # 设置分类阈值
    image_infer.cls_thresh = cls_thresh if isinstance(cls_thresh, float) else image_infer.cls_thresh
    
    # 执行检测
    dets = image_infer.ui_infer(image)
    
    if isinstance(dets, np.ndarray):
        boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        for i in range(len(boxes)):
            box = boxes[i]
            box[box < 0] = 0  # 确保坐标非负
            box = box.tolist() if isinstance(box, (np.ndarray,)) else box
            elem_type = image_infer.UI_CLASSES[int(cls_inds[i])]
            score = scores[i]
            
            # 构造返回结果
            data.append(
                {
                    "elem_det_type": "image" if elem_type == 'pic' else elem_type,  # 元素类型
                    "elem_det_region": box,  # 检测区域 [x1, y1, x2, y2]
                    "probability": score  # 置信度
                }
            )
    return data


if __name__ == '__main__':
    """
    调试代码
    用于测试UI目标检测功能
    """
    image_path = "./capture/image_1.png"
    infer_result_path = "./capture/local_images"
    assert os.path.exists(image_path)
    assert os.path.exists(IMAGE_INFER_MODEL_PATH)
    if not os.path.exists(infer_result_path):
        os.mkdir(infer_result_path)
    
    # 执行检测并计时
    t1 = time.time()
    dets = image_infer.ui_infer(image_path)
    print(f"Infer time: {round(time.time()-t1, 3)}s")
    
    # 保存可视化结果
    infer_result_name = f"infer_result.png"
    image_infer.show_infer(dets, cv2.imread(image_path), os.path.join(infer_result_path, infer_result_name))
    print(f"Result saved {infer_result_path}/{infer_result_name}")
