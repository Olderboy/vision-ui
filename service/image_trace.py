"""
图像追踪服务
基于CLIP模型的语义图像搜索和目标追踪
支持图像和文本的语义匹配，适用于智能UI元素定位
"""

import time
import cv2
import os
import numpy
import torch
import clip
import numpy as np
from PIL import Image
from scipy import spatial
from config import CLIP_MODEL_PATH, OP_NUM_THREADS
from service.image_infer import get_ui_infer
from dbnet_crnn.image_text import ImageText
from service.image_utils import get_roi_image, img_show, get_image_patches, proposal_fine_tune, get_infer_area
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import onnxruntime


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def cosine_similar(l1, l2):
    """
    计算两个向量的余弦相似度
    :param l1: 向量1
    :param l2: 向量2
    :return: 余弦相似度
    """
    return 1 - spatial.distance.cosine(l1, l2)


def _convert_image_to_rgb(image):
    """
    将图像转换为RGB格式
    :param image: 输入图像
    :return: RGB图像
    """
    return image.convert("RGB")


def target_roi_text_diff_rate(target_img, source_img, proposals, tp):
    """
    计算目标区域与源图像区域的文本差异率
    :param target_img: 目标图像
    :param source_img: 源图像
    :param proposals: 候选区域列表
    :param tp: 排序后的索引
    :return: 文本差异率
    """
    image_text = ImageText()
    count = 0
    
    # 计算目标图像的文本
    target_l = target_img.shape[0] if target_img.shape[0] > target_img.shape[1] else target_img.shape[1]
    target_text = image_text.get_text(target_img, target_l)
    
    # 提取源图像中对应区域的文本
    x1, y1, x2, y2 = list(map(int, proposals[tp[-1]]['elem_det_region']))
    roi = get_roi_image(source_img, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    roi_l = roi.shape[0] if roi.shape[0] > roi.shape[1] else roi.shape[1]
    source_text = image_text.get_text(roi, roi_l)
    
    # 计算文本匹配数量
    for t in target_text:
        for s in source_text:
            if t['text'] in s['text'] or s['text'] in t['text']:
                count = count + 1
                break
    
    # 计算匹配率
    rate = count / len(target_text) if len(target_text) > 0 else 0
    return rate


def filter_patches(target_image, patches):
    """
    过滤候选区域，保留与目标图像长宽比相似的区域
    :param target_image: 目标图像
    :param patches: 候选区域列表
    :return: 过滤后的候选区域
    """
    result = []
    target_h, target_w, _ = target_image.shape
    target_ratio = target_w / target_h
    
    for patch in patches:
        x0, y0, x1, y1 = patch['elem_det_region']
        patch_w = x1 - x0
        patch_h = y1 - y0
        patch_ratio = patch_w / patch_h
        
        # 如果长宽比差异小于0.5，保留该区域
        if abs(patch_ratio - target_ratio) < 0.5:
            result.append(patch)
    return result


def get_proposals(target_image, source_image_path, provider="ui-infer"):
    """
    获取候选区域
    :param target_image: 目标图像
    :param source_image_path: 源图像路径
    :param provider: 候选区域提供方式 ('ui-infer' 或 'patches')
    :return: 候选区域列表
    """
    # 使用UI推理模型获取候选区域
    if provider == 'ui-infer':
        image_infer_result = get_ui_infer(source_image_path, 0.01)
    # 使用滑动窗口生成候选区域
    else:
        h, w, _ = target_image.shape
        # 定义不同分辨率的滑动窗口参数
        resolution_map = {'M': [0.6, 0.6], 'N': [0.5, 0.5]}
        image_infer_result = []
        
        for k in resolution_map.keys():
            resolution = resolution_map[k]
            source_img = cv2.imread(source_image_path)
            _h, _w, _ = source_img.shape
            
            # 如果目标图像比源图像大，进行缩放
            if w >= _w:
                ratio = _w / w
                target_image = cv2.resize(target_image, (0, 0), fx=ratio, fy=ratio)
                h, w, _ = target_image.shape
            
            # 根据图像尺寸比例调整分辨率
            resolution[0] = round(resolution[0]/2, 1) if _w / w < 6 else resolution[0]
            resolution[1] = round(resolution[1]/2, 1) if _h / h < 6 else resolution[1]
            
            # 生成候选区域
            image_infer_result.extend(get_image_patches(source_img, w, h, resolution[0], resolution[1]))
        
        # 过滤候选区域
        image_infer_result = filter_patches(target_image, image_infer_result)
    
    return image_infer_result


class ImageTrace(object):
    """
    图像追踪器
    基于CLIP模型实现语义图像搜索
    """
    def __init__(self):
        """
        初始化图像追踪器
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}.\nStart loading model")
        self.n_px = 224  # CLIP模型输入尺寸
        self.template_target_image = np.zeros([100, 100, 3], dtype=np.uint8) + 100
        
        # 设置图像预处理流程
        self.preprocess = self._get_preprocess()
        
        # 配置ONNX运行时
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = OP_NUM_THREADS
        cuda_provider = 'CUDAExecutionProvider'
        provider = cuda_provider if cuda_provider in onnxruntime.get_available_providers() and self.device == 'cuda' \
            else 'CPUExecutionProvider'
        print(f"ORT provider: {provider}")
        
        # 加载CLIP模型
        self.ort_sess = onnxruntime.InferenceSession(CLIP_MODEL_PATH, sess_options=so,
                                                     providers=[provider])
        print("Finish loading")

    def _get_preprocess(self):
        """
        获取CLIP模型的图像预处理流程
        :return: 预处理函数
        """
        return Compose([
            Resize(self.n_px, interpolation=BICUBIC),
            CenterCrop(self.n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def search_image(self, target_image_info: dict, source_image_path, top_k, image_alpha, text_alpha, provider):
        """
        执行语义图像搜索
        :param target_image_info: 目标图像信息 {'img': 图像路径, 'desc': 文本描述}
        :param source_image_path: 源图像路径
        :param top_k: 返回结果数量
        :param image_alpha: 图像相似度权重
        :param text_alpha: 文本相似度权重
        :param provider: 候选区域提供方式
        :return: 搜索结果
        """
        roi_list = []
        img_text_score = []
        
        # 读取目标图像和源图像
        target_image = target_image_info.get('img', self.template_target_image)
        target_image = cv2.imread(target_image) if isinstance(target_image, str) else target_image
        target_image_desc = target_image_info.get('desc', '')
        source_image = cv2.imread(source_image_path)
        
        # 获取候选区域
        proposals = get_proposals(target_image, source_image_path, provider=provider)
        
        # 对文本描述进行编码
        text = clip.tokenize([target_image_desc])
        
        # 提取候选区域图像
        for roi in proposals:
            x1, y1, x2, y2 = list(map(int, roi['elem_det_region']))
            roi = get_roi_image(source_image, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_list.append(self.preprocess(img_pil))
        
        # 预处理目标图像
        img_pil = Image.fromarray(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
        target_image_input = self.preprocess(img_pil).unsqueeze(0).clone().detach()
        source_image_input = torch.tensor(np.stack(roi_list))
        
        # 使用CLIP模型进行推理
        _, logits_per_text, source_image_features, = self.ort_sess.run(
            ["LOGITS_PER_IMAGE", "LOGITS_PER_TEXT", "onnx::Mul_3493"],
            {"IMAGE": source_image_input.numpy(), "TEXT": text.numpy()}
        )
        
        # 计算文本相似度概率
        probs = torch.from_numpy(logits_per_text).softmax(dim=-1).cpu().numpy()
        
        # 计算图像相似度特征
        if image_alpha != 0:
            target_image_features = self.ort_sess.run(
                ["onnx::Mul_3493"], {"IMAGE": target_image_input.numpy(), "TEXT": text.numpy()}
            )
        else:
            target_image_features = numpy.zeros([len(target_image_input)], dtype=np.uint8)
        
        # 计算综合相似度分数
        for i, source_image_feature in enumerate(source_image_features):
            score = cosine_similar(target_image_features[0][0], source_image_feature) if image_alpha != 0 else 0
            img_text_score.append(score * image_alpha + probs[0][i] * text_alpha)
        
        # 计算最大置信度
        max_confidence = round(np.max(img_text_score) / (image_alpha + text_alpha), 3)
        
        # 获取目标图像的UI检测结果
        target_image_infer = get_ui_infer(target_image, 0.1)
        target_area_rate = get_infer_area(target_image_infer) / (target_image.shape[0] * target_image.shape[1])
        
        # 归一化分数
        score_norm = (img_text_score - np.min(img_text_score)) / (np.max(img_text_score) - np.min(img_text_score))
        top_k_ids = np.argsort(score_norm)[-top_k:]
        
        # 微调候选区域
        proposal_fine_tune(score_norm, proposals, 0.9)
        
        # 如果目标区域较小且使用patches方式，考虑文本差异
        if target_area_rate < 0.1 and provider == 'patches':
            text_rate = target_roi_text_diff_rate(target_image, source_image, proposals, np.argsort(score_norm)[:])
            max_confidence = max_confidence - 0.08*(1-text_rate)
        
        return top_k_ids, score_norm, proposals, max_confidence

    def get_trace_result(self, target_image_info, source_image_path, top_k=3, image_alpha=1.0,
                         text_alpha=0.6, proposal_provider='ui-infer'):
        """
        获取追踪结果并可视化
        :param target_image_info: 目标图像信息
        :param source_image_path: 源图像路径
        :param top_k: 返回结果数量
        :param image_alpha: 图像相似度权重
        :param text_alpha: 文本相似度权重
        :param proposal_provider: 候选区域提供方式
        :return: 可视化结果图像
        """
        # 执行搜索
        top_k_ids, scores, infer_result, max_confidence = self.search_image(
            target_image_info, source_image_path, top_k, image_alpha, text_alpha, proposal_provider
        )
        print(f"Max confidence:{max_confidence}")
        
        # 构造可视化数据
        cls_ids = np.zeros(len(top_k_ids), dtype=int)
        boxes = [infer_result[i]['elem_det_region'] for i in top_k_ids]
        scores = [float(scores[i])*max_confidence for i in top_k_ids]
        
        # 生成可视化结果
        image_show = img_show(cv2.imread(source_image_path), boxes, scores, cls_ids, conf=0.5, class_names=['T'])
        return image_show

    def video_target_track(self, video_path, target_image_info, work_path):
        """
        视频目标追踪
        :param video_path: 视频文件路径
        :param target_image_info: 目标图像信息
        :param work_path: 工作目录
        """
        video_cap = cv2.VideoCapture(video_path)
        _, im = video_cap.read()
        
        # 设置视频输出
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        im_save_path = os.path.join(work_path, 'im_temp.png')
        video_out_path = os.path.join(work_path, 'video_out.mp4')
        out = cv2.VideoWriter(video_out_path, fourcc, 20, (im.shape[1], im.shape[0]))
        
        i = 0
        while 1:
            i = i + 1
            if i % 2 == 0:  # 每隔一帧处理一次
                continue
            print(f"video parsing {i}")
            ret, im = video_cap.read()
            if ret:
                # 保存当前帧
                cv2.imwrite(im_save_path, im)
                # 执行追踪
                trace_result = self.get_trace_result(target_image_info, im_save_path, top_k=1)
                out.write(trace_result)
            else:
                print("finish.")
                out.release()
                break


# 创建全局追踪器实例
image_trace = ImageTrace()


def trace_target_video():
    """
    视频追踪示例函数
    """
    target_image_info = {
        'path': "./capture/local_images/img_play_icon.png",
        'desc': "picture with play button"
    }
    target_image_info['img'] = cv2.imread(target_image_info['path'])
    video_path = "./capture/local_images/video.mp4"
    work_path = './capture/local_images'
    image_trace.video_target_track(video_path, target_image_info, work_path)


def search_target_image():
    """
    目标图像搜索示例函数
    """
    # 图像目标系数
    image_alpha = 0.0
    # 文本描述系数
    text_alpha = 1.0
    # 最大匹配目标数量
    top_k = 1
    
    # 调试用，构造目标图像
    target_img = np.zeros([100, 100, 3], dtype=np.uint8)+255
    cv2.putText(target_img, 'Q', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), thickness=3)
    # target_img = cv2.imread("./capture/local_images/mario.png")
    
    # 目标语言描述
    desc = "mario"
    target_image_info = {'img': target_img, 'desc': desc}
    source_image_path = "./capture/image_2.png"
    trace_result_path = "./capture/local_images/"
    
    if not os.path.exists(trace_result_path):
        os.mkdir(trace_result_path)
    
    # 查找目标
    t1 = time.time()
    image_trace_show = image_trace.get_trace_result(target_image_info, source_image_path, top_k=top_k,
                                                    image_alpha=image_alpha, text_alpha=text_alpha,
                                                    proposal_provider='ui-infer')
    print(f"Infer time:{round(time.time() - t1, 3)} s", )
    cv2.imwrite(trace_result_path+'trace_result.png', image_trace_show)
    print(f"Result saved {trace_result_path}")


if __name__ == '__main__':
    assert os.path.exists(CLIP_MODEL_PATH)
    search_target_image()
