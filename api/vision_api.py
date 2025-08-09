"""
Vision UI API接口文件
提供多种视觉分析功能的RESTful API接口
包括图像差异检测、图像融合、相似度计算、UI目标检测、文本识别、语义搜索等功能
"""

import hashlib
import os
from flask import jsonify
from flask import request
from flask import Blueprint
from flask import make_response
from service.image_diff import ImageDiff
from service.image_infer import get_ui_infer
from service.image_merge import Stitcher
from service.image_similar import HashSimilar
from service.image_text import get_image_text
from service.image_trace import image_trace
from service.image_utils import download_image, get_pop_v, save_base64_image


# 创建Flask蓝图，用于组织API路由
vision = Blueprint('vision', __name__, url_prefix='/vision')


@vision.route('/diff', methods=["POST"])
def vision_diff():
    """
    图像差异检测接口
    比较两张图片的差异，返回相似度分数
    
    请求参数:
    - image1: 第一张图片文件名
    - image2: 第二张图片文件名  
    - image_diff_name: 差异图片保存文件名
    
    返回: 相似度分数 (0-1.0)
    """
    data = {
        "code": 0,
        "data": ImageDiff().get_image_score(request.json['image1'], request.json['image2'],
                                            request.json['image_diff_name'])
    }
    return jsonify(data)


@vision.route('/merge', methods=["POST"])
def vision_merge():
    """
    图像融合接口
    将多张图片智能拼接成一张长图
    
    请求参数:
    - image_list: 图片文件名列表
    - name: 融合后图片保存文件名
    - without_padding: 是否去除填充 (可选)
    
    返回: 融合后图片文件名
    """
    data = {
        "code": 0,
        "data": Stitcher(request.json['image_list']).image_merge(
            request.json['name'],
            without_padding=request.json.get('without_padding')
        )
    }
    return jsonify(data)


@vision.route('/similar', methods=["POST"])
def vision_similar():
    """
    图像相似度计算接口
    计算两张图片的相似度分数
    
    请求参数:
    - image1: 第一张图片文件名
    - image2: 第二张图片文件名
    
    返回: 相似度分数 (0-1.0)
    """
    data = {
        "code": 0,
        "data": HashSimilar().get_hash_similar(request.json['image1'], request.json['image2'])
    }
    return jsonify(data)


@vision.route('/pop', methods=["POST"])
def vision_pop():
    """
    弹窗检测接口
    检测图片中是否存在弹窗
    
    请求参数:
    - image: 图片文件名
    
    返回: V通道均值，用于判断弹窗存在
    """
    data = {
        "code": 0,
        "data": get_pop_v(request.json['image'])
    }
    return jsonify(data)


@vision.route('/text', methods=["POST"])
def vision_text():
    """
    图像文本识别接口
    识别图片中的文字内容
    
    请求参数:
    - image: 图片文件名
    
    返回: 识别到的文本信息列表
    """
    data = {
        "code": 0,
        "data": get_image_text(request.json['image'])
    }
    resp = make_response(jsonify(data))
    resp.headers["Content-Type"] = "application/json;charset=utf-8"
    return resp


@vision.route('/ui-infer', methods=["POST"])
def vision_infer():
    """
    UI目标检测接口
    检测图片中的UI元素（图标、图片等）
    
    请求参数:
    - type: 图片类型 ('url' 或 'base64')
    - url: 图片URL (type为url时)
    - image: base64编码图片 (type为base64时)
    - cls_thresh: 分类阈值 (可选，默认0.5)
    
    返回: 检测到的UI元素列表
    """
    code = 0
    image_type = request.json.get('type', 'url')
    cls_thresh = request.json.get('cls_thresh', 0.5)
    
    # 根据图片类型处理输入
    if image_type == 'url':
        img_url = request.json['url']
        # 使用URL的MD5值作为文件名
        image_name = f'{hashlib.md5(img_url.encode(encoding="utf-8")).hexdigest()}.{img_url.split(".")[-1]}'
        image_path = download_image(img_url, image_name)
    elif image_type == 'base64':
        base64_image = request.json['image']
        # 使用base64内容的MD5值作为文件名
        image_name = f'{hashlib.md5(base64_image.encode(encoding="utf-8")).hexdigest()}.png'
        image_path = save_base64_image(base64_image, image_name)
    else:
        raise Exception(f'UI infer API don`t support this type: {image_type}')

    try:
        # 执行UI目标检测
        data = get_ui_infer(image_path, cls_thresh)
    finally:
        # 清理临时文件
        os.remove(image_path)

    result = {
        "code": code,
        "data": data
    }
    resp = make_response(jsonify(result))
    resp.headers["Content-Type"] = "application/json;charset=utf-8"
    return resp


@vision.route('/semantic-search', methods=["POST"])
def semantic_search():
    """
    语义搜索接口
    在目标图片中搜索与查询图片/文本语义相似的区域
    
    请求参数:
    - type: 图片类型 ('url' 或 'base64', 默认'url')
    - target_image: 查询图片URL或base64
    - source_image: 目标图片URL或base64
    - target_desc: 查询文本描述
    - image_alpha: 图像相似度权重 (0.0-1.0)
    - text_alpha: 文本相似度权重 (0.0-1.0)
    - top_k: 返回结果数量 (可选，默认1)
    - proposal_provider: 候选区域提供方式 ('ui-infer' 或 'patches', 默认'ui-infer')
    
    返回: 搜索结果列表，包含匹配区域和置信度
    """
    code = 0
    image_type = request.json.get('type', 'url')
    target_image = request.json.get('target_image')
    source_image = request.json.get('source_image')
    
    # 根据图片类型下载或保存图片
    if image_type == 'url':
        image_name = f'{hashlib.md5(target_image.encode(encoding="utf-8")).hexdigest()}.{target_image.split(".")[-1]}'
        target_image_path = download_image(target_image, image_name)
        image_name = f'{hashlib.md5(source_image.encode(encoding="utf-8")).hexdigest()}.{source_image.split(".")[-1]}'
        source_image_path = download_image(source_image, image_name)
    elif image_type == 'base64':
        image_name = f'{hashlib.md5(target_image.encode(encoding="utf-8")).hexdigest()}.png'
        target_image_path = save_base64_image(target_image, image_name)
        image_name = f'{hashlib.md5(source_image.encode(encoding="utf-8")).hexdigest()}.png'
        source_image_path = save_base64_image(source_image, image_name)
    else:
        raise Exception(f'Not supported type: {image_type}')
    
    try:
        # 构造目标图像信息
        target_image_info = {'img': target_image_path, 'desc': request.json.get('target_desc', '')}
        
        # 执行语义搜索
        top_k_ids, scores, infer_result, max_confidence = image_trace.search_image(
            target_image_info, source_image_path, request.json.get('top_k', 1), request.json.get('image_alpha'),
            request.json.get('text_alpha'), request.json.get('proposal_provider', 'ui-infer')
        )
        
        # 构造返回结果
        data = []
        for i in top_k_ids:
            data.append({
                'score': round(float(scores[i]), 2),
                'boxes': [int(k) for k in infer_result[i]['elem_det_region']]
            })
    finally:
        # 清理临时文件
        os.remove(target_image_path)
        os.remove(source_image_path)
    
    result = {
        "code": code,
        "data": {
            'max_confidence': max_confidence,
            'search_result': data
        }
    }
    resp = make_response(jsonify(result))
    resp.headers["Content-Type"] = "application/json;charset=utf-8"
    return resp
