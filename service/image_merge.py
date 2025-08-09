"""
图像融合服务
基于模板匹配的智能图像拼接算法
能够将多张截图智能拼接成一张长图，适用于长页面截图拼接
"""

import numpy
import cv2


class Stitcher(object):
    """
    图像拼接器
    实现基于模板匹配的智能图像拼接
    """
    def __init__(self, pictures):
        """
        初始化拼接器
        :param pictures: 图片文件名列表
        """
        self.img_list = pictures
        self.w = 80  # 填充宽度

    @staticmethod
    def add_padding(img, w):
        """
        为图像添加右侧填充
        :param img: 输入图像
        :param w: 填充宽度
        :return: 填充后的图像
        """
        h, _, _ = img.shape
        # 创建白色填充区域
        padding = numpy.zeros((h, w, 3), numpy.uint8) + 210
        return numpy.hstack((img, padding))

    @staticmethod
    def merge_with_param(img1, img2, w, roi_scale, tail_scale, index):
        """
        使用指定参数进行图像拼接
        :param img1: 第一张图像
        :param img2: 第二张图像
        :param w: 填充宽度
        :param roi_scale: ROI区域比例
        :param tail_scale: 尾部裁剪比例
        :param index: 图像索引
        :return: 拼接结果和匹配分数
        """
        img1 = img1.copy()
        img2 = img2.copy()
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        
        # 裁剪第一张图像的尾部
        img1 = img1[0:h1 - int(tail_scale * h2), :, ]
        
        # 如果宽度不同，添加填充
        if w1 == w2:
            img1 = Stitcher.add_padding(img1, w)
        
        h1, w1, _ = img1.shape
        _img1 = img1[:, :w1 - w, :]  # 去除填充区域
        h1, w1, _ = _img1.shape
        
        # 检查宽度是否匹配
        if w1 != w2:
            raise Exception("Image merge: different width")
        
        # 计算ROI区域
        roi_y = int(h2 * roi_scale)
        roi = _img1[(h1 - roi_y):h1, :, :]
        
        # 模板匹配
        res = cv2.matchTemplate(img2, roi, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # 处理低方差区域
        std = numpy.std(roi)
        if std < 10:
            max_val = 0.95
        
        # 计算拼接点
        cut_point = (max_loc[0], max_loc[1] + int(roi_scale * h2))
        img2 = Stitcher.add_padding(img2, w)
        
        # 执行拼接
        img = numpy.vstack((img1, img2[cut_point[1]:h2, :, :]))
        
        # 添加索引标记
        cv2.putText(img, "-" + str(index), (w2, h1), cv2.FONT_ITALIC, 1.5, (123, 189, 60), 5)
        return img, max_val

    @staticmethod
    def stack_image(img1, img2, w, index):
        """
        简单堆叠拼接（无模板匹配）
        :param img1: 第一张图像
        :param img2: 第二张图像
        :param w: 填充宽度
        :param index: 图像索引
        :return: 拼接结果
        """
        img1 = img1.copy()
        img2 = img2.copy()
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        
        # 如果宽度不同，添加填充
        if w1 == w2:
            img1 = Stitcher.add_padding(img1, w)
        
        h1, w1, _ = img1.shape
        _img1 = img1[:, :w1 - w, :]
        h1, w1, _ = _img1.shape
        
        # 检查宽度是否匹配
        if w1 != w2:
            raise Exception("Image width different")
        
        img2 = Stitcher.add_padding(img2, w)
        img = numpy.vstack((img1, img2))
        
        # 添加索引标记
        cv2.putText(img, "-" + str(index), (w2, h1), cv2.FONT_ITALIC, 1.5, (123, 189, 60), 5)
        return img

    @staticmethod
    def img_merge(img1, img2, index, w, merge=True):
        """
        图像拼接主函数
        :param img1: 第一张图像
        :param img2: 第二张图像
        :param index: 图像索引
        :param w: 填充宽度
        :param merge: 是否使用智能拼接
        :return: 拼接结果
        """
        match = 0.98  # 高匹配阈值
        min_match = 0.92  # 最小匹配阈值
        
        # 定义多组ROI和尾部裁剪参数
        scale_list = [
            {
                "roi_scale": 0.12,  # ROI区域占图像高度的12%
                "tail_scale": 0.18  # 尾部裁剪占图像高度的18%
            }, {
                "roi_scale": 0.08,
                "tail_scale": 0.32
            }, {
                "roi_scale": 0.08,
                "tail_scale": 0.08
            }, {
                "roi_scale": 0.05,
                "tail_scale": 0.2
            }, {
                "roi_scale": 0.1,
                "tail_scale": 0.4
            }, {
                "roi_scale": 0.08,
                "tail_scale": 0.15
            }
        ]
        
        if merge:
            img_list = []
            score_list = []
            
            # 尝试不同的参数组合
            for scale in scale_list:
                img, score = Stitcher.merge_with_param(img1, img2, w, scale["roi_scale"], scale["tail_scale"], index)
                img_list.append(img)
                score_list.append(score)
                
                # 如果匹配分数很高，直接使用
                if score > match:
                    break
                
                # 如果所有参数都尝试完且分数都不够高，使用简单堆叠
                if scale == scale_list[-1] and max(score_list) < min_match:
                    img = Stitcher.stack_image(img1, img2, w, index)
                else:
                    # 选择分数最高的结果
                    img = img_list[score_list.index(max(score_list))]
        else:
            # 直接使用简单堆叠
            img = Stitcher.stack_image(img1, img2, w, index)
        return img

    def image_merge(self, name, without_padding, merge=True):
        """
        批量图像拼接
        :param name: 输出图像文件名
        :param merge: 是否使用智能拼接
        :param without_padding: 是否去除填充
        :return: 拼接后图像文件名
        """
        img_list = []
        w = 0 if without_padding else self.w  # 根据参数决定是否使用填充
        
        # 构造完整的图像路径
        for img in self.img_list:
            img_list.append('capture/'+img)
        name = 'capture/'+name
        
        # 如果只有一张图像，直接保存
        if len(img_list) < 2:
            cv2.imwrite(name, cv2.imread(img_list[0]))
        else:
            # 逐张拼接图像
            img1 = img_list[0]
            for img in img_list[1:]:
                index = img_list.index(img)
                if img_list.index(img) == 1:
                    # 第一张拼接
                    img1 = self.img_merge(cv2.imread(img1), cv2.imread(img), index, w, merge)
                else:
                    # 后续拼接
                    img1 = self.img_merge(img1, cv2.imread(img), index, w, merge)
            img_merge = img1
            cv2.imwrite(name, img_merge)
        return name
