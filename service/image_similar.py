"""
图像相似度计算服务
基于感知哈希和注意力机制的图像相似度计算
提供多种相似度计算方法，适用于图像内容比较
"""

import cv2
import numpy


class HashSimilar(object):
    """
    图像相似度计算器
    实现基于感知哈希和注意力机制的相似度计算
    """
    
    @staticmethod
    def perception_hash(img_gray, precision=64):
        """
        计算图像的感知哈希值
        基于图像的平均像素值生成二进制哈希码
        :param img_gray: 灰度图像
        :param precision: 哈希精度（图像缩放尺寸）
        :return: 感知哈希值列表
        """
        # 将图像缩放到指定精度
        img_scale = cv2.resize(img_gray, (precision, precision))
        img_list = img_scale.flatten()
        # 计算平均像素值
        avg = sum(img_list)*1./len(img_list)
        # 生成二进制哈希码
        avg_list = ['0' if i < avg else '1' for i in img_list]
        # 将二进制码转换为整数列表
        return [int(''.join(avg_list[x:x+4]), 2) for x in range(0, precision*precision)]

    @staticmethod
    def get_image(img_file):
        """
        读取并预处理图像
        :param img_file: 图像文件路径
        :return: 预处理后的灰度图像
        """
        img = cv2.imread(img_file)
        h, w, _ = img.shape
        # 去除顶部区域（通常包含状态栏等）
        img = img[int(w * 0.078):, :, :]
        # 转换为灰度图
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    @staticmethod
    def get_image_list(img_gray):
        """
        将图像分割成多个水平条带
        :param img_gray: 灰度图像
        :return: 图像条带列表
        """
        i = 0
        img_list = []
        h, w = img_gray.shape
        stride = int(w*0.05)  # 条带高度为图像宽度的5%
        
        # 按条带分割图像
        while i < h:
            img_list.append(img_gray[i:i + stride, :])
            i = i + stride
        return img_list

    @staticmethod
    def hamming_dist(s1, s2):
        """
        计算两个序列的汉明距离
        :param s1: 序列1
        :param s2: 序列2
        :return: 汉明距离
        """
        assert len(s1) == len(s2)
        return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

    @staticmethod
    def get_attention(img1, img2):
        """
        计算图像间的注意力分数
        通过模板匹配计算每个条带的相似度
        :param img1: 图像A
        :param img2: 图像B
        :return: 注意力分数列表
        """
        img1_list = HashSimilar.get_image_list(img1)
        img2_list = HashSimilar.get_image_list(img2)
        l = min(len(img1_list), len(img2_list))
        score_list = []
        
        # 对每个条带计算相似度
        for i in range(0, l, 1):
            # 计算搜索范围
            start = i - 5 if i - 5 >= 0 else 0
            # 将多个条带堆叠作为搜索区域
            img_stack = numpy.vstack(img2_list[start:i+5])
            # 使用模板匹配计算相似度
            res = cv2.matchTemplate(img1_list[i], img_stack, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            score_list.append(max_val)
        return score_list

    @staticmethod
    def get_attention_similar(image1, image2) -> float:
        """
        计算图像内容相似度
        基于注意力机制和统计特征的相似度计算
        :param image1: 图像A文件路径
        :param image2: 图像B文件路径
        :return: 相似度分数
            - 0.2: 图像来自不同的界面
            - 0.8: 图像有增量内容变化
            - 1.0: 图像A与图像B相似
        """
        img1 = HashSimilar.get_image(image1)
        img2 = HashSimilar.get_image(image2)
        
        # 计算图像标准差
        std1 = numpy.std(img1)
        std2 = numpy.std(img2)
        
        img1_list = HashSimilar.get_image_list(img1)
        img2_list = HashSimilar.get_image_list(img2)
        l = min(len(img1_list), len(img2_list))
        
        # 计算注意力分数
        score_list = HashSimilar.get_attention(img1, img2)
        score_list.sort()
        
        # 判断图像是否来自不同界面
        if score_list[int(len(score_list) * 0.8) - 1] < 0.8 or max(len(img1_list), len(img2_list)) - l > l \
                or abs(std1 - std2) > 35:
            return 0.2
        
        # 判断图像是否完全相似
        if min(score_list) > 0.99:
            return 1.0
        
        # 图像有增量变化
        return 0.8

    @staticmethod
    def get_hash_similar(image1, image2) -> float:
        """
        基于感知哈希的图像相似度计算
        :param image1: 图像A文件名
        :param image2: 图像B文件名
        :return: 哈希相似度分数 (0-1.0)
        """
        # 读取并预处理图像
        img1 = HashSimilar.get_image('capture/'+image1)
        img2 = HashSimilar.get_image('capture/'+image2)
        
        # 计算感知哈希值
        hash1 = HashSimilar.perception_hash(img1)
        hash2 = HashSimilar.perception_hash(img2)
        
        # 计算汉明距离并转换为相似度分数
        score = 1 - HashSimilar.hamming_dist(hash1, hash2) * 1.0 / (64 * 64)
        return score
