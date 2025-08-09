"""
图像差异检测服务
基于感知哈希和行特征比较的图像差异检测算法
能够检测UI界面中的细微变化，适用于视觉回归测试
"""

import cv2
from service.image_similar import HashSimilar
from service.image_utils import get_hash_score, m_diff


class LineFeatureEqual(object):
    """
    行特征比较器
    用于比较图像行特征的相似性
    """
    def __init__(self):
        self.thresh = 0.85  # 相似度阈值

    def equal(self, a, b):
        """
        比较两个行特征是否相似
        :param a: 行特征A
        :param b: 行特征B
        :return: 是否相似
        """
        if get_hash_score(a, b) > self.thresh:
            return True
        else:
            return False


class ImageDiff(object):
    """
    图像差异检测器
    实现基于感知哈希和行特征比较的图像差异检测
    """
    def __init__(self, w=9, padding=80, w_scale=850, h_scale=0.08, pixel_value=28):
        """
        初始化差异检测器
        :param w: 滤波器窗口大小
        :param padding: 图像填充大小
        :param w_scale: 宽度缩放比例
        :param h_scale: 高度缩放比例
        :param pixel_value: 像素差异阈值
        """
        self.filter_w = w
        self.padding = padding
        self.size_scale = w_scale
        self.head_scale = h_scale
        self.pixel_value = pixel_value

    @staticmethod
    def get_line_list(op_list):
        """
        从操作列表中提取行列表
        :param op_list: 操作列表
        :return: 插入行列表和删除行列表
        """
        line1_list = []  # 插入的行
        line2_list = []  # 删除的行
        for op in op_list:
            if op["operation"] == "insert":
                line1_list.append(op["position_new"])
            if op["operation"] == "delete":
                line2_list.append(op["position_old"])
        return line1_list, line2_list

    @staticmethod
    def get_line_feature(image, precision=8):
        """
        提取图像的行特征
        :param image: 输入图像
        :param precision: 特征精度
        :return: 行特征列表
        """
        line_feature = []
        for y in range(image.shape[0]):
            # 将每行图像缩放到指定精度
            img = cv2.resize(image[y], (precision, precision))
            img_list = img.flatten()
            avg = sum(img_list) * 1. / len(img_list)
            # 计算感知哈希
            avg_list = ["0" if i < avg else "1" for i in img_list]
            line_feature.append([int(''.join(avg_list[x:x+4]), 2) for x in range(0, precision*precision)])
        return line_feature

    def get_image_feature(self, img1, img2):
        """
        提取两个图像的特征，并进行填充处理
        :param img1: 图像A
        :param img2: 图像B
        :return: 两个图像的特征
        """
        h1, w = img1.shape
        # 去除右侧填充区域
        img1 = img1[:, :w-self.padding]
        img2 = img2[:, :w-self.padding]
        img1_feature = self.get_line_feature(img1)
        img2_feature = self.get_line_feature(img2)
        return img1_feature, img2_feature

    def line_filter(self, line_list):
        """
        对行列表进行滤波处理
        :param line_list: 原始行列表
        :return: 滤波后的行列表
        """
        i = 0
        w = self.filter_w
        line = []
        while i < len(line_list)-w-1:
            f = line_list[i:i+w]
            s = 0
            # 计算相邻行的差值
            for j in range(w-1):
                s = s + f[j+1] - f[j]
            # 如果差值小于阈值，认为是连续的变化
            if s - w <= 6:
                for l in f:
                    if l not in line:
                        line.append(l)
            i = i + 1
        return line

    def get_image(self, image_file):
        """
        读取并预处理图像
        :param image_file: 图像文件路径
        :return: 预处理后的图像
        """
        image = cv2.imread(image_file)
        # 转换为灰度图
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 高斯模糊去噪
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        h, w = img.shape
        # 按比例缩放图像
        scale = self.size_scale/w
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return img

    @staticmethod
    def get_pixel(img, x, y):
        """
        获取图像指定位置的像素值
        :param img: 图像
        :param x: x坐标
        :param y: y坐标
        :return: 像素值
        """
        h, w = img.shape
        p = 0
        if y < h:
            p = img[y][x]
        return p

    def increment_diff(self, image1, image2, image_show) -> int:
        """
        计算增量图像差异
        :param image1: 输入图像A
        :param image2: 输入图像B
        :param image_show: 差异显示图像路径
        :return: 差异点数量
        """
        img1 = self.get_image(image1)
        img2 = self.get_image(image2)
        # 计算注意力分数
        score_list = HashSimilar.get_attention(img1, img2)
        # 提取图像特征
        img1_feature, img2_feature = self.get_image_feature(img1, img2)
        # 计算行差异
        line1, line2 = self.get_line_list(m_diff(img1_feature, img2_feature, equal_obj=LineFeatureEqual()))
        line = line1 + line2
        line = self.line_filter(line)
        
        # 创建显示图像
        img_show = img2.copy() if img2.shape[0] > img1.shape[0] else img1.copy()
        (h, w) = img_show.shape
        img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
        points = []
        line_attention = []
        
        # 过滤注意力分数较低的行
        for l in line:
            i = int((len(score_list) * (l - 1) / h))
            i = 0 if i < 0 else i
            if score_list[i] < 0.98:
                line_attention.append(l)
        line = line_attention
        
        # 逐像素比较差异
        for y in range(int(h*0.95)):
            if y > int(w * self.head_scale):
                if y in line:
                    for x in range(w-self.padding):
                        p1 = int(self.get_pixel(img1, x, y))
                        p2 = int(self.get_pixel(img2, x, y))
                        if abs(p1 - p2) < self.pixel_value:
                            pass
                        else:
                            points.append([x, y])
        
        # 在差异点绘制红色圆圈
        for point in points:
            cv2.circle(img_show, (point[0], point[1]), 1, (0, 0, 255), -1)
        cv2.imwrite(image_show, img_show)
        return len(points)

    def get_image_score(self, image1, image2, image_diff_name):
        """
        计算图像相似度分数
        :param image1: 图像A文件名
        :param image2: 图像B文件名
        :param image_diff_name: 差异图像文件名
        :return: 相似度分数
        """
        # 计算整体相似度
        score = HashSimilar.get_attention_similar('capture/'+image1, 'capture/'+image2)
        if score < 1.0:
            if score > 0.2:
                # 如果相似度在0.2-1.0之间，进行增量差异检测
                points_size = self.increment_diff('capture/'+image1, 'capture/'+image2, 'capture/'+image_diff_name)
                # 如果差异点很少，认为图像相似
                if points_size < 50:
                    score = 1.0
        return score
