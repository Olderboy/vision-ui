"""
Vision UI 服务器入口文件
提供基于Flask的Web API服务，集成多种视觉分析功能
"""

from flask import Flask
from flask import jsonify
from flask_cors import CORS
from api.vision_api import vision


# 创建Flask应用实例
app = Flask(__name__)
# 启用跨域支持，允许前端跨域访问API
CORS(app)

# 注册vision蓝图，所有API路由都以/vision为前缀
app.register_blueprint(vision, url_prefix='/vision')


@app.errorhandler(Exception)
def error(e):
    """
    全局异常处理器
    捕获所有未处理的异常并返回统一的错误响应格式
    """
    ret = dict()
    ret["code"] = 1  # 错误码1表示异常
    ret["data"] = repr(e)  # 返回异常信息
    return jsonify(ret)


if __name__ == '__main__':
    # 启动Flask服务器，监听所有网络接口的9092端口
    app.run(host="0.0.0.0", port=9092)
