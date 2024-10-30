import ddddocr
from base64 import b64decode
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
ocr = ddddocr.DdddOcr(show_ad=False)  # 初始化 ddddocr 识别器

class OCRRequest(BaseModel):
    """请求体模型，用于接收 OCR 请求中的 base64 编码的图像数据。"""
    base64_img: str
  
@app.get("/")
async def root():
    """
    根路由，测试服务器是否正常运行。
    返回一个简单的消息，表明服务器正常工作。
    """
    return {"message": "OCR-Server Successful!"}

@app.post("/ocr/base64")
async def captcha_base64(data: OCRRequest):
    """
    从 base64 编码的图像字符串进行 OCR 识别。
    
    :param data: 包含 base64 编码的图像字符串
    :return: 包含识别结果的 JSON 对象
    """
    base64_img = data.base64_img  # 获取 base64 编码的图像字符串
    try:
        # 处理图像，包括降噪
        processed_image = process_image(load_image_from_base64(base64_img))
    except Exception as e:
        # 捕获图像处理中的异常并返回错误信息
        return {"code": -1, "result": f"Error: {str(e)}"}
    
    # 将处理后的图像转换为字节流
    _, image_bytes = cv2.imencode('.png', processed_image)

    try:
        # 使用 ddddocr 进行图像识别
        captcha = ocr.classification(image_bytes.tobytes())
        print(f"captcha: {captcha}")
    except Exception as e:
        # 捕获识别中的异常并返回错误信息
        return {"code": -1, "result": str(e)}

    # 返回识别结果
    return {"code": 0, "result": captcha}

@app.post("/ocr/bin")
async def captcha_bin(data: bytes):
    """
    从二进制图像数据进行 OCR 识别。
    
    :param data: 二进制的图像数据
    :return: 包含识别结果的 JSON 对象
    """
    try:
        # 处理图像，包括降噪和二值化
        processed_image = process_image(load_image_from_bin(data))
    except Exception as e:
        # 捕获图像处理中的异常并返回错误信息
        return {"code": -1, "result": f"Error: {str(e)}"}

    # 将处理后的图像转换为字节流
    _, image_bytes = cv2.imencode('.png', processed_image)

    try:
        # 使用 ddddocr 进行图像识别
        captcha = ocr.classification(image_bytes.tobytes())
    except Exception as e:
        # 捕获识别中的异常并返回错误信息
        return {"code": -1, "result": str(e)}

    # 返回识别结果
    return {"code": 0, "result": captcha}

def load_image_from_base64(b64_string: str) -> np.ndarray:
    """
    将 base64 编码的字符串转换为 OpenCV 图像对象。
    
    :param b64_string: base64 编码的图像字符串
    :return: OpenCV 图像对象
    """
    image_data = b64decode(b64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def load_image_from_bin(image_bin: bytes) -> np.ndarray:
    """
    将二进制图像数据转换为 OpenCV 图像对象。
    
    :param image_bin: 二进制的图像数据
    :return: OpenCV 图像对象
    """
    nparr = np.frombuffer(image_bin, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def process_image(image: np.ndarray) -> np.ndarray:
    """
    对图像进行预处理，包括降噪和二值化。
    
    :param image: 输入的 OpenCV 图像对象
    :return: 处理后的 OpenCV 图像对象
    """
    ret, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    # 图像四周边框置白
    image = around_white(thresh)
    # 去除孤立的噪点
    denoised_image = noise_unsome_pixel(image)
    return denoised_image

def around_white(image: np.ndarray) -> np.ndarray:
    """
    将图像四周的 5 像素宽边框置为白色。
    
    :param image: 输入的 OpenCV 图像对象
    :return: 边框置白后的 OpenCV 图像对象
    """
    # 创建一个白色的边框
    border_color = (255, 255, 255)
    cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), border_color, 5)
    return image


def noise_unsome_pixel(image: np.ndarray, similarity_threshold: int = 30) -> np.ndarray:
    """
    对图像中非边缘的孤立像素点进行降噪处理，并使用相邻颜色的均值填充孤立像素。
    
    :param image: 输入的图像数组
    :param similarity_threshold: 用于确定颜色相似性的阈值（默认30）。
    :return: 降噪后的图像数组
    """
    # 确保输入是 RGB 格式
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_array = image
    else:
        raise ValueError("Input image must be a BGR or RGB image.")
    
    # 获取图像的宽度和高度
    h, w, _ = img_array.shape  # 注意这里的顺序是 (高度, 宽度, 通道)
    
    # 创建一个与输入图像相同大小的输出图像
    output_image = img_array.copy()
    checked_pixels = np.zeros((h, w), dtype=bool)  # 标记已检查的像素

    # 判断两个颜色是否相似
    def is_similar_color(c1, c2, threshold):
        return np.linalg.norm(c1 - c2) < threshold

    # 遍历每个像素，判断其是否为孤立噪点
    for _h in range(1, h - 1):
        for _w in range(1, w - 1):
            if checked_pixels[_h, _w]:  # 如果已经检查过，则跳过
                continue

            center_color = img_array[_h, _w]
            neighbors = [
                img_array[_h - 1, _w - 1], img_array[_h - 1, _w], img_array[_h - 1, _w + 1],  # 上方3个邻居
                img_array[_h, _w - 1],                            img_array[_h, _w + 1],      # 左右邻居
                img_array[_h + 1, _w - 1], img_array[_h + 1, _w], img_array[_h + 1, _w + 1]   # 下方3个邻居
            ]

            # 统计相邻像素与中心像素的相似度
            cnt = sum(is_similar_color(center_color, neighbor, similarity_threshold) for neighbor in neighbors)
            
            # 如果相似的邻居少于1个，用相邻像素的均值替代中心像素
            if cnt < 1:
                # 计算周围像素的均值
                mean_color = np.mean(neighbors, axis=0)
                output_image[_h, _w] = mean_color  # 用均值填充
                checked_pixels[_h, _w] = True  # 标记为已检查
            else:
                # 如果中心像素与其某些邻居相似，则将这些相似的邻居也标记为已检查
                for i, neighbor in enumerate(neighbors):
                    if is_similar_color(center_color, neighbor, similarity_threshold):
                        # 根据邻居的索引更新 checked_pixels
                        if i == 0:
                            checked_pixels[_h - 1, _w - 1] = True  # 左上
                        elif i == 1:
                            checked_pixels[_h - 1, _w] = True      # 上
                        elif i == 2:
                            checked_pixels[_h - 1, _w + 1] = True  # 右上
                        elif i == 3:
                            checked_pixels[_h, _w - 1] = True      # 左
                        elif i == 4:
                            checked_pixels[_h, _w + 1] = True      # 右
                        elif i == 5:
                            checked_pixels[_h + 1, _w - 1] = True  # 左下
                        elif i == 6:
                            checked_pixels[_h + 1, _w] = True      # 下
                        elif i == 7:
                            checked_pixels[_h + 1, _w + 1] = True  # 右下
                checked_pixels[_h, _w] = True  # 标记中心像素为已检查

    return output_image


if __name__ == '__main__':
    ocr.set_ranges(6)   # 小写英文a-z + 大写英文A-Z + 整数0-9
    # 启动 FastAPI 服务器
    uvicorn.run('main:app', host="0.0.0.0", port=8899, reload=False)
