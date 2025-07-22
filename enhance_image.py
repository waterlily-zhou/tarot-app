#!/usr/bin/env python3
"""
图像增强工具 - 提高边缘卡牌检测率
"""

import cv2
import numpy as np
from PIL import Image

def enhance_image_for_edge_detection(image_path: str, save_path: str = None):
    """增强图像以提高边缘检测效果"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. 增强对比度
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 2. 锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # 3. 边缘增强
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 4. 混合原图和边缘
    result = cv2.addWeighted(sharpened, 0.8, edges_colored, 0.2, 0)
    
    if save_path:
        cv2.imwrite(save_path, result)
        print(f"✅ 增强图像已保存: {save_path}")
    
    return result

# 使用方法
if __name__ == "__main__":
    enhanced = enhance_image_for_edge_detection(
        "data/card_images/spread_0_4821735726296_.pic.jpg",
        "enhanced_spread.jpg"
    ) 