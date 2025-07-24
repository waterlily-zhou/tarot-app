#!/usr/bin/env python3
"""
分析图像边缘区域，检查是否有内容被裁切
"""
import cv2
import numpy as np
from pathlib import Path
from enhance_image import enhance_image_for_edge_detection

IMG = "data/card_images/spread_0_4821735726296_.pic.jpg"

def analyze_image(img_path, name):
    """分析图像的边缘区域"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取 {name}: {img_path}")
        return
    
    h, w = img.shape[:2]
    print(f"\n📊 {name} 分析:")
    print(f"   尺寸: {w}x{h}")
    
    # 分析各个边缘区域
    regions = {
        "左边5%": (0, 0, int(0.05*w), h),
        "右边5%": (int(0.95*w), 0, w, h),
        "上边5%": (0, 0, w, int(0.05*h)),
        "下边5%": (0, int(0.95*h), w, h)
    }
    
    for region_name, (x1, y1, x2, y2) in regions.items():
        crop = img[y1:y2, x1:x2]
        # 计算该区域的平均亮度和对比度
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)
        
        print(f"   {region_name}: 亮度={mean_brightness:.1f}, 对比度={std_contrast:.1f}")
        
        # 保存边缘区域
        save_path = f"debug_{name}_{region_name.replace('%', 'pct')}.jpg"
        cv2.imwrite(save_path, crop)
        
        # 检查是否有卡牌特征（边缘检测）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        print(f"   {region_name}: 边缘密度={edge_density:.3f}")

def main():
    print("🔍 图像边缘分析工具")
    print("="*50)
    
    # 分析原图
    analyze_image(IMG, "原图")
    
    # 生成并分析增强图
    enhanced_path = "debug_enhanced_full.jpg"
    enhanced_img = enhance_image_for_edge_detection(IMG, enhanced_path)
    
    if enhanced_img is not None:
        analyze_image(enhanced_path, "增强图")
    else:
        print("❌ 增强图生成失败")
    
    # 生成右边缘特写
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    
    # 右边 15%、20%、25% 三个版本
    for pct in [15, 20, 25]:
        edge = img[:, int((100-pct)/100 * w):]
        save_path = f"debug_right_{pct}pct.jpg"
        cv2.imwrite(save_path, edge)
        print(f"✅ 已保存右边{pct}%区域: {save_path}")

if __name__ == "__main__":
    main() 