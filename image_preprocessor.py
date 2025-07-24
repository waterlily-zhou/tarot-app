#!/usr/bin/env python3
"""
图片预处理模块
用于优化塔罗卡牌图片的识别效果
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import tempfile
from typing import Tuple, Optional

class ImagePreprocessor:
    """图片预处理器"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "tarot_preprocessed"
        self.temp_dir.mkdir(exist_ok=True)
    
    def add_safe_margin(self, 
                       image_path: str, 
                       margin_size: int = 15,
                       fill_color: Tuple[int, int, int] = (255, 255, 255)) -> str:
        """
        为图片添加安全边距
        
        Args:
            image_path: 原图片路径
            margin_size: 边距大小（像素）
            fill_color: 填充颜色，默认白色 (255, 255, 255)
            
        Returns:
            处理后的图片路径
        """
        try:
            # 读取图片
            image = Image.open(image_path)
            
            # 获取原始尺寸
            original_width, original_height = image.size
            
            # 计算新尺寸
            new_width = original_width + 2 * margin_size
            new_height = original_height + 2 * margin_size
            
            # 创建新的图片（填充指定颜色）
            new_image = Image.new(image.mode, (new_width, new_height), fill_color)
            
            # 将原图片粘贴到中心位置
            paste_x = margin_size
            paste_y = margin_size
            new_image.paste(image, (paste_x, paste_y))
            
            # 保存到临时文件
            original_name = Path(image_path).stem
            output_path = self.temp_dir / f"{original_name}_with_margin.jpg"
            new_image.save(output_path, "JPEG", quality=95)
            
            print(f"✅ 已添加{margin_size}px边距: {original_width}x{original_height} → {new_width}x{new_height}")
            print(f"📁 处理后图片: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 图片预处理失败: {e}")
            return image_path  # 返回原路径作为备用
    
    def adjust_coordinates(self, 
                          coordinates: list, 
                          margin_size: int = 15) -> list:
        """
        调整坐标以适应添加边距后的图片
        
        Args:
            coordinates: 原始坐标列表 [(x1, y1), (x2, y2), ...]
            margin_size: 边距大小
            
        Returns:
            调整后的坐标列表
        """
        adjusted_coords = []
        
        for coord in coordinates:
            if isinstance(coord, (tuple, list)) and len(coord) >= 2:
                # 坐标需要减去边距偏移量（因为原图在新图中向右下移动了margin_size）
                new_x = coord[0] - margin_size
                new_y = coord[1] - margin_size
                adjusted_coords.append((new_x, new_y))
            else:
                adjusted_coords.append(coord)
        
        return adjusted_coords
    
    def parse_coordinate_string(self, coord_str: str) -> Tuple[Optional[int], Optional[int]]:
        """
        解析坐标字符串
        
        Args:
            coord_str: 坐标字符串，如 "(123, 456)"、"(123,456)" 或 "(块1)"
            
        Returns:
            (x, y) 坐标元组，如果解析失败返回 (None, None)
        """
        try:
            # 移除空白字符
            coord_str = coord_str.strip()
            
            # 检查是否是 "(块N)" 格式
            if "块" in coord_str:
                return (None, None)
            
            # 解析 "(x, y)" 或 "(x,y)" 格式
            if coord_str.startswith('(') and coord_str.endswith(')'):
                # 移除括号
                inner = coord_str[1:-1]
                # 分割坐标（处理有无空格的情况）
                parts = inner.split(',')
                
                if len(parts) == 2:
                    x = int(float(parts[0].strip()))
                    y = int(float(parts[1].strip()))
                    return (x, y)
            
            return (None, None)
            
        except (ValueError, AttributeError):
            return (None, None)
    
    def process_recognition_result(self, 
                                 recognition_result: list, 
                                 margin_size: int = 15) -> list:
        """
        处理识别结果，调整坐标以匹配原始图片
        
        Args:
            recognition_result: 识别结果列表
            margin_size: 添加的边距大小
            
        Returns:
            调整后的识别结果
        """
        processed_result = []
        
        # 检测坐标类型（网格坐标 vs 像素坐标）
        coord_type = self._detect_coordinate_type(recognition_result)
        
        for card in recognition_result:
            card_copy = card.copy()
            
            # 解析位置坐标
            position = card.get('position', '')
            x, y = self.parse_coordinate_string(position)
            
            if x is not None and y is not None:
                if coord_type == "grid":
                    # 网格坐标：不需要调整，保持原样
                    adjusted_x = x
                    adjusted_y = y
                    print(f"🔄 网格坐标保持: {card['card_name']} {position}")
                else:
                    # 像素坐标：需要减去边距偏移
                    adjusted_x = x - margin_size
                    adjusted_y = y - margin_size
                    
                    # 确保坐标不为负数
                    adjusted_x = max(0, adjusted_x)
                    adjusted_y = max(0, adjusted_y)
                    
                    print(f"🔄 像素坐标调整: {card['card_name']} {position} → ({adjusted_x}, {adjusted_y})")
                
                # 更新位置信息
                card_copy['position'] = f"({adjusted_x}, {adjusted_y})"
            
            processed_result.append(card_copy)
        
        return processed_result
    
    def _detect_coordinate_type(self, recognition_result: list) -> str:
        """
        检测坐标类型：网格坐标还是像素坐标
        
        Args:
            recognition_result: 识别结果列表
            
        Returns:
            "grid" 或 "pixel"
        """
        max_coord = 0
        
        for card in recognition_result:
            position = card.get('position', '')
            x, y = self.parse_coordinate_string(position)
            
            if x is not None and y is not None:
                max_coord = max(max_coord, x, y)
        
        # 如果最大坐标 <= 10，很可能是网格坐标
        if max_coord <= 10:
            print(f"🔍 检测到网格坐标系（最大值: {max_coord}）")
            return "grid"
        else:
            print(f"🔍 检测到像素坐标系（最大值: {max_coord}）")
            return "pixel"
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for temp_file in self.temp_dir.glob("*_with_margin.*"):
                temp_file.unlink()
            print("🧹 临时文件已清理")
        except Exception as e:
            print(f"⚠️ 清理临时文件时出错: {e}")
    
    def crop_right_edge(self, image_path: str, crop_percentage: float = 0.2) -> str:
        """
        裁剪图片右侧区域用于单独识别
        
        Args:
            image_path: 原图片路径
            crop_percentage: 裁剪百分比，0.2表示右侧20%
            
        Returns:
            裁剪后的图片路径
        """
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # 计算裁剪区域（右侧指定百分比）
            crop_start_x = int(width * (1 - crop_percentage))
            crop_region = (crop_start_x, 0, width, height)
            
            # 裁剪图片
            cropped_image = image.crop(crop_region)
            
            # 保存到临时文件（提高质量）
            original_name = Path(image_path).stem
            output_path = self.temp_dir / f"{original_name}_right_{int(crop_percentage*100)}pct.jpg"
            cropped_image.save(output_path, "JPEG", quality=100, optimize=False)
            
            print(f"✂️ 已裁剪右侧{int(crop_percentage*100)}%: {width}x{height} → {cropped_image.size}")
            print(f"📁 裁剪图片: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 图片裁剪失败: {e}")
            return image_path

    def get_image_info(self, image_path: str) -> dict:
        """获取图片信息"""
        try:
            image = Image.open(image_path)
            return {
                "path": image_path,
                "size": image.size,
                "mode": image.mode,
                "format": image.format
            }
        except Exception as e:
            return {"error": str(e)}

def main():
    """测试图片预处理功能"""
    print("🖼️ 图片预处理模块测试")
    print("=" * 40)
    
    # 初始化预处理器
    preprocessor = ImagePreprocessor()
    
    # 测试图片路径
    test_image = "data/card_images/spread_0_4821735726296_.pic.jpg"
    
    if Path(test_image).exists():
        # 获取原图信息
        original_info = preprocessor.get_image_info(test_image)
        print(f"📋 原图信息:")
        print(f"  尺寸: {original_info.get('size', 'Unknown')}")
        print(f"  格式: {original_info.get('format', 'Unknown')}")
        
        # 添加边距
        processed_image = preprocessor.add_safe_margin(test_image, margin_size=15)
        
        # 获取处理后图片信息
        processed_info = preprocessor.get_image_info(processed_image)
        print(f"\n📋 处理后图片信息:")
        print(f"  尺寸: {processed_info.get('size', 'Unknown')}")
        print(f"  路径: {processed_image}")
        
        # 测试坐标调整
        test_coords = [
            {"card_name": "测试卡1", "position": "(100, 150)"},
            {"card_name": "测试卡2", "position": "(200, 250)"},
            {"card_name": "测试卡3", "position": "(块1)"}  # 这个不会被调整
        ]
        
        print(f"\n🔄 坐标调整测试:")
        adjusted_coords = preprocessor.process_recognition_result(test_coords, margin_size=15)
        
        for original, adjusted in zip(test_coords, adjusted_coords):
            print(f"  {original['card_name']}: {original['position']} → {adjusted['position']}")
    
    else:
        print(f"❌ 测试图片不存在: {test_image}")
        print("💡 请确保有测试图片或修改测试路径")
    
    # 清理测试文件
    preprocessor.cleanup_temp_files()

if __name__ == "__main__":
    main() 