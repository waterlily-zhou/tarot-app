#!/usr/bin/env python3
"""
圆形/星形塔罗牌阵检测器
专门处理圆形布局的牌阵
"""

from typing import List, Dict, Tuple
import google.generativeai as genai
from PIL import Image
from pathlib import Path
import os

def load_env_file():
    """加载.env.local文件"""
    env_file = Path('.env.local')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def configure_api():
    """配置Gemini API"""
    load_env_file()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ 需要Google API Key")
        print("💡 请在.env.local文件中设置: GOOGLE_API_KEY=你的API密钥")
        return False
    
    genai.configure(api_key=api_key)
    return True

class CircularDetector:
    """圆形牌阵检测器"""
    
    def detect_with_circular_prompt(self, image_path: str) -> List[Dict]:
        """
        使用圆形位置描述进行检测
        """
        img = Image.open(image_path)
        
        prompt = """
这张图片显示的是一个圆形/星形塔罗牌阵。请根据卡牌在圆形布局中的实际位置来描述它们的位置。

位置描述系统：
- 中心区域：CENTER (圆心附近的卡牌)
- 内圈：INNER_N, INNER_NE, INNER_E, INNER_SE, INNER_S, INNER_SW, INNER_W, INNER_NW (内圈8个方位)
- 外圈：OUTER_N, OUTER_NE, OUTER_E, OUTER_SE, OUTER_S, OUTER_SW, OUTER_W, OUTER_NW (外圈8个方位)

输出格式：
卡牌名称,正位/逆位,位置描述

例如：
愚人,正位,CENTER
权杖三,逆位,INNER_N
星币皇后,正位,OUTER_E

请仔细观察每张卡牌在圆形布局中的位置：
"""
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, img])
        
        cards = []
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                if ',' in line and not line.strip().startswith('卡牌名称'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        card_name = parts[0].strip()
                        orientation = parts[1].strip()
                        position = parts[2].strip()
                        
                        cards.append({
                            'card_name': card_name,
                            'orientation': orientation,
                            'circular_position': position,
                            'position_type': self._get_position_type(position)
                        })
        
        return cards
    
    def _get_position_type(self, position: str) -> str:
        """获取位置类型"""
        if 'CENTER' in position:
            return 'center'
        elif 'INNER' in position:
            return 'inner'
        elif 'OUTER' in position:
            return 'outer'
        else:
            return 'unknown'
    
    def visualize_circular(self, image_path: str, cards: List[Dict], output_path: str = None) -> str:
        """
        在图片上可视化圆形位置
        """
        import cv2
        import numpy as np
        
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # 绘制圆形网格参考线
        inner_radius = min(width, height) // 4
        outer_radius = min(width, height) // 3
        
        # 绘制内圈和外圈
        cv2.circle(image, center, inner_radius, (128, 128, 128), 2)
        cv2.circle(image, center, outer_radius, (128, 128, 128), 2)
        
        # 绘制方位线
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            angle_rad = np.radians(angle)
            end_x = int(center[0] + outer_radius * np.cos(angle_rad))
            end_y = int(center[1] + outer_radius * np.sin(angle_rad))
            cv2.line(image, center, (end_x, end_y), (128, 128, 128), 1)
        
        # 标记卡牌
        for i, card in enumerate(cards):
            # 简单的位置映射（需要根据实际结果调整）
            if card['position_type'] == 'center':
                x, y = center
                color = (0, 255, 0)  # 绿色
            elif card['position_type'] == 'inner':
                # 根据方位计算位置
                angle = self._get_angle_from_position(card['circular_position'])
                x = int(center[0] + inner_radius * 0.8 * np.cos(np.radians(angle)))
                y = int(center[1] + inner_radius * 0.8 * np.sin(np.radians(angle)))
                color = (255, 255, 0)  # 黄色
            else:  # outer
                angle = self._get_angle_from_position(card['circular_position'])
                x = int(center[0] + outer_radius * 0.9 * np.cos(np.radians(angle)))
                y = int(center[1] + outer_radius * 0.9 * np.sin(np.radians(angle)))
                color = (0, 0, 255)  # 红色
            
            cv2.circle(image, (x, y), 8, color, -1)
            cv2.putText(image, str(i + 1), (x - 5, y + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存结果
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_circular.jpg")
        
        cv2.imwrite(output_path, image)
        print(f"📁 圆形检测结果已保存: {output_path}")
        return output_path
    
    def _get_angle_from_position(self, position: str) -> float:
        """从位置描述获取角度"""
        angle_map = {
            'N': 270, 'NE': 315, 'E': 0, 'SE': 45,
            'S': 90, 'SW': 135, 'W': 180, 'NW': 225
        }
        
        for direction, angle in angle_map.items():
            if direction in position:
                return angle
        return 0

def main():
    """测试圆形检测功能"""
    print("🌟 圆形牌阵检测测试")
    print("=" * 40)
    
    # 配置API
    if not configure_api():
        return
    
    detector = CircularDetector()
    
    # 测试图片
    test_image = "data/card_images/spread_0_4821735726296_.pic.jpg"
    
    if Path(test_image).exists():
        cards = detector.detect_with_circular_prompt(test_image)
        
        if cards:
            print(f"\n📋 检测到 {len(cards)} 张卡牌:")
            for i, card in enumerate(cards, 1):
                print(f"{i:2d}. {card['card_name']} ({card['orientation']}) - {card['circular_position']}")
            
            # 生成可视化结果
            output_path = detector.visualize_circular(test_image, cards)
        else:
            print("⚠️ 未检测到卡牌")
    else:
        print(f"❌ 测试图片不存在: {test_image}")

if __name__ == "__main__":
    main() 