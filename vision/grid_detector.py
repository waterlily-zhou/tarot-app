#!/usr/bin/env python3
"""
基于网格的塔罗卡牌位置检测
将图片分割为网格，使用相对位置而不是绝对坐标
"""

from typing import List, Dict, Tuple
import google.generativeai as genai
from PIL import Image
from pathlib import Path
import os

# 配置API密钥
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

class GridDetector:
    """网格位置检测器"""
    
    def __init__(self, grid_size: Tuple[int, int] = (8, 6)):
        """
        初始化网格检测器
        
        Args:
            grid_size: 网格尺寸 (列数, 行数)，默认8x6平衡精度和准确性
        """
        self.grid_cols, self.grid_rows = grid_size
        
    def detect_with_grid_prompt(self, image_path: str) -> List[Dict]:
        """
        使用网格提示词进行检测
        """
        img = Image.open(image_path)
        
        prompt = f"""
请将这张塔罗牌阵图片分成 {self.grid_cols}x{self.grid_rows} 的网格，然后识别每张卡牌位于哪个网格位置。

网格系统说明：
- 横向分为 {self.grid_cols} 列：A, B, C, D, E, F, G, H (从左到右)  
- 纵向分为 {self.grid_rows} 行：1, 2, 3, 4, 5, 6 (从上到下)

这是一个圆形/星形牌阵，卡牌分布在圆形区域内。请仔细观察每张卡牌的中心位置，判断它最接近哪个网格区域。

输出格式：
卡牌名称,正位/逆位,网格位置

例如：
愚人,正位,D3
权杖三,逆位,B2
星币皇后,正位,F5

请仔细观察每张卡牌的位置，给出准确的网格坐标：
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
                        grid_pos = parts[2].strip()
                        
                        # 转换网格位置为数值坐标
                        numeric_coords = self.grid_to_numeric(grid_pos)
                        
                        cards.append({
                            'card_name': card_name,
                            'orientation': orientation,
                            'grid_position': grid_pos,
                            'numeric_coords': numeric_coords
                        })
        
        return cards
    
    def detect_relative_positions(self, image_path: str) -> List[Dict]:
        """
        检测卡牌间的相对位置关系（以牌数量为单位）
        """
        img = Image.open(image_path)
        
        prompt = """
请分析这张塔罗牌阵图片，识别所有卡牌并描述它们之间的相对位置关系。

步骤：
1. 首先识别所有可见的卡牌
2. 选择位于中心区域的一张卡牌作为原点(0,0)
3. 描述其他卡牌相对于中心牌的位置，用"卡牌数量"作为距离单位

位置描述规则：
- 中心牌：(0, 0)
- 右边的牌：正x值，如(1, 0)表示右边隔1张牌，(2, 0)表示右边隔2张牌
- 左边的牌：负x值，如(-1, 0)表示左边隔1张牌，(-3, 0)表示左边隔3张牌  
- 上方的牌：正y值，如(0, 1)表示上方隔1张牌，(0, 2)表示上方隔2张牌
- 下方的牌：负y值，如(0, -1)表示下方隔1张牌，(0, -2)表示下方隔2张牌
- 对角线：如(1, 1)表示右上方，(-1, -1)表示左下方

输出格式：
卡牌名称,正位/逆位,相对位置(x,y)

例如：
圣杯国王,正位,(0,0)
星币侍从,正位,(0,2)  
力量,逆位,(0,-2)
宝剑三,正位,(-2,0)
星币七,正位,(2,0)

请开始分析：
"""
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, img])
        
        cards = []
        if response.text:
            print("🔍 Gemini原始输出:")
            print("-" * 30)
            print(response.text)
            print("-" * 30)
            
            lines = response.text.strip().split('\n')
            for line in lines:
                if ',' in line and not line.strip().startswith('卡牌名称'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        card_name = parts[0].strip()
                        orientation = parts[1].strip()
                        position_str = parts[2].strip()
                        
                        # 解析相对位置
                        relative_pos = self._parse_relative_position(position_str)
                        
                        cards.append({
                            'card_name': card_name,
                            'orientation': orientation,
                            'relative_position': position_str,
                            'relative_coords': relative_pos
                        })
        
        return cards
    
    def _parse_relative_position(self, pos_str: str) -> Tuple[int, int]:
        """解析相对位置坐标"""
        try:
            # 移除空格和括号
            pos_str = pos_str.strip().replace('(', '').replace(')', '')
            if ',' in pos_str:
                x_str, y_str = pos_str.split(',')
                x = int(float(x_str.strip()))
                y = int(float(y_str.strip()))
                return (x, y)
        except (ValueError, AttributeError):
            pass
        return (0, 0)
    
    def grid_to_numeric(self, grid_pos: str) -> Tuple[int, int]:
        """
        将网格位置转换为数值坐标
        
        Args:
            grid_pos: 网格位置，如 "A1", "I8", "R15"
            
        Returns:
            (x, y) 数值坐标
        """
        if len(grid_pos) >= 2:
            col = grid_pos[0].upper()
            row_str = grid_pos[1:]
            
            try:
                # 列：A=0, B=1, C=2, ... R=17
                col_num = ord(col) - ord('A')
                # 行：1=0, 2=1, 3=2, ... 15=14
                row_num = int(row_str) - 1
                
                return (col_num, row_num)
            except ValueError:
                return (0, 0)
        
        return (0, 0)
    
    def visualize_grid(self, image_path: str, cards: List[Dict], output_path: str = None) -> str:
        """
        在图片上可视化网格和卡牌位置
        """
        import cv2
        import numpy as np
        
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # 计算网格线位置
        col_width = width // self.grid_cols
        row_height = height // self.grid_rows
        
        # 绘制网格线
        for i in range(1, self.grid_cols):
            x = i * col_width
            cv2.line(image, (x, 0), (x, height), (128, 128, 128), 2)
        
        for i in range(1, self.grid_rows):
            y = i * row_height
            cv2.line(image, (0, y), (width, y), (128, 128, 128), 2)
        
        # 添加网格标签（8x6网格可以显示所有标签）
        for col in range(self.grid_cols):
            for row in range(self.grid_rows):
                label = f"{chr(ord('A') + col)}{row + 1}"
                x = col * col_width + 10
                y = row * row_height + 30
                cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 标记卡牌位置
        print(f"\n🔍 调试信息 - 图片尺寸: {width}x{height}")
        print(f"📏 网格尺寸: {self.grid_cols}列 x {self.grid_rows}行")
        print(f"📐 单元格尺寸: {col_width}x{row_height}像素")
        print()
        
        for i, card in enumerate(cards):
            col_num, row_num = card['numeric_coords']
            center_x = col_num * col_width + col_width // 2
            center_y = row_num * row_height + row_height // 2
            
            print(f"🎴 卡牌{i+1}: {card['card_name']}")
            print(f"   网格位置: {card['grid_position']} → 数值坐标: ({col_num}, {row_num})")
            print(f"   计算像素位置: ({center_x}, {center_y})")
            print()
            
            # 绘制卡牌标记
            cv2.circle(image, (center_x, center_y), 15, (0, 255, 0), -1)
            cv2.putText(image, str(i + 1), (center_x - 8, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 保存结果
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_grid.jpg")
        
        cv2.imwrite(output_path, image)
        print(f"📁 网格检测结果已保存: {output_path}")
        return output_path

    def visualize_relative_positions(self, image_path: str, cards: List[Dict], output_path: str = None) -> str:
        """
        可视化相对位置关系
        """
        import cv2
        import numpy as np
        
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # 找到中心牌(0,0)
        center_card = None
        for card in cards:
            if card['relative_coords'] == (0, 0):
                center_card = card
                break
        
        if center_card:
            print(f"🎯 中心牌: {center_card['card_name']} ({center_card['orientation']})")
        
        # 绘制中心点
        cv2.circle(image, center, 10, (0, 255, 255), -1)  # 黄色中心点
        cv2.putText(image, "CENTER(0,0)", (center[0] - 50, center[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制相对位置
        card_spacing = 120  # 卡牌间距（像素）
        
        for i, card in enumerate(cards):
            rel_x, rel_y = card['relative_coords']
            
            # 计算可视化位置（注意y轴翻转）
            vis_x = center[0] + rel_x * card_spacing
            vis_y = center[1] - rel_y * card_spacing  # y轴翻转（图像坐标系）
            
            # 选择颜色
            if (rel_x, rel_y) == (0, 0):
                color = (0, 255, 255)  # 黄色 - 中心牌
            elif rel_x == 0:
                color = (255, 0, 0)    # 蓝色 - 垂直方向
            elif rel_y == 0:
                color = (0, 255, 0)    # 绿色 - 水平方向
            else:
                color = (255, 0, 255)  # 紫色 - 对角线
            
            # 绘制卡牌标记
            cv2.circle(image, (vis_x, vis_y), 12, color, -1)
            cv2.putText(image, str(i + 1), (vis_x - 6, vis_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 添加坐标标签
            coord_text = f"({rel_x},{rel_y})"
            cv2.putText(image, coord_text, (vis_x - 20, vis_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 保存结果
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_relative.jpg")
        
        cv2.imwrite(output_path, image)
        print(f"📁 相对位置结果已保存: {output_path}")
        return output_path

def main():
    """测试位置检测功能"""
    print("🎯 塔罗牌位置检测测试")
    print("=" * 40)
    
    # 配置API
    if not configure_api():
        return
    
    detector = GridDetector()
    
    # 测试图片
    test_image = "data/card_images/spread_0_4821735726296_.pic.jpg"
    
    if Path(test_image).exists():
        print("选择检测方法:")
        print("1. 网格位置检测")
        print("2. 相对位置检测（推荐）")
        
        choice = input("请选择 (1-2): ").strip()
        
        if choice == "2":
            # 相对位置检测
            print("\n🎯 使用相对位置检测...")
            cards = detector.detect_relative_positions(test_image)
            
            if cards:
                print(f"\n📋 检测到 {len(cards)} 张卡牌（相对位置）:")
                print("=" * 50)
                
                # 找出中心牌
                center_card = None
                for card in cards:
                    if card['relative_coords'] == (0, 0):
                        center_card = card
                        break
                
                if center_card:
                    print(f"🎯 中心牌: {center_card['card_name']} ({center_card['orientation']}) - (0,0)")
                    print("-" * 30)
                
                # 显示所有卡牌
                for i, card in enumerate(cards, 1):
                    rel_x, rel_y = card['relative_coords']
                    
                    # 生成方向描述
                    if rel_x == 0 and rel_y == 0:
                        direction = "🎯中心"
                    else:
                        direction = ""
                        if rel_x > 0: direction += "右"
                        elif rel_x < 0: direction += "左"
                        if rel_y > 0: direction += "上"
                        elif rel_y < 0: direction += "下"
                    
                    print(f"{i:2d}. {card['card_name']} ({card['orientation']}) - ({rel_x:2d},{rel_y:2d}) {direction}")
                
                print("=" * 50)
                
                # 生成可视化结果
                output_path = detector.visualize_relative_positions(test_image, cards)
                
            else:
                print("⚠️ 未检测到卡牌")
        
        else:
            # 原来的网格检测
            print("\n🎯 使用网格位置检测...")
            cards = detector.detect_with_grid_prompt(test_image)
            
            if cards:
                print(f"\n📋 检测到 {len(cards)} 张卡牌（网格位置）:")
                for i, card in enumerate(cards, 1):
                    print(f"{i:2d}. {card['card_name']} ({card['orientation']}) - {card['grid_position']}")
                
                # 生成可视化结果
                output_path = detector.visualize_grid(test_image, cards)
            else:
                print("⚠️ 未检测到卡牌")
    else:
        print(f"❌ 测试图片不存在: {test_image}")

if __name__ == "__main__":
    main() 