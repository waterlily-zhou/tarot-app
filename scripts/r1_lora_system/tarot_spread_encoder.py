#!/usr/bin/env python3
"""
塔罗牌阵位置编码系统
基于中心轴、底牌、对称性的通用牌阵表示方法
"""

class TarotSpreadEncoder:
    """塔罗牌阵编码器"""
    
    def __init__(self):
        # 定义牌阵模板
        self.spread_templates = {
            "七大脉轮牌阵": {
                "positions": 7,
                "layout": "vertical_chakra",
                "structure": {
                    "center_axis": [1, 2, 3, 4, 5, 6, 7],  # 垂直中心轴
                    "base_card": 1,      # 海底轮为底牌
                    "crown_card": 7,     # 顶轮为顶牌
                    "symmetry": "vertical"
                },
                "position_meanings": {
                    1: "海底轮-基础能量",
                    2: "生殖轮-情感创造", 
                    3: "太阳神经丛-意志力",
                    4: "心轮-爱与平衡",
                    5: "喉轮-表达沟通",
                    6: "三眼轮-直觉智慧",
                    7: "顶轮-灵性连接"
                }
            },
            
            "六芒星牌阵": {
                "positions": 6,
                "layout": "hexagram",
                "structure": {
                    "center_axis": [3, 4],       # 中心轴
                    "base_card": 3,              # 底部中心
                    "crown_card": 4,             # 顶部中心  
                    "left_wing": [1, 5],         # 左翼
                    "right_wing": [2, 6],        # 右翼
                    "symmetry": "bilateral"       # 双边对称
                },
                "position_meanings": {
                    1: "左下-过去影响",
                    2: "右下-现在状况", 
                    3: "底部-根本原因",
                    4: "顶部-最终结果",
                    5: "左上-内在因素",
                    6: "右上-外在环境"
                }
            },
            
            "四季牌阵": {
                "positions": 5,
                "layout": "seasonal_cross",
                "structure": {
                    "center_axis": [5],           # 中心点
                    "base_card": 3,              # 秋-基础
                    "crown_card": 1,             # 春-顶点
                    "left_wing": [4],            # 冬-左
                    "right_wing": [2],           # 夏-右
                    "symmetry": "cross"
                },
                "position_meanings": {
                    1: "春分-新生能量",
                    2: "夏至-顶峰力量",
                    3: "秋分-收获智慧", 
                    4: "冬至-内省转化",
                    5: "中心-核心主题"
                }
            },
            
            "金字塔牌阵": {
                "positions": 5,
                "layout": "pyramid",
                "structure": {
                    "center_axis": [1, 5],        # 底到顶的轴心
                    "base_card": 1,              # 金字塔底部
                    "crown_card": 5,             # 金字塔顶点
                    "left_wing": [2, 4],         # 左侧结构  
                    "right_wing": [3, 4],        # 右侧结构
                    "symmetry": "triangular"
                },
                "position_meanings": {
                    1: "基础-根本状况",
                    2: "左支撑-内在力量",
                    3: "右支撑-外在资源",
                    4: "中层-发展过程", 
                    5: "顶点-最终目标"
                }
            }
        }
    
    def encode_spread_position(self, spread_name, card_position, card_name, orientation="正位"):
        """编码单张牌的位置信息"""
        if spread_name not in self.spread_templates:
            return f"{card_name}({orientation})"
        
        template = self.spread_templates[spread_name]
        position_meaning = template["position_meanings"].get(card_position, f"位置{card_position}")
        
        # 分析位置特征
        structure = template["structure"]
        position_features = []
        
        if card_position == structure.get("base_card"):
            position_features.append("底牌")
        if card_position == structure.get("crown_card"):
            position_features.append("顶牌")
        if card_position in structure.get("center_axis", []):
            position_features.append("中心轴")
        if card_position in structure.get("left_wing", []):
            position_features.append("左翼")
        if card_position in structure.get("right_wing", []):
            position_features.append("右翼")
        
        # 构建位置编码
        position_code = f"[{position_meaning}"
        if position_features:
            position_code += f"|{','.join(position_features)}"
        position_code += "]"
        
        return f"{card_name}({orientation}){position_code}"
    
    def encode_full_spread(self, spread_name, cards_data):
        """编码整个牌阵"""
        """
        cards_data格式:
        [
            {"position": 1, "card": "太阳", "orientation": "正位"},
            {"position": 2, "card": "皇后", "orientation": "正位"},
            ...
        ]
        """
        if spread_name not in self.spread_templates:
            return "未知牌阵"
        
        template = self.spread_templates[spread_name]
        encoded_cards = []
        
        for card_data in cards_data:
            encoded_card = self.encode_spread_position(
                spread_name, 
                card_data["position"],
                card_data["card"],
                card_data["orientation"]
            )
            encoded_cards.append(encoded_card)
        
        # 添加牌阵结构信息
        structure_info = f"对称性:{template['structure']['symmetry']}"
        
        return {
            "spread": spread_name,
            "structure": structure_info,
            "cards": encoded_cards,
            "encoded_text": f"牌阵：{spread_name}({structure_info}) - " + "; ".join(encoded_cards)
        }

def enhance_training_data_with_positions():
    """为现有训练数据增加位置编码"""
    print("🎴 增强训练数据的位置编码...")
    
    encoder = TarotSpreadEncoder()
    
    # 示例：如何解析和编码现有数据
    sample_cases = [
        {
            "original": "牌：力量(正位); 圣杯九(正位); 太阳(正位); 审判(正位); 教皇(正位); 月亮(正位); 皇后(正位)",
            "spread": "七大脉轮牌阵",
            "cards": [
                {"position": 1, "card": "太阳", "orientation": "正位"},      # 海底轮
                {"position": 2, "card": "皇后", "orientation": "正位"},      # 生殖轮  
                {"position": 3, "card": "教皇", "orientation": "正位"},      # 太阳神经丛
                {"position": 4, "card": "审判", "orientation": "正位"},      # 心轮
                {"position": 5, "card": "月亮", "orientation": "正位"},      # 喉轮
                {"position": 6, "card": "圣杯九", "orientation": "正位"},    # 三眼轮
                {"position": 7, "card": "力量", "orientation": "正位"}       # 顶轮
            ]
        }
    ]
    
    for case in sample_cases:
        encoded = encoder.encode_full_spread(case["spread"], case["cards"])
        print(f"\n原始: {case['original']}")
        print(f"增强: {encoded['encoded_text']}")
        print(f"结构: {encoded['structure']}")
        
        # 显示每张牌的详细位置信息
        for card in encoded['cards']:
            print(f"  - {card}")

def create_position_aware_dataset():
    """创建位置感知的训练数据集"""
    print("📊 创建位置感知数据集...")
    
    # 这里可以读取原始数据并添加位置编码
    # 为了演示，我们创建一个示例
    
    enhanced_prompt_template = """塔罗解读：
咨询者：{client}
问题：{question}
牌阵：{spread_with_structure}
牌：{cards_with_positions}

请基于牌阵位置的能量特征提供专业解读：
- 分析中心轴牌的核心主题
- 解读底牌的根基能量
- 考虑对称位置的平衡关系
- 综合位置意义与牌义"""
    
    print("✅ 位置感知模板已创建")
    print("💡 可以通过此模板增强现有训练数据")

if __name__ == "__main__":
    print("🎴 塔罗牌阵位置编码系统")
    print("="*50)
    
    # 演示位置编码
    enhance_training_data_with_positions()
    
    # 创建位置感知数据集
    create_position_aware_dataset()
    
    print("\n🌟 位置编码系统已就绪！")
    print("📝 可用于增强训练数据的位置信息")