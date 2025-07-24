#!/usr/bin/env python3
"""
塔罗牌阵系统
定义各种经典牌阵，包括位置含义、卡牌关系、解读逻辑
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class CardPosition:
    """牌阵中的卡牌位置"""
    position_id: str
    name: str
    meaning: str
    description: str
    x: float  # 标准化坐标 (0-1)
    y: float  # 标准化坐标 (0-1)
    importance: int  # 重要性等级 1-5
    
@dataclass
class SpreadRelationship:
    """牌阵中卡牌间的关系"""
    name: str
    positions: List[str]
    relationship_type: str  # "opposition", "support", "sequence", "axis"
    description: str

class TarotSpread:
    """塔罗牌阵基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.positions: Dict[str, CardPosition] = {}
        self.relationships: List[SpreadRelationship] = []
        self.interpretation_guide = ""
        
    def add_position(self, position: CardPosition):
        """添加位置"""
        self.positions[position.position_id] = position
        
    def add_relationship(self, relationship: SpreadRelationship):
        """添加卡牌关系"""
        self.relationships.append(relationship)
        
    def get_position_meaning(self, position_id: str) -> str:
        """获取位置含义"""
        if position_id in self.positions:
            return self.positions[position_id].meaning
        return "未知位置"
        
    def get_related_positions(self, position_id: str) -> List[str]:
        """获取相关位置"""
        related = []
        for rel in self.relationships:
            if position_id in rel.positions:
                related.extend([p for p in rel.positions if p != position_id])
        return list(set(related))

class TarotSpreadSystem:
    """塔罗牌阵管理系统"""
    
    def __init__(self):
        self.spreads: Dict[str, TarotSpread] = {}
        self._initialize_spreads()
        
    def _initialize_spreads(self):
        """初始化标准牌阵"""
        
        # 1. 三卡展开 (过去-现在-未来)
        three_card = TarotSpread(
            "三卡展开", 
            "最经典的三卡牌阵，展现时间线或情况的三个方面"
        )
        
        three_card.add_position(CardPosition(
            "past", "过去", "过去的影响", 
            "导致当前情况的过去因素、根源、已发生的事件", 
            0.2, 0.5, 4
        ))
        three_card.add_position(CardPosition(
            "present", "现在", "当前状况", 
            "目前的状态、当下的挑战、需要关注的核心议题", 
            0.5, 0.5, 5
        ))
        three_card.add_position(CardPosition(
            "future", "未来", "未来趋势", 
            "可能的结果、发展方向、需要准备的事项", 
            0.8, 0.5, 4
        ))
        
        three_card.add_relationship(SpreadRelationship(
            "时间轴", ["past", "present", "future"], "sequence",
            "从过去到现在到未来的发展序列，显示事物的演进过程"
        ))
        
        three_card.interpretation_guide = """
        三卡解读要点：
        1. 首先分析时间线的连贯性
        2. 关注中心位置（现在）与两侧的关系
        3. 看过去如何影响现在，现在如何导向未来
        4. 注意卡牌间的能量流动和转换
        """
        
        self.spreads["three_card"] = three_card
        
        # 2. 凯尔特十字
        celtic_cross = TarotSpread(
            "凯尔特十字",
            "最著名的十卡牌阵，提供全面深入的生活洞察"
        )
        
        positions = [
            ("situation", "当前情况", "核心议题", "当前面临的主要情况或挑战", 0.4, 0.5, 5),
            ("challenge", "挑战/机遇", "交叉影响", "影响当前情况的挑战或机遇因素", 0.6, 0.5, 4),
            ("distant_past", "遥远过去", "根源", "形成当前情况的深层根源", 0.4, 0.3, 3),
            ("recent_past", "近期过去", "最近影响", "最近发生的相关事件", 0.2, 0.5, 3),
            ("possible_outcome", "可能结果", "潜在未来", "按当前轨迹可能的结果", 0.4, 0.7, 4),
            ("immediate_future", "近期未来", "即将发生", "接下来几周或几个月的发展", 0.6, 0.5, 4),
            ("your_approach", "你的方法", "内在态度", "你对情况的态度和处理方式", 0.8, 0.8, 4),
            ("external_influence", "外在影响", "环境因素", "他人或环境对你的影响", 0.8, 0.6, 3),
            ("hopes_fears", "希望与恐惧", "内心状态", "你的期望和担忧", 0.8, 0.4, 3),
            ("final_outcome", "最终结果", "最终结果", "综合所有因素后的最终可能结果", 0.8, 0.2, 5)
        ]
        
        for pos_id, name, meaning, desc, x, y, importance in positions:
            celtic_cross.add_position(CardPosition(pos_id, name, meaning, desc, x, y, importance))
        
        # 添加关系
        celtic_cross.add_relationship(SpreadRelationship(
            "核心轴", ["situation", "challenge"], "axis",
            "当前情况与其主要影响因素的核心轴线"
        ))
        celtic_cross.add_relationship(SpreadRelationship(
            "时间线", ["distant_past", "recent_past", "situation", "immediate_future", "possible_outcome"], "sequence",
            "从遥远过去到可能未来的时间发展线"
        ))
        celtic_cross.add_relationship(SpreadRelationship(
            "内外对比", ["your_approach", "external_influence"], "opposition",
            "内在态度与外在环境的对比关系"
        ))
        
        celtic_cross.interpretation_guide = """
        凯尔特十字解读要点：
        1. 先解读核心轴（位置1-2），理解主要情况
        2. 分析时间线（位置3-4-1-6-5），看发展脉络
        3. 解读右侧职员塔（位置7-8-9-10），了解心理和环境因素
        4. 综合分析，特别注意位置10的最终结果
        5. 关注卡牌间的相互呼应和矛盾
        """
        
        self.spreads["celtic_cross"] = celtic_cross
        
        # 3. 关系牌阵
        relationship = TarotSpread(
            "关系牌阵",
            "专门分析两人关系的七卡牌阵"
        )
        
        rel_positions = [
            ("you", "你的状态", "你在关系中的状态", "你在这段关系中的心态、感受和表现", 0.2, 0.6, 4),
            ("them", "对方状态", "对方在关系中的状态", "对方在这段关系中的心态、感受和表现", 0.8, 0.6, 4),
            ("connection", "关系连接", "关系的本质", "你们之间的连接本质和关系特质", 0.5, 0.4, 5),
            ("challenge", "关系挑战", "面临的挑战", "关系中需要克服的困难或问题", 0.5, 0.8, 4),
            ("strength", "关系优势", "关系的优势", "关系中的积极因素和优势", 0.5, 0.2, 4),
            ("advice", "建议", "关系建议", "改善或维护关系的建议", 0.3, 0.1, 4),
            ("outcome", "关系前景", "关系的发展前景", "按当前情况发展的关系前景", 0.7, 0.1, 5)
        ]
        
        for pos_id, name, meaning, desc, x, y, importance in rel_positions:
            relationship.add_position(CardPosition(pos_id, name, meaning, desc, x, y, importance))
        
        relationship.add_relationship(SpreadRelationship(
            "双方对比", ["you", "them"], "opposition",
            "关系中双方的状态对比"
        ))
        relationship.add_relationship(SpreadRelationship(
            "挑战-优势", ["challenge", "strength"], "opposition", 
            "关系中的正负面因素对比"
        ))
        
        self.spreads["relationship"] = relationship
        
        # 4. 决策牌阵
        decision = TarotSpread(
            "决策牌阵",
            "帮助做重要决策的五卡牌阵"
        )
        
        dec_positions = [
            ("situation", "当前情况", "决策背景", "需要做决策的当前情况", 0.5, 0.8, 5),
            ("option_a", "选择A", "第一个选择", "第一个选择及其后果", 0.2, 0.5, 4),
            ("option_b", "选择B", "第二个选择", "第二个选择及其后果", 0.8, 0.5, 4),
            ("advice", "建议", "决策建议", "帮助做决策的智慧指导", 0.5, 0.2, 4),
            ("outcome", "最佳结果", "理想结果", "做出正确决策后的理想结果", 0.5, 0.05, 3)
        ]
        
        for pos_id, name, meaning, desc, x, y, importance in dec_positions:
            decision.add_position(CardPosition(pos_id, name, meaning, desc, x, y, importance))
        
        decision.add_relationship(SpreadRelationship(
            "选择对比", ["option_a", "option_b"], "opposition",
            "两个选择之间的对比分析"
        ))
        
        self.spreads["decision"] = decision
        
    def get_spread(self, spread_name: str) -> Optional[TarotSpread]:
        """获取指定牌阵"""
        return self.spreads.get(spread_name)
        
    def list_spreads(self) -> List[str]:
        """列出所有可用牌阵"""
        return list(self.spreads.keys())
        
    def analyze_card_layout(self, cards: List[Dict]) -> Tuple[str, Dict]:
        """根据卡牌布局自动识别牌阵类型"""
        if not cards:
            return "unknown", {}
            
        num_cards = len(cards)
        
        # 简单的牌阵识别逻辑
        if num_cards == 3:
            return "three_card", self._match_three_card_layout(cards)
        elif num_cards == 10:
            return "celtic_cross", self._match_celtic_cross_layout(cards)
        elif num_cards == 7:
            return "relationship", self._match_relationship_layout(cards)
        elif num_cards == 5:
            return "decision", self._match_decision_layout(cards)
        else:
            return "custom", self._create_custom_layout(cards)
            
    def _match_three_card_layout(self, cards: List[Dict]) -> Dict:
        """匹配三卡布局"""
        if len(cards) != 3:
            return {}
            
        # 按X坐标排序
        sorted_cards = sorted(cards, key=lambda c: self._extract_x_coord(c.get('position', '')))
        
        layout = {}
        position_ids = ["past", "present", "future"]
        
        for i, card in enumerate(sorted_cards):
            if i < len(position_ids):
                layout[position_ids[i]] = card
                
        return layout
        
    def _match_celtic_cross_layout(self, cards: List[Dict]) -> Dict:
        """匹配凯尔特十字布局（简化版）"""
        # 这里需要更复杂的布局识别算法
        # 暂时按顺序分配
        layout = {}
        position_ids = [
            "situation", "challenge", "distant_past", "recent_past", "possible_outcome",
            "immediate_future", "your_approach", "external_influence", "hopes_fears", "final_outcome"
        ]
        
        for i, card in enumerate(cards):
            if i < len(position_ids):
                layout[position_ids[i]] = card
                
        return layout
        
    def _match_relationship_layout(self, cards: List[Dict]) -> Dict:
        """匹配关系牌阵布局"""
        layout = {}
        position_ids = ["you", "them", "connection", "challenge", "strength", "advice", "outcome"]
        
        for i, card in enumerate(cards):
            if i < len(position_ids):
                layout[position_ids[i]] = card
                
        return layout
        
    def _match_decision_layout(self, cards: List[Dict]) -> Dict:
        """匹配决策牌阵布局"""
        layout = {}
        position_ids = ["situation", "option_a", "option_b", "advice", "outcome"]
        
        for i, card in enumerate(cards):
            if i < len(position_ids):
                layout[position_ids[i]] = card
                
        return layout
        
    def _create_custom_layout(self, cards: List[Dict]) -> Dict:
        """创建自定义布局"""
        layout = {}
        for i, card in enumerate(cards):
            layout[f"card_{i+1}"] = card
        return layout
        
    def _extract_x_coord(self, position_str: str) -> float:
        """从位置字符串提取X坐标"""
        try:
            if '(' in position_str and ')' in position_str:
                coords = position_str.strip('()').split(',')
                if len(coords) >= 2:
                    return float(coords[0].strip())
        except:
            pass
        return 0.0

def main():
    """测试牌阵系统"""
    spread_system = TarotSpreadSystem()
    
    print("🎴 塔罗牌阵系统测试")
    print("=" * 40)
    
    # 列出所有牌阵
    spreads = spread_system.list_spreads()
    print(f"可用牌阵: {spreads}")
    
    # 测试三卡展开
    three_card = spread_system.get_spread("three_card")
    if three_card:
        print(f"\n📖 {three_card.name}:")
        print(f"描述: {three_card.description}")
        
        for pos_id, position in three_card.positions.items():
            print(f"  {position.name}: {position.meaning}")
            
        print(f"\n解读指南:\n{three_card.interpretation_guide}")
    
    # 测试布局识别
    test_cards = [
        {"card_name": "愚人", "orientation": "正位", "position": "(1, 3)"},
        {"card_name": "魔法师", "orientation": "正位", "position": "(2, 3)"},
        {"card_name": "女祭司", "orientation": "逆位", "position": "(3, 3)"}
    ]
    
    spread_type, layout = spread_system.analyze_card_layout(test_cards)
    print(f"\n🔍 布局识别结果: {spread_type}")
    print(f"卡牌分配: {layout}")

if __name__ == "__main__":
    main() 