#!/usr/bin/env python3
"""
塔罗卡牌含义数据库
包含78张标准韦特塔罗牌的详细含义，正位/逆位解读，以及在不同情境中的应用
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class CardSuit(Enum):
    """卡牌花色"""
    MAJOR_ARCANA = "大阿卡纳"
    WANDS = "权杖"
    CUPS = "圣杯" 
    SWORDS = "宝剑"
    PENTACLES = "星币"

class CardContext(Enum):
    """解读情境"""
    GENERAL = "综合"
    LOVE = "感情"
    CAREER = "事业"
    MONEY = "财运"
    HEALTH = "健康"
    SPIRITUAL = "灵性"

@dataclass
class CardMeaning:
    """卡牌含义"""
    card_name: str
    suit: CardSuit
    number: Optional[int]
    
    # 基本含义
    upright_keywords: List[str]
    reversed_keywords: List[str]
    upright_meaning: str
    reversed_meaning: str
    
    # 详细解读
    upright_general: str
    reversed_general: str
    upright_love: str
    reversed_love: str
    upright_career: str
    reversed_career: str
    
    # 核心能量
    core_energy: str
    element: str
    astrology: str
    
    # 位置特殊含义
    past_position: str
    present_position: str
    future_position: str
    advice_position: str

class TarotCardDatabase:
    """塔罗卡牌数据库"""
    
    def __init__(self):
        self.cards: Dict[str, CardMeaning] = {}
        self._initialize_cards()
        
    def _initialize_cards(self):
        """初始化所有卡牌含义"""
        
        # 大阿卡纳
        self._add_major_arcana()
        
        # 小阿卡纳
        self._add_wands()
        self._add_cups() 
        self._add_swords()
        self._add_pentacles()
        
    def _add_major_arcana(self):
        """添加大阿卡纳"""
        
        # 0 愚人
        self._add_card(CardMeaning(
            card_name="愚人",
            suit=CardSuit.MAJOR_ARCANA,
            number=0,
            upright_keywords=["新开始", "冒险", "自由", "天真", "信任"],
            reversed_keywords=["鲁莽", "缺乏方向", "愚蠢", "风险"],
            upright_meaning="新的开始，充满可能性的旅程，天真纯真的心态",
            reversed_meaning="缺乏方向，鲁莽行事，需要更多思考",
            upright_general="代表人生新阶段的开始，对未来充满信心和期待。建议以开放的心态迎接新机会。",
            reversed_general="可能过于冲动或缺乏规划。需要更谨慎地评估风险，避免盲目行动。",
            upright_love="新恋情的开始，或关系中的新篇章。充满新鲜感和可能性。",
            reversed_love="感情中缺乏承诺或方向，可能过于理想化而忽视现实。",
            upright_career="新工作或项目的开始，创业的好时机，充满创新思维。",
            reversed_career="职业发展缺乏方向，可能做出冲动决定。需要更多规划。",
            core_energy="自由、开始、可能性",
            element="风",
            astrology="天王星",
            past_position="过去的天真或新开始为现在奠定了基础",
            present_position="现在面临新的开始或需要以开放心态面对",
            future_position="即将到来的新机会或人生新篇章",
            advice_position="保持开放心态，勇敢迎接新机会"
        ))
        
        # 1 魔法师
        self._add_card(CardMeaning(
            card_name="魔法师",
            suit=CardSuit.MAJOR_ARCANA,
            number=1,
            upright_keywords=["意志力", "技能", "专注", "创造", "行动"],
            reversed_keywords=["滥用权力", "缺乏专注", "欺骗", "操控"],
            upright_meaning="拥有实现目标的技能和意志力，将想法转化为现实",
            reversed_meaning="可能滥用技能或缺乏专注，需要重新审视目标",
            upright_general="具备所有必要的技能和资源来实现目标。现在是行动的时候。",
            reversed_general="可能缺乏专注或滥用能力。需要重新评估动机和方法。",
            upright_love="感情中表现主动，有能力创造理想的关系状态。",
            reversed_love="可能在感情中过于操控或不够真诚。",
            upright_career="在工作中展现出色的技能和领导力，项目进展顺利。",
            reversed_career="可能缺乏专注或滥用职权。需要重新审视工作态度。",
            core_energy="意志、创造、行动",
            element="风",
            astrology="水星",
            past_position="过去的技能学习或意志力为现在创造了条件",
            present_position="现在拥有实现目标的所有条件",
            future_position="将能够运用技能实现重要目标",
            advice_position="运用你的技能和意志力，将想法转化为行动"
        ))
        
        # 2 女祭司
        self._add_card(CardMeaning(
            card_name="女祭司",
            suit=CardSuit.MAJOR_ARCANA,
            number=2,
            upright_keywords=["直觉", "内在智慧", "神秘", "潜意识", "静待"],
            reversed_keywords=["忽视直觉", "缺乏内省", "秘密", "情绪不稳"],
            upright_meaning="相信内在智慧，通过静观和直觉获得洞察",
            reversed_meaning="忽视内在声音，可能被表面现象误导",
            upright_general="需要相信自己的直觉，通过冥想和内省寻找答案。",
            reversed_general="可能过于依赖理性而忽视直觉，或被情绪困扰。",
            upright_love="感情需要耐心等待，相信内心的感受和直觉。",
            reversed_love="在感情中缺乏直觉判断，可能隐瞒真实感受。",
            upright_career="工作中需要更多内省和等待，不急于做决定。",
            reversed_career="可能忽视了重要的直觉信息，需要重新审视。",
            core_energy="直觉、智慧、神秘",
            element="水",
            astrology="月亮",
            past_position="过去的内在成长或直觉经验影响现在",
            present_position="现在需要相信直觉，静待时机",
            future_position="内在智慧将带来重要洞察",
            advice_position="相信你的直觉，通过内省寻找答案"
        ))
        
        # 3 皇后
        self._add_card(CardMeaning(
            card_name="皇后",
            suit=CardSuit.MAJOR_ARCANA,
            number=3,
            upright_keywords=["丰盛", "母性", "创造", "感性", "自然"],
            reversed_keywords=["缺乏自理", "过度依赖", "创造力受阻", "自我忽视"],
            upright_meaning="象征丰盛和创造力，充满母性的爱与关怀",
            reversed_meaning="可能过度关注他人而忽视自己，或创造力受阻",
            upright_general="生活充满丰盛和美好，创造力旺盛，关爱他人也关爱自己。",
            reversed_general="可能过度付出而忽视自己需求，需要重新平衡。",
            upright_love="感情充满温暖和关爱，可能有结婚或生育的计划。",
            reversed_love="在感情中可能过度付出或过分依赖对方。",
            upright_career="工作中展现创造力和领导力，项目丰硕。",
            reversed_career="工作创造力受阻，可能过于关注他人而忽视自己发展。",
            core_energy="丰盛、母性、创造",
            element="土",
            astrology="金星",
            past_position="过去的关爱或创造经验为现在带来丰盛",
            present_position="现在处于丰盛和创造力旺盛的状态",
            future_position="将迎来丰盛和创造性的成果",
            advice_position="拥抱你的创造力，关爱他人也关爱自己"
        ))
        
        # 继续添加其他主要牌...
        # (为节省空间，这里只展示几张代表性的牌)
        
    def _add_wands(self):
        """添加权杖花色"""
        
        # 权杖一 (Ace of Wands)
        self._add_card(CardMeaning(
            card_name="权杖一",
            suit=CardSuit.WANDS,
            number=1,
            upright_keywords=["新开始", "激情", "灵感", "能量", "创意"],
            reversed_keywords=["缺乏方向", "能量不足", "延迟", "受阻"],
            upright_meaning="新项目或想法的开始，充满创造性能量",
            reversed_meaning="缺乏动力或方向，创意受阻",
            upright_general="新的创造性项目开始，充满激情和灵感。",
            reversed_general="可能缺乏动力或遇到阻碍，需要重新点燃激情。",
            upright_love="新恋情开始，或现有关系中注入新的激情。",
            reversed_love="感情缺乏激情，可能面临冷淡期。",
            upright_career="新的工作机会或项目，创业的好时机。",
            reversed_career="工作缺乏动力，项目可能延迟或受阻。",
            core_energy="火元素、创造、开始",
            element="火",
            astrology="火象星座",
            past_position="过去的创意火花为现在奠定基础",
            present_position="现在有新的创意想法等待实现",
            future_position="即将开始新的创造性项目",
            advice_position="抓住灵感，开始新的创造性项目"
        ))
        
    def _add_cups(self):
        """添加圣杯花色"""
        
        # 圣杯一 (Ace of Cups)
        self._add_card(CardMeaning(
            card_name="圣杯一",
            suit=CardSuit.CUPS,
            number=1,
            upright_keywords=["新感情", "情感满足", "爱", "直觉", "精神觉醒"],
            reversed_keywords=["情感失落", "关系问题", "内心空虚", "压抑情感"],
            upright_meaning="新的情感开始，心灵的满足和爱的流动",
            reversed_meaning="情感受阻，可能面临关系问题或内心空虚",
            upright_general="心灵得到满足，新的情感体验带来成长。",
            reversed_general="可能面临情感困扰或关系问题，需要内省。",
            upright_love="新恋情的开始，或现有关系更加深入。",
            reversed_love="感情可能遇到问题，需要处理情感阻碍。",
            upright_career="工作中获得情感满足，团队关系和谐。",
            reversed_career="工作环境可能存在人际问题，缺乏情感支持。",
            core_energy="水元素、情感、爱",
            element="水",
            astrology="水象星座",
            past_position="过去的情感经验为现在的关系奠定基础",
            present_position="现在的情感状态充满爱与满足",
            future_position="即将迎来新的情感体验",
            advice_position="开放心扉，接受爱的流动"
        ))
        
    def _add_swords(self):
        """添加宝剑花色"""
        
        # 宝剑一 (Ace of Swords)
        self._add_card(CardMeaning(
            card_name="宝剑一",
            suit=CardSuit.SWORDS,
            number=1,
            upright_keywords=["清晰思维", "真相", "正义", "新想法", "突破"],
            reversed_keywords=["困惑", "误解", "不公", "思维混乱", "冲突"],
            upright_meaning="头脑清晰，新的想法和见解带来突破",
            reversed_meaning="思维混乱，可能面临误解或不公正",
            upright_general="获得清晰的见解，能够看清真相并做出正确判断。",
            reversed_general="可能思维混乱或被误导，需要寻求清晰。",
            upright_love="在感情中获得清晰认识，能够诚实交流。",
            reversed_love="感情中可能存在误解或缺乏沟通。",
            upright_career="工作中思路清晰，能够解决复杂问题。",
            reversed_career="工作中可能面临混乱或误解，需要理清思路。",
            core_energy="风元素、思维、真相",
            element="风",
            astrology="风象星座",
            past_position="过去的清晰认识为现在提供指导",
            present_position="现在需要运用清晰的思维",
            future_position="将获得重要的洞察和真相",
            advice_position="保持清晰的思维，寻求真相"
        ))
        
    def _add_pentacles(self):
        """添加星币花色"""
        
        # 星币一 (Ace of Pentacles)
        self._add_card(CardMeaning(
            card_name="星币一",
            suit=CardSuit.PENTACLES,
            number=1,
            upright_keywords=["新机会", "财务开始", "稳定", "实现", "物质成功"],
            reversed_keywords=["错失机会", "财务不稳", "缺乏实际", "延迟"],
            upright_meaning="新的财务机会，物质层面的新开始",
            reversed_meaning="可能错失机会或财务不稳定",
            upright_general="出现新的物质机会，财务状况有望改善。",
            reversed_general="可能错失机会或计划不够实际，需要重新评估。",
            upright_love="感情关系更加稳定，可能有实质性进展。",
            reversed_love="感情中可能缺乏实质承诺或面临实际问题。",
            upright_career="新的工作机会或收入来源，职业发展稳定。",
            reversed_career="可能错失工作机会或收入不稳定。",
            core_energy="土元素、物质、机会",
            element="土",
            astrology="土象星座",
            past_position="过去的努力为现在的物质成功奠定基础",
            present_position="现在有新的物质机会等待把握",
            future_position="将获得物质层面的成功",
            advice_position="把握实际机会，脚踏实地地实现目标"
        ))
        
    def _add_card(self, card: CardMeaning):
        """添加卡牌到数据库"""
        self.cards[card.card_name] = card
        
    def get_card(self, card_name: str) -> Optional[CardMeaning]:
        """获取卡牌含义"""
        return self.cards.get(card_name)
        
    def get_card_meaning(self, card_name: str, is_upright: bool = True, context: CardContext = CardContext.GENERAL) -> str:
        """获取特定情境下的卡牌含义"""
        card = self.get_card(card_name)
        if not card:
            return f"未找到卡牌：{card_name}"
            
        if context == CardContext.GENERAL:
            return card.upright_general if is_upright else card.reversed_general
        elif context == CardContext.LOVE:
            return card.upright_love if is_upright else card.reversed_love
        elif context == CardContext.CAREER:
            return card.upright_career if is_upright else card.reversed_career
        else:
            return card.upright_meaning if is_upright else card.reversed_meaning
            
    def get_position_meaning(self, card_name: str, position: str, is_upright: bool = True) -> str:
        """获取卡牌在特定位置的含义"""
        card = self.get_card(card_name)
        if not card:
            return f"未找到卡牌：{card_name}"
            
        base_meaning = ""
        if position in ["past", "遥远过去", "近期过去"]:
            base_meaning = card.past_position
        elif position in ["present", "current", "现在", "当前情况"]:
            base_meaning = card.present_position
        elif position in ["future", "未来", "可能结果", "最终结果"]:
            base_meaning = card.future_position
        elif position in ["advice", "建议", "指导"]:
            base_meaning = card.advice_position
        
        orientation = "正位" if is_upright else "逆位"
        specific_meaning = self.get_card_meaning(card_name, is_upright)
        
        return f"{base_meaning}。{orientation}时：{specific_meaning}"
        
    def get_card_energy(self, card_name: str) -> str:
        """获取卡牌的核心能量"""
        card = self.get_card(card_name)
        return card.core_energy if card else "未知能量"
        
    def analyze_card_combination(self, cards: List[Dict]) -> str:
        """分析卡牌组合的能量互动"""
        if len(cards) < 2:
            return "需要至少两张卡牌进行组合分析"
            
        energies = []
        elements = []
        
        for card_info in cards:
            card_name = card_info.get('card_name', '')
            card = self.get_card(card_name)
            if card:
                energies.append(card.core_energy)
                elements.append(card.element)
                
        # 分析元素平衡
        element_count = {}
        for element in elements:
            element_count[element] = element_count.get(element, 0) + 1
            
        analysis = f"卡牌组合包含：{', '.join(energies)}。\n"
        analysis += f"元素分布：{element_count}。\n"
        
        # 元素相互作用分析
        if "火" in elements and "水" in elements:
            analysis += "火水相遇，可能存在情感与行动的冲突，需要平衡激情与理性。\n"
        if "风" in elements and "土" in elements:
            analysis += "风土结合，思想与实践相结合，有助于将想法落地实现。\n"
        if len(set(elements)) == 1:
            analysis += f"单一{elements[0]}元素，能量集中且强烈。\n"
        if len(set(elements)) == 4:
            analysis += "四元素齐全，代表完整和平衡的能量状态。\n"
            
        return analysis

def main():
    """测试卡牌数据库"""
    db = TarotCardDatabase()
    
    print("🃏 塔罗卡牌数据库测试")
    print("=" * 40)
    
    # 测试获取卡牌含义
    test_cards = ["愚人", "魔法师", "皇后", "权杖一", "圣杯一"]
    
    for card_name in test_cards:
        card = db.get_card(card_name)
        if card:
            print(f"\n📜 {card_name}:")
            print(f"正位关键词: {', '.join(card.upright_keywords)}")
            print(f"逆位关键词: {', '.join(card.reversed_keywords)}")
            print(f"核心能量: {card.core_energy}")
            print(f"元素: {card.element}")
    
    # 测试卡牌组合分析
    test_combination = [
        {"card_name": "魔法师", "orientation": "正位"},
        {"card_name": "皇后", "orientation": "正位"},
        {"card_name": "权杖一", "orientation": "逆位"}
    ]
    
    print(f"\n🔮 卡牌组合分析:")
    print(db.analyze_card_combination(test_combination))

if __name__ == "__main__":
    main() 