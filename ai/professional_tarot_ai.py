#!/usr/bin/env python3
"""
专业塔罗AI解牌系统
整合图片识别、牌阵分析、卡牌含义和RAG知识，提供真正专业的解牌服务
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 导入依赖系统
from rag_system import TarotRAGSystem
from tarot_spread_system import TarotSpreadSystem
from tarot_card_meanings import TarotCardDatabase, CardContext

@dataclass
class ReadingCard:
    """解牌中的单张卡牌"""
    card_name: str
    orientation: str  # "正位" or "逆位"
    position: str     # 位置坐标或描述
    position_id: str  # 在牌阵中的位置ID
    position_meaning: str  # 该位置的含义
    order: int        # 识别顺序

@dataclass
class TarotReading:
    """完整的塔罗解读"""
    cards: List[ReadingCard]
    spread_type: str
    spread_name: str
    question: str
    user_id: str
    
    # 分析结果
    spread_analysis: str
    card_analyses: Dict[str, str]
    relationship_analysis: str
    elemental_analysis: str
    overall_interpretation: str
    advice: str
    
    # 元数据
    timestamp: float
    generation_time: float
    model_used: str
    confidence_score: float

class ProfessionalTarotAI:
    """专业塔罗AI系统"""
    
    def __init__(self, 
                 llm_model: str = "qwen2.5:1.5b",
                 ollama_url: str = "http://localhost:11434"):
        
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        
        print("🔮 初始化专业塔罗AI系统...")
        
        # 初始化各个子系统
        print("📚 加载知识库...")
        self.rag = TarotRAGSystem()
        
        print("🎴 加载牌阵系统...")
        self.spread_system = TarotSpreadSystem()
        
        print("🃏 加载卡牌数据库...")
        self.card_db = TarotCardDatabase()
        
        # 测试LLM连接
        self._test_llm_connection()
        
        print("✅ 专业塔罗AI系统初始化完成！")
    
    def _test_llm_connection(self):
        """测试LLM连接"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                
                if self.llm_model in available_models:
                    print(f"✅ LLM模型可用: {self.llm_model}")
                else:
                    print(f"⚠️  模型 {self.llm_model} 未找到")
                    print(f"可用模型: {available_models}")
            else:
                print(f"❌ Ollama服务连接失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ Ollama连接测试失败: {e}")
    
    def _call_llm(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """调用本地LLM"""
        try:
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_tokens": 3000
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"LLM调用失败，状态码: {response.status_code}"
                
        except Exception as e:
            return f"LLM调用出错: {e}"
    
    def analyze_cards_from_recognition(self, recognized_cards: List[Dict]) -> List[ReadingCard]:
        """分析图片识别结果，转换为解牌格式"""
        if not recognized_cards:
            return []
        
        # 识别牌阵类型
        spread_type, layout = self.spread_system.analyze_card_layout(recognized_cards)
        spread = self.spread_system.get_spread(spread_type)
        
        reading_cards = []
        
        for card_info in recognized_cards:
            card_name = card_info.get('card_name', '')
            orientation = card_info.get('orientation', '正位')
            position = card_info.get('position', '')
            order = card_info.get('order', 0)
            
            # 确定在牌阵中的位置
            position_id = self._find_position_in_layout(card_info, layout)
            position_meaning = ""
            
            if spread and position_id in spread.positions:
                position_meaning = spread.positions[position_id].description
            else:
                position_meaning = f"第{order}张卡牌"
            
            reading_card = ReadingCard(
                card_name=card_name,
                orientation=orientation,
                position=position,
                position_id=position_id,
                position_meaning=position_meaning,
                order=order
            )
            
            reading_cards.append(reading_card)
        
        return reading_cards
    
    def _find_position_in_layout(self, card_info: Dict, layout: Dict) -> str:
        """在布局中找到卡牌对应的位置ID"""
        # 简化实现：通过order或在layout中的匹配来确定位置
        order = card_info.get('order', 0)
        
        for position_id, card_data in layout.items():
            if card_data.get('card_name') == card_info.get('card_name'):
                return position_id
        
        # 如果没有找到，使用通用位置
        position_ids = list(layout.keys())
        if order <= len(position_ids):
            return position_ids[order - 1]
        
        return f"position_{order}"
    
    def generate_professional_reading(self, 
                                    cards: List[ReadingCard],
                                    spread_type: str,
                                    question: str = None,
                                    user_id: str = None) -> TarotReading:
        """生成专业塔罗解读"""
        
        print(f"🔮 开始专业解牌...")
        print(f"牌阵: {spread_type}, 卡牌数: {len(cards)}")
        print(f"问题: {question}")
        
        start_time = time.time()
        
        # 1. 获取牌阵信息
        spread = self.spread_system.get_spread(spread_type)
        spread_name = spread.name if spread else "自定义牌阵"
        
        # 2. 分析每张卡牌
        print("🃏 分析各张卡牌...")
        card_analyses = {}
        for card in cards:
            analysis = self._analyze_single_card(card, question)
            card_analyses[card.position_id] = analysis
        
        # 3. 分析卡牌关系
        print("🔗 分析卡牌关系...")
        relationship_analysis = self._analyze_card_relationships(cards, spread)
        
        # 4. 分析元素平衡
        print("⚡ 分析元素能量...")
        elemental_analysis = self._analyze_elemental_balance(cards)
        
        # 5. 分析牌阵整体
        print("🎴 分析牌阵结构...")
        spread_analysis = self._analyze_spread_structure(cards, spread, question)
        
        # 6. 生成整体解读
        print("🧠 生成整体解读...")
        overall_interpretation = self._generate_overall_interpretation(
            cards, spread_analysis, card_analyses, relationship_analysis, 
            elemental_analysis, question, user_id
        )
        
        # 7. 生成建议
        print("💡 生成指导建议...")
        advice = self._generate_advice(cards, overall_interpretation, question)
        
        # 8. 计算置信度
        confidence_score = self._calculate_confidence_score(cards, spread_type)
        
        generation_time = time.time() - start_time
        
        # 创建解读结果
        reading = TarotReading(
            cards=cards,
            spread_type=spread_type,
            spread_name=spread_name,
            question=question or "综合运势",
            user_id=user_id or "anonymous",
            spread_analysis=spread_analysis,
            card_analyses=card_analyses,
            relationship_analysis=relationship_analysis,
            elemental_analysis=elemental_analysis,
            overall_interpretation=overall_interpretation,
            advice=advice,
            timestamp=time.time(),
            generation_time=generation_time,
            model_used=self.llm_model,
            confidence_score=confidence_score
        )
        
        # 保存用户上下文
        if user_id:
            self._save_user_context(reading)
        
        print(f"✅ 专业解牌完成 (用时 {generation_time:.2f}秒)")
        return reading
    
    def _analyze_single_card(self, card: ReadingCard, question: str = None) -> str:
        """分析单张卡牌"""
        is_upright = card.orientation == "正位"
        
        # 获取基础含义
        card_meaning = self.card_db.get_card_meaning(card.card_name, is_upright)
        position_meaning = self.card_db.get_position_meaning(
            card.card_name, card.position_id, is_upright
        )
        
        # 确定解读情境
        context = CardContext.GENERAL
        if question:
            if any(word in question for word in ["感情", "爱情", "恋爱", "关系"]):
                context = CardContext.LOVE
            elif any(word in question for word in ["事业", "工作", "职业", "事业发展"]):
                context = CardContext.CAREER
        
        context_meaning = self.card_db.get_card_meaning(card.card_name, is_upright, context)
        
        # 获取相关知识
        knowledge_context = self.rag.generate_context_for_query(
            f"{card.card_name} {card.orientation} {card.position_meaning}", 
            max_context_length=800
        )
        
        # 构建分析
        analysis = f"""
**{card.card_name} ({card.orientation})**
位置含义: {card.position_meaning}

基础解读: {card_meaning}

位置特定含义: {position_meaning}

情境解读: {context_meaning}

相关知识参考: {knowledge_context[:300] if knowledge_context else "无相关参考"}
        """
        
        return analysis.strip()
    
    def _analyze_card_relationships(self, cards: List[ReadingCard], spread) -> str:
        """分析卡牌间的关系"""
        if len(cards) < 2:
            return "单卡牌阵，无需分析卡牌关系。"
        
        # 使用卡牌数据库分析元素互动
        card_data = [{"card_name": c.card_name, "orientation": c.orientation} for c in cards]
        combination_analysis = self.card_db.analyze_card_combination(card_data)
        
        # 分析牌阵中的特定关系
        relationship_insights = []
        
        if spread and spread.relationships:
            for rel in spread.relationships:
                related_cards = []
                for pos_id in rel.positions:
                    for card in cards:
                        if card.position_id == pos_id:
                            related_cards.append(card)
                
                if len(related_cards) >= 2:
                    insight = f"**{rel.name}关系**: {rel.description}\n"
                    insight += f"相关卡牌: {', '.join([f'{c.card_name}({c.orientation})' for c in related_cards])}\n"
                    
                    # 分析这些卡牌的具体互动
                    if rel.relationship_type == "opposition":
                        insight += "这些卡牌形成对比关系，需要寻找平衡点。\n"
                    elif rel.relationship_type == "sequence":
                        insight += "这些卡牌显示发展序列，体现事物演进过程。\n"
                    elif rel.relationship_type == "support":
                        insight += "这些卡牌相互支持，增强彼此的能量。\n"
                    
                    relationship_insights.append(insight)
        
        # 分析相邻卡牌的影响
        adjacency_analysis = self._analyze_adjacent_cards(cards)
        
        return f"""
{combination_analysis}

**牌阵关系分析:**
{chr(10).join(relationship_insights) if relationship_insights else "无特殊牌阵关系"}

**位置邻近分析:**
{adjacency_analysis}
        """.strip()
    
    def _analyze_adjacent_cards(self, cards: List[ReadingCard]) -> str:
        """分析相邻卡牌的影响"""
        if len(cards) < 2:
            return "单卡无需分析邻近影响。"
        
        insights = []
        
        # 简单的相邻分析：按order排序
        sorted_cards = sorted(cards, key=lambda c: c.order)
        
        for i in range(len(sorted_cards) - 1):
            current = sorted_cards[i]
            next_card = sorted_cards[i + 1]
            
            current_energy = self.card_db.get_card_energy(current.card_name)
            next_energy = self.card_db.get_card_energy(next_card.card_name)
            
            insight = f"{current.card_name}({current_energy}) → {next_card.card_name}({next_energy})"
            insights.append(insight)
        
        return "能量流动: " + " → ".join([f"{c.card_name}" for c in sorted_cards])
    
    def _analyze_elemental_balance(self, cards: List[ReadingCard]) -> str:
        """分析元素平衡"""
        elements = {"火": 0, "水": 0, "风": 0, "土": 0}
        
        for card in cards:
            card_data = self.card_db.get_card(card.card_name)
            if card_data:
                element = card_data.element
                if element in elements:
                    elements[element] += 1
        
        total_cards = len(cards)
        analysis = "**元素分析:**\n"
        
        for element, count in elements.items():
            percentage = (count / total_cards) * 100 if total_cards > 0 else 0
            analysis += f"{element}: {count}张 ({percentage:.1f}%) "
            
            if percentage > 50:
                analysis += "- 主导元素，能量强烈\n"
            elif percentage == 0:
                analysis += "- 缺失，可能需要补充此类能量\n"
            else:
                analysis += "- 平衡存在\n"
        
        # 元素互动分析
        if elements["火"] > 0 and elements["水"] > 0:
            analysis += "\n火水并存：情感与行动之间存在张力，需要平衡激情与理性。"
        if elements["风"] > 0 and elements["土"] > 0:
            analysis += "\n风土结合：思想与实践相结合，有利于将想法具体化。"
        
        return analysis
    
    def _analyze_spread_structure(self, cards: List[ReadingCard], spread, question: str) -> str:
        """分析牌阵结构"""
        if not spread:
            return f"自定义{len(cards)}卡牌阵，按卡牌顺序解读。"
        
        analysis = f"**{spread.name}牌阵分析:**\n"
        analysis += f"{spread.description}\n\n"
        
        # 分析核心位置的卡牌
        important_positions = [pos for pos in spread.positions.values() if pos.importance >= 4]
        
        if important_positions:
            analysis += "**核心位置:**\n"
            for pos in important_positions:
                # 找到该位置的卡牌
                pos_card = None
                for card in cards:
                    if card.position_id == pos.position_id:
                        pos_card = card
                        break
                
                if pos_card:
                    analysis += f"- {pos.name} ({pos.meaning}): {pos_card.card_name} ({pos_card.orientation})\n"
        
        # 添加解读指导
        if spread.interpretation_guide:
            analysis += f"\n**解读要点:**\n{spread.interpretation_guide}"
        
        return analysis
    
    def _generate_overall_interpretation(self, cards: List[ReadingCard], 
                                       spread_analysis: str, card_analyses: Dict,
                                       relationship_analysis: str, elemental_analysis: str,
                                       question: str, user_id: str) -> str:
        """生成整体解读"""
        
        # 构建上下文
        context_query = f"{question} {' '.join([c.card_name for c in cards])}"
        rag_context = self.rag.generate_context_for_query(context_query, user_id, max_context_length=1000)
        
        # 准备解读数据
        cards_summary = []
        for card in cards:
            cards_summary.append(f"{card.card_name}({card.orientation})在{card.position_meaning}")
        
        system_prompt = """你是一位极其专业的塔罗占卜师，拥有数十年的解牌经验。你的解读特点：

1. **深度专业**: 精通塔罗象征学、数字学、占星学
2. **整体思维**: 从牌阵整体、卡牌关系、元素平衡等多维度解读
3. **个性化指导**: 结合提问者的具体情况给出针对性建议
4. **诗意表达**: 语言优美富有洞察力，避免生硬的条文式解读
5. **实用性**: 提供可行的人生指导，不是抽象的理论

解读原则：
- 将卡牌作为一个有机整体来解读，不是简单罗列
- 重点关注卡牌间的对话和能量流动
- 结合位置含义深化解读
- 考虑正逆位的具体影响
- 给出具体可行的建议

请进行深度、专业、富有洞察力的解读。"""

        user_prompt = f"""请为以下塔罗牌阵进行专业解读：

**问题**: {question or '综合运势指导'}

**卡牌组合**: {', '.join(cards_summary)}

**牌阵分析**:
{spread_analysis}

**卡牌关系分析**:
{relationship_analysis}

**元素能量分析**:
{elemental_analysis}

**相关知识背景**:
{rag_context}

请进行整体性的专业解读，重点关注：
1. 整体能量和主题
2. 卡牌间的深层对话
3. 对当前情况的洞察
4. 发展趋势和建议
5. 需要注意的要点

请用温暖、智慧、富有洞察力的语言进行解读："""

        interpretation = self._call_llm(user_prompt, system_prompt, temperature=0.8)
        return interpretation
    
    def _generate_advice(self, cards: List[ReadingCard], interpretation: str, question: str) -> str:
        """生成具体建议"""
        system_prompt = """你是一位智慧的人生导师，根据塔罗解读提供实用的人生指导。

你的建议特点：
1. **实用性强**: 给出具体可执行的行动建议
2. **积极正面**: 即使面对挑战也要给出建设性建议
3. **个性化**: 针对具体情况和问题定制建议
4. **平衡性**: 考虑不同层面（情感、理性、行动、等待等）
5. **温暖支持**: 语气充满关怀和鼓励

请提供3-5条具体的建议。"""

        cards_info = ', '.join([f"{c.card_name}({c.orientation})" for c in cards])
        
        user_prompt = f"""基于以下塔罗解读，请提供具体的人生指导建议：

**问题**: {question or '综合运势指导'}
**卡牌**: {cards_info}

**解读内容**:
{interpretation[:800]}...

请提供3-5条具体的建议，包括：
- immediate actions（即时行动）
- mindset shifts（心态调整）  
- things to watch for（需要注意的）
- long-term guidance（长期指导）

请用温暖、支持的语言表达："""

        advice = self._call_llm(user_prompt, system_prompt, temperature=0.7)
        return advice
    
    def _calculate_confidence_score(self, cards: List[ReadingCard], spread_type: str) -> float:
        """计算解读置信度"""
        score = 0.5  # 基础分数
        
        # 根据卡牌数量调整
        if len(cards) >= 3:
            score += 0.2
        elif len(cards) == 1:
            score -= 0.1
        
        # 根据牌阵类型调整
        if spread_type in ["three_card", "celtic_cross"]:
            score += 0.2
        
        # 根据卡牌识别的明确性调整
        for card in cards:
            if card.card_name and card.orientation in ["正位", "逆位"]:
                score += 0.05
        
        # 确保在0-1范围内
        return min(max(score, 0.0), 1.0)
    
    def _save_user_context(self, reading: TarotReading):
        """保存用户上下文"""
        context_data = {
            "type": "professional_reading",
            "summary": f"专业解读了{reading.spread_name}，包含{len(reading.cards)}张卡牌",
            "cards": [c.card_name for c in reading.cards],
            "themes": [reading.question] if reading.question else ["综合运势"],
            "notes": f"置信度: {reading.confidence_score:.2f}, 用时: {reading.generation_time:.2f}秒"
        }
        
        self.rag.add_user_context(reading.user_id, context_data)
    
    def format_reading_result(self, reading: TarotReading) -> str:
        """格式化输出解读结果"""
        result = f"""
{'='*80}
🔮 专业塔罗解读结果
{'='*80}

📋 **基本信息**
牌阵：{reading.spread_name} ({reading.spread_type})
问题：{reading.question}
卡牌数：{len(reading.cards)}
置信度：{reading.confidence_score:.2%}
生成时间：{reading.generation_time:.2f}秒

🎴 **卡牌组合**
"""
        
        for card in reading.cards:
            result += f"  • {card.card_name} ({card.orientation}) - {card.position_meaning}\n"
        
        result += f"""
🎯 **牌阵分析**
{reading.spread_analysis}

🔗 **关系分析**  
{reading.relationship_analysis}

⚡ **元素能量**
{reading.elemental_analysis}

📖 **整体解读**
{reading.overall_interpretation}

💡 **指导建议**
{reading.advice}

{'='*80}
        """
        
        return result.strip()

def main():
    """测试专业塔罗AI系统"""
    print("🌟 专业塔罗AI系统测试")
    print("=" * 60)
    
    # 初始化系统
    ai = ProfessionalTarotAI()
    
    # 模拟图片识别结果
    recognized_cards = [
        {"card_name": "皇后", "orientation": "正位", "position": "(1, 3)", "order": 1},
        {"card_name": "力量", "orientation": "正位", "position": "(2, 3)", "order": 2},
        {"card_name": "星币七", "orientation": "逆位", "position": "(3, 3)", "order": 3}
    ]
    
    # 转换为解牌格式
    reading_cards = ai.analyze_cards_from_recognition(recognized_cards)
    
    # 生成专业解读
    reading = ai.generate_professional_reading(
        cards=reading_cards,
        spread_type="three_card",
        question="关于个人成长和心轮能量的指导",
        user_id="mel"
    )
    
    # 显示结果
    print(ai.format_reading_result(reading))

if __name__ == "__main__":
    main() 