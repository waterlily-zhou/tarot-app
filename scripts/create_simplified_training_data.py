#!/usr/bin/env python3
"""
创建简化版训练数据 - 保留核心解牌风格但简化复杂内容
"""

import json
import re
from pathlib import Path

class TarotDataSimplifier:
    def __init__(self):
        self.input_file = "data/finetune/tarot_readings.jsonl"
        self.output_file = "data/finetune/tarot_readings_simplified.jsonl"
    
    def simplify_response(self, response: str) -> str:
        """简化解读回应，保留核心但去除过于复杂的部分"""
        
        # 1. 移除过长的占星学解释
        response = re.sub(r'太阳.*?星系.*?能量.*?\。', '', response)
        response = re.sub(r'南天银河.*?母性能量.*?\。', '', response)
        response = re.sub(r'\d+\s*~\s*\d+\s*宫.*?\。', '', response)
        
        # 2. 简化专业术语
        response = response.replace('灵力值', '内在力量')
        response = response.replace('磁场', '能量场')
        response = response.replace('baby牌', '内在纯真')
        response = response.replace('四大元素', '能量元素')
        
        # 3. 保留核心牌意但简化表达
        # 保留直接的牌名分析
        
        # 4. 移除过长的段落 (超过500字符的段落)
        paragraphs = response.split('。')
        simplified_paragraphs = []
        
        for para in paragraphs:
            if len(para.strip()) > 0:
                if len(para) <= 500:
                    simplified_paragraphs.append(para.strip())
                else:
                    # 对超长段落进行进一步简化
                    # 保留前200字符作为核心观点
                    core_idea = para[:200].strip()
                    if core_idea:
                        simplified_paragraphs.append(core_idea)
        
        # 5. 重新组合
        simplified = '。'.join(simplified_paragraphs)
        
        # 6. 确保长度适中 (2000字符内)
        if len(simplified) > 2000:
            simplified = simplified[:2000]
            # 找到最后一个句号，优雅截断
            last_period = simplified.rfind('。')
            if last_period > len(simplified) * 0.8:
                simplified = simplified[:last_period + 1]
        
        return simplified.strip()
    
    def extract_core_insights(self, response: str) -> str:
        """提取核心洞察，重新组织为清晰结构"""
        
        # 尝试提取关键模式
        core_points = []
        
        # 寻找牌名相关的解释
        card_patterns = [
            r'(愚人|力量|星币|宝剑|圣杯|权杖|魔法师|皇帝|皇后|隐者|正义|倒吊人|死神|节制|恶魔|塔|星星|月亮|太阳|审判|世界).*?([。！？])',
            r'(正位|逆位).*?代表.*?([。！？])',
            r'这张牌.*?([。！？])',
            r'牌.*?象征.*?([。！？])'
        ]
        
        for pattern in card_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if isinstance(match, tuple):
                    sentence = ''.join(match)
                else:
                    sentence = match
                
                if len(sentence) < 200 and sentence not in core_points:
                    core_points.append(sentence)
        
        # 寻找建议性内容
        advice_patterns = [
            r'建议.*?([。！？])',
            r'需要.*?([。！？])',
            r'可以.*?([。！？])',
            r'应该.*?([。！？])'
        ]
        
        for pattern in advice_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if len(match) < 150 and match not in core_points:
                    core_points.append(match)
        
        if core_points:
            return '。'.join(core_points[:5])  # 只取前5个核心点
        else:
            # 如果没有找到模式，使用简化方法
            return self.simplify_response(response)
    
    def create_structured_response(self, cards: list, response: str, person: str) -> str:
        """创建结构化的回应"""
        
        # 提取核心洞察
        core_content = self.extract_core_insights(response)
        
        # 创建结构化格式
        structured = f"这是关于{person}的塔罗解读。\n\n"
        
        if cards:
            # 为每张主要牌创建简短解释
            main_cards = cards[:3]  # 只分析前3张牌
            for i, card in enumerate(main_cards, 1):
                if card in core_content:
                    # 提取与这张牌相关的内容
                    card_specific = re.search(f'{re.escape(card)}.*?([。！？])', core_content)
                    if card_specific:
                        structured += f"{i}. {card}: {card_specific.group()}\n\n"
                else:
                    # 生成通用解释框架
                    structured += f"{i}. {card}: 这张牌在当前情况下显示了重要的能量。\n\n"
        
        # 添加核心洞察
        if core_content:
            structured += f"整体来看：{core_content[:500]}"
        
        return structured
    
    def process_data(self):
        """处理所有数据"""
        print("🔄 开始简化训练数据...")
        
        input_path = Path(self.input_file)
        if not input_path.exists():
            print(f"❌ 输入文件不存在: {self.input_file}")
            return
        
        simplified_samples = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    sample = json.loads(line)
                    
                    # 获取基本信息
                    person = sample['metadata']['person']
                    cards = sample['metadata']['cards']
                    original_response = sample['response']
                    
                    # 跳过过短的回应
                    if len(original_response) < 100:
                        continue
                    
                    # 简化回应
                    simplified_response = self.create_structured_response(
                        cards, original_response, person
                    )
                    
                    # 创建新样本
                    new_sample = {
                        "instruction": sample['instruction'],
                        "response": simplified_response,
                        "metadata": sample['metadata'].copy()
                    }
                    new_sample['metadata']['simplified'] = True
                    
                    simplified_samples.append(new_sample)
                    
                    print(f"✅ 处理样本 {i+1}: {sample['metadata']['title'][:50]}...")
                    
                except Exception as e:
                    print(f"❌ 处理样本 {i+1} 时出错: {e}")
        
        # 保存简化数据
        output_path = Path(self.output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in simplified_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"🎉 简化完成！")
        print(f"   原始样本: {i+1}")
        print(f"   简化样本: {len(simplified_samples)}")
        print(f"   输出文件: {self.output_file}")
        
        # 显示示例
        if simplified_samples:
            print(f"\n📖 简化示例:")
            example = simplified_samples[0]
            print(f"指令: {example['instruction'][:100]}...")
            print(f"回应: {example['response'][:200]}...")

def main():
    simplifier = TarotDataSimplifier()
    simplifier.process_data()

if __name__ == "__main__":
    main() 