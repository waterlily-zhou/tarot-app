#!/usr/bin/env python3
"""
简化版RAG增强塔罗AI系统 - 避免兼容性问题
"""

import os
import sys
import torch
import json
import sqlite3
from pathlib import Path
from typing import List, Dict
import re

# 设置环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    print("✅ 核心依赖库导入成功")
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    sys.exit(1)

class SimpleTarotKnowledgeBase:
    """简化的塔罗知识库 - 基于关键词检索"""
    
    def __init__(self, db_path: str = "data/simple_tarot_knowledge.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建表格
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person TEXT,
                question TEXT,
                cards TEXT,
                spread TEXT,
                content TEXT,
                keywords TEXT,
                source_file TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ 简化知识库初始化完成: {self.db_path}")
    
    def extract_keywords(self, text: str) -> str:
        """提取关键词"""
        # 简单的关键词提取
        keywords = []
        
        # 塔罗牌名
        tarot_cards = [
            "愚人", "魔术师", "女教皇", "皇后", "皇帝", "教皇", "恋人", "战车", "力量", "隐士",
            "命运之轮", "正义", "倒吊人", "死神", "节制", "恶魔", "塔", "星星", "月亮", "太阳",
            "审判", "世界", "权杖", "圣杯", "宝剑", "星币", "国王", "皇后", "骑士", "侍者"
        ]
        
        # 情感词汇
        emotion_words = [
            "爱情", "事业", "财运", "健康", "学习", "家庭", "友谊", "成长", "挑战", "机会",
            "困难", "成功", "失败", "希望", "恐惧", "勇气", "智慧", "直觉", "变化", "稳定"
        ]
        
        all_keywords = tarot_cards + emotion_words
        
        for keyword in all_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return ";".join(keywords)
    
    def add_reading(self, person: str, question: str, cards: List[str], 
                   spread: str, content: str, source_file: str):
        """添加解读到知识库"""
        # 提取关键词
        full_text = f"{question} {' '.join(cards)} {content}"
        keywords = self.extract_keywords(full_text)
        
        # 存储到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO readings (person, question, cards, spread, content, keywords, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (person, question, ';'.join(cards), spread, content, keywords, source_file))
        
        conn.commit()
        conn.close()
    
    def search_similar_readings(self, person: str, question: str, 
                              cards: List[str], spread: str, top_k: int = 3) -> List[Dict]:
        """搜索相似的解读"""
        # 生成查询关键词
        query_text = f"{question} {' '.join(cards)}"
        query_keywords = self.extract_keywords(query_text).split(";")
        
        # 从数据库检索所有记录
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM readings')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # 计算关键词匹配度
        similarities = []
        for row in rows:
            stored_keywords = row[6].split(";") if row[6] else []
            
            # 计算关键词重叠
            overlap = len(set(query_keywords) & set(stored_keywords))
            total_keywords = len(set(query_keywords) | set(stored_keywords))
            
            similarity = overlap / max(total_keywords, 1) if total_keywords > 0 else 0
            
            # 如果是同一个人，增加权重
            if row[1] == person:
                similarity += 0.3
            
            similarities.append({
                'id': row[0],
                'person': row[1],
                'question': row[2],
                'cards': row[3],
                'spread': row[4],
                'content': row[5],
                'source_file': row[7],
                'similarity': similarity
            })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]

class SimpleRAGTarotAI:
    """简化的RAG增强塔罗AI"""
    
    def __init__(self):
        self.knowledge_base = SimpleTarotKnowledgeBase()
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_model(self):
        """加载训练好的模型"""
        print("🤖 加载简化RAG塔罗AI...")
        
        # 检查设备
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ 使用 Apple Silicon MPS")
        else:
            self.device = "cpu"
            print("⚠️ 使用 CPU")
        
        model_path = "./models/qwen-tarot-24gb"
        base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
        
        if not Path(model_path).exists():
            print(f"❌ 模型路径不存在: {model_path}")
            print("请先运行模型训练")
            return False
        
        try:
            # 修复LoRA配置
            adapter_config_path = Path(model_path) / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                config["inference_mode"] = False
                with open(adapter_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载base model
            print("📥 加载基础模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 加载PEFT model
            print("📥 加载微调适配器...")
            self.model = PeftModel.from_pretrained(base_model, model_path)
            
            # 移动到设备
            if self.device == "mps":
                print("🔄 移动到MPS设备...")
                self.model = self.model.to("mps")
            
            print("✅ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 建议检查模型文件是否完整")
            return False
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("📚 构建简化塔罗知识库...")
        
        # 检查是否已有数据
        conn = sqlite3.connect(self.knowledge_base.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM readings')
        count = cursor.fetchone()[0]
        conn.close()
        
        if count > 0:
            print(f"✅ 知识库已有 {count} 条记录")
            return
        
        # 从训练数据构建知识库
        data_file = "data/finetune/tarot_readings.jsonl"
        if not Path(data_file).exists():
            print(f"❌ 训练数据不存在: {data_file}")
            return
        
        count = 0
        print("📄 正在处理训练数据...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    metadata = sample['metadata']
                    
                    # 提取信息
                    person = metadata.get('person', '')
                    cards = metadata.get('cards', [])
                    spread = metadata.get('spread', '')
                    content = sample['response']
                    source_file = metadata.get('source_file', '')
                    
                    # 从指令中提取问题
                    instruction = sample['instruction']
                    question_match = re.search(r'问题：([^\n]+)', instruction)
                    question = question_match.group(1) if question_match else ''
                    
                    # 添加到知识库 (保留所有高质量的专业数据)
                    self.knowledge_base.add_reading(
                        person, question, cards, spread, content, source_file
                    )
                    
                    count += 1
                    if count % 20 == 0:
                        print(f"   已处理 {count} 条记录...")
                        
                except Exception as e:
                    print(f"⚠️ 处理记录时出错: {e}")
        
        print(f"✅ 知识库构建完成，共 {count} 条记录")
    
    def generate_enhanced_reading(self, person: str, question: str, 
                                cards: List[str], spread: str = "自由牌阵") -> str:
        """生成RAG增强的解读"""
        
        # 1. 检索相似解读
        print("🔍 检索相似解读...")
        similar_readings = self.knowledge_base.search_similar_readings(
            person, question, cards, spread, top_k=3
        )
        
        # 2. 构建增强prompt
        enhanced_prompt = self._build_enhanced_prompt(
            person, question, cards, spread, similar_readings
        )
        
        # 3. 生成解读
        print("🤖 生成增强解读...")
        generated = self._generate_with_model(enhanced_prompt)
        
        # 4. 格式化最终结果
        final_reading = self._format_final_reading(
            person, question, cards, spread, generated, similar_readings
        )
        
        return final_reading
    
    def _build_enhanced_prompt(self, person: str, question: str, cards: List[str], 
                             spread: str, similar_readings: List[Dict]) -> str:
        """构建增强prompt - 专注你的解读风格"""
        
        # 构建更专业的prompt，体现你的解读特色
        prompt = f"""专业塔罗解读任务：

基本信息：
- 咨询者：{person}
- 问题：{question}
- 牌阵：{spread}
- 抽到的牌：{';'.join(cards)}

"""
        
        # 添加更多上下文参考
        if similar_readings:
            prompt += "参考你以往的专业解读风格：\n\n"
            for i, reading in enumerate(similar_readings[:2], 1):  # 只用前2个最相关的
                if reading['similarity'] > 0.05:
                    prompt += f"参考案例{i}：\n"
                    prompt += f"咨询者: {reading['person']}\n"
                    prompt += f"问题: {reading['question']}\n"
                    prompt += f"牌组: {reading['cards']}\n"
                    
                    # 提取解读的核心部分（更多内容）
                    ref_content = reading['content']
                    # 取前800字符，保留更多专业内容
                    if len(ref_content) > 800:
                        ref_content = ref_content[:800] + "..."
                    
                    prompt += f"解读风格参考:\n{ref_content}\n\n"
        
        # 更明确的指令
        prompt += f"""请基于以上参考，为{person}提供深度专业的塔罗解读。
要求：
1. 针对具体抽到的牌：{';'.join(cards)}
2. 结合{person}的个人能量特质
3. 体现专业的解读深度和洞察力
4. 保持你一贯的解读风格

开始解读："""
        
        return prompt
    
    def _generate_with_model(self, prompt: str) -> str:
        """使用模型生成解读"""
        try:
            # 检查模型是否加载成功
            if self.model is None or self.tokenizer is None:
                return self._generate_simple_rule_based(prompt)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1200)
            if self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    top_p=0.8,
                    no_repeat_ngram_size=3
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = full_response[len(prompt):].strip()
            
            # 后处理：确保生成内容符合要求
            generated = self._post_process_generated(generated, prompt)
            
            return generated
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return self._generate_simple_rule_based(prompt)
    
    def _post_process_generated(self, generated: str, original_prompt: str) -> str:
        """后处理生成的内容"""
        # 提取提示中的牌
        import re
        prompt_cards = re.findall(r'牌：([^\\n]+)', original_prompt)
        if prompt_cards:
            actual_cards = prompt_cards[0].split(';')
            
            # 检查生成内容是否提到了正确的牌
            mentioned_correct_cards = any(card.strip() in generated for card in actual_cards)
            
            if not mentioned_correct_cards:
                # 如果没有提到正确的牌，生成基于规则的解读
                return self._generate_card_based_reading(actual_cards)
        
        # 清理生成内容
        if len(generated) < 50:
            return self._generate_simple_rule_based(original_prompt)
        
        # 移除重复内容
        lines = generated.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line)
        
        return '\n'.join(cleaned_lines)
    
    def _generate_card_based_reading(self, cards: List[str]) -> str:
        """基于卡牌生成简单解读"""
        card_meanings = {
            "愚人": {"正位": "新的开始、冒险精神、纯真", "逆位": "鲁莽、缺乏计划"},
            "力量": {"正位": "内在力量、勇气、耐心", "逆位": "软弱、缺乏自信"},
            "星币十": {"正位": "财富圆满、家庭和谐、物质成功", "逆位": "财务损失、家庭问题"},
            "圣杯二": {"正位": "爱情、伙伴关系、和谐", "逆位": "关系破裂、不和"},
            "圣杯十": {"正位": "家庭幸福、情感满足、和谐", "逆位": "家庭不和、情感空虚"},
            "宝剑七": {"正位": "策略、机智、独立行动", "逆位": "欺骗、逃避"},
            "恋人": {"正位": "爱情、选择、和谐关系", "逆位": "关系问题、错误选择"}
        }
        
        reading = "根据你抽到的牌，我为你解读如下：\n\n"
        
        for i, card in enumerate(cards, 1):
            card_name = card.replace("(正位)", "").replace("(逆位)", "").strip()
            position = "正位" if "(正位)" in card else "逆位" if "(逆位)" in card else "正位"
            
            if card_name in card_meanings:
                meaning = card_meanings[card_name].get(position, "代表着重要的人生转折")
                reading += f"{i}. **{card}**: {meaning}。"
            else:
                reading += f"{i}. **{card}**: 这张牌提醒你关注内心的声音和直觉。"
            
            reading += "\n"
        
        reading += "\n整体而言，这次抽牌显示了你当前生活中的重要议题。建议你保持开放的心态，相信自己的直觉，在面对选择时要谨慎考虑。"
        
        return reading
    
    def _generate_simple_rule_based(self, prompt: str) -> str:
        """简单的基于规则的生成"""
        return "由于技术限制，暂时无法生成详细解读。建议你重新抽牌或联系专业塔罗师进行解读。"
    
    def _is_good_quality_data(self, question: str, cards: List[str], content: str) -> bool:
        """检查数据质量"""
        # 过滤包含HTML标签或图片识别错误的数据
        if '<!--' in question or '<' in question or 'Card 1:' in question:
            return False
        
        # 过滤问题为空或过短的数据
        if not question or len(question.strip()) < 3:
            return False
        
        # 过滤卡牌信息异常的数据
        if not cards or len(cards) == 0:
            return False
        
        # 过滤内容过短的数据
        if not content or len(content.strip()) < 100:
            return False
        
        # 过滤包含过多专业术语的数据（这些对模型学习没有帮助）
        complex_terms_count = sum(1 for term in [
            '星宿', '宫位', '海王星', '天底', '北极', '南天银河', '恒星',
            '房宿', '心宿', '室宿', '奇数宫', '偶数宫', '拱向', '刑克'
        ] if term in content)
        
        if complex_terms_count > 5:  # 如果专业术语过多，跳过
            return False
        
        return True
    
    def _format_final_reading(self, person: str, question: str, cards: List[str], 
                            spread: str, generated: str, similar_readings: List[Dict]) -> str:
        """格式化最终解读"""
        
        final_reading = f"""🔮 {person}的塔罗解读

📋 咨询信息：
   问题：{question}
   牌阵：{spread}
   抽到的牌：{' | '.join(cards)}

🎯 AI解读：
{generated}

"""
        
        # 添加参考信息
        if similar_readings:
            high_sim_refs = [r for r in similar_readings if r['similarity'] > 0.2]
            if high_sim_refs:
                final_reading += "📚 相关参考：\n"
                for i, reading in enumerate(high_sim_refs[:2], 1):
                    final_reading += f"   参考{i}: {reading['person']}的{reading['question']} (匹配度: {reading['similarity']:.1%})\n"
        
        return final_reading

def main():
    print("🔮 简化版RAG增强塔罗AI系统")
    print("="*50)
    
    # 初始化系统
    rag_ai = SimpleRAGTarotAI()
    
    # 加载模型
    if not rag_ai.load_model():
        print("⚠️ 模型加载失败，将使用纯检索模式")
        return
    
    # 构建知识库
    rag_ai.build_knowledge_base()
    
    # 测试案例
    print("\n🧪 测试简化RAG解读...")
    
    test_cases = [
        {
            "person": "Mel",
            "question": "事业发展方向",
            "cards": ["愚人(正位)", "力量(正位)", "星币十(正位)"],
            "spread": "三张牌解读"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试案例 {i}")
        print('='*60)
        
        reading = rag_ai.generate_enhanced_reading(
            case["person"], case["question"], 
            case["cards"], case["spread"]
        )
        
        print(reading)
    
    # 交互模式
    print(f"\n{'='*60}")
    choice = input("是否进入交互模式？(y/n): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        print("\n🎯 进入简化RAG交互模式 (输入 'quit' 退出)")
        
        while True:
            try:
                print("\n" + "-"*40)
                person = input("咨询者姓名: ").strip()
                if person.lower() == 'quit':
                    break
                
                question = input("问题: ").strip()
                if question.lower() == 'quit':
                    break
                
                cards_input = input("抽到的牌 (用分号;分隔): ").strip()
                if cards_input.lower() == 'quit':
                    break
                
                cards = [card.strip() for card in cards_input.split(';') if card.strip()]
                
                spread = input("牌阵类型 (可选): ").strip() or "自由牌阵"
                if spread.lower() == 'quit':
                    break
                
                print("\n🔮 正在生成简化RAG解读...")
                reading = rag_ai.generate_enhanced_reading(person, question, cards, spread)
                
                print(f"\n{reading}")
                
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用简化RAG塔罗AI!")
                break
            except Exception as e:
                print(f"❌ 生成失败: {e}")
    
    print("\n🎉 简化RAG塔罗AI系统测试完成!")

if __name__ == "__main__":
    main() 