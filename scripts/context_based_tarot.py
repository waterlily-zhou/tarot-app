#!/usr/bin/env python3
"""
Direct Learning 塔罗AI系统 - 直接学习完整训练数据
"""
import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import re
import openai

class PersonContext:
    """个人Context管理"""
    def __init__(self, person: str, db_path: str):
        self.person = person
        self.db_path = db_path
        self.readings_history = []
    
    def load_person_context(self):
        """加载个人所有历史记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT question, cards, spread, content, source_file 
            FROM readings 
            WHERE person = ?
            ORDER BY rowid DESC
        ''', (self.person,))
        
        rows = cursor.fetchall()
        self.readings_history = []
        
        for row in rows:
            self.readings_history.append({
                'question': row[0],
                'cards': row[1].split(';') if row[1] else [],
                'spread': row[2],
                'content': row[3],
                'source_file': row[4]
            })
        
        conn.close()
        return len(self.readings_history)
    
    def get_recent_readings(self, limit: int = 3) -> List[Dict]:
        """获取最近几次解读作为个人context"""
        return self.readings_history[:limit]

class DirectLearningTarotAI:
    """直接学习完整训练数据的塔罗AI系统"""
    
    def __init__(self):
        self.client = None
        self.db_path = "data/deepseek_tarot_knowledge.db"
        self.training_examples = []
        self.init_deepseek()
        self.init_database()
    
    def init_deepseek(self):
        """初始化DeepSeek客户端"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            # 尝试从.env.local读取
            env_file = Path(".env.local")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("DEEPSEEK_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
        
        if not api_key:
            print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
            print("export DEEPSEEK_API_KEY='your-api-key'")
            print("🔗 获取API Key: https://platform.deepseek.com/api_keys")
            return False
        
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        print("✅ DeepSeek R1 客户端初始化成功")
        return True
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person TEXT,
                question TEXT,
                cards TEXT,
                spread TEXT,
                content TEXT,
                embedding BLOB,
                source_file TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ 知识库初始化完成: {self.db_path}")
    
    def load_all_training_examples(self):
        """加载所有训练数据作为学习范例"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT person, question, cards, spread, content, source_file FROM readings')
        rows = cursor.fetchall()
        
        self.training_examples = []
        for row in rows:
            self.training_examples.append({
                'person': row[0],
                'question': row[1], 
                'cards': row[2].split(';') if row[2] else [],
                'spread': row[3],
                'content': row[4],
                'source_file': row[5]
            })
        
        conn.close()
        print(f"✅ 加载了 {len(self.training_examples)} 个完整训练案例")
        return len(self.training_examples)
    
    def build_knowledge_base(self):
        """构建知识库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查是否已有数据
        cursor.execute('SELECT COUNT(*) FROM readings')
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"✅ 知识库已有 {count} 条记录")
            conn.close()
            return
        
        # 从JSONL文件加载数据
        jsonl_file = "data/finetune/tarot_readings.jsonl"
        if not Path(jsonl_file).exists():
            print(f"❌ 训练数据文件不存在: {jsonl_file}")
            return
        
        print("📚 构建知识库...")
        count = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    metadata = sample['metadata']
                    
                    person = metadata.get('person', '')
                    cards = metadata.get('cards', [])
                    spread = metadata.get('spread', '')
                    content = sample['response']
                    source_file = metadata.get('source_file', '')
                    
                    # 从指令中提取问题
                    instruction = sample['instruction']
                    question_match = re.search(r'问题：([^\n]+)', instruction)
                    question = question_match.group(1) if question_match else ''
                    
                    # 存储到数据库
                    cursor.execute('''
                        INSERT INTO readings (person, question, cards, spread, content, embedding, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (person, question, ';'.join(cards), spread, content, 
                          None, source_file))
                    
                    count += 1
                    if count % 20 == 0:
                        print(f"   已处理 {count} 条记录...")
                        
                except Exception as e:
                    print(f"⚠️ 处理记录时出错: {e}")
        
        conn.commit()
        conn.close()
        print(f"✅ 知识库构建完成，共 {count} 条记录")
    
    def generate_direct_learning_reading(self, person: str, question: str, 
                                       cards: List[str], spread: str = "自由牌阵") -> str:
        """基于完整训练数据直接学习生成解读"""
        
        if not self.client:
            return "❌ DeepSeek 客户端未初始化"
        
        # 1. 加载个人context
        print(f"📋 加载 {person} 的个人context...")
        person_context = PersonContext(person, self.db_path)
        history_count = person_context.load_person_context()
        recent_readings = person_context.get_recent_readings(3)
        
        print(f"   找到 {history_count} 条历史记录")
        
        # 2. 加载所有训练案例
        if not self.training_examples:
            self.load_all_training_examples()
        
        # 3. 选择最相关的训练案例（包括同人案例和相似牌组案例）
        relevant_examples = []
        
        # 同人案例（最重要）
        same_person_examples = [ex for ex in self.training_examples if ex['person'] == person]
        relevant_examples.extend(same_person_examples[:5])  # 最多5个同人案例
        
        # 相似牌组案例
        query_cards_set = set([card.strip().replace('(正位)', '').replace('(逆位)', '') for card in cards])
        similar_examples = []
        for ex in self.training_examples:
            if ex['person'] != person:  # 排除同人案例
                ex_cards_set = set([card.strip().replace('(正位)', '').replace('(逆位)', '') for card in ex['cards']])
                overlap = len(query_cards_set & ex_cards_set)
                if overlap >= 1:  # 至少有一张牌重叠
                    similar_examples.append((ex, overlap))
        
        # 按重叠度排序，取前3个
        similar_examples.sort(key=lambda x: x[1], reverse=True)
        relevant_examples.extend([ex[0] for ex in similar_examples[:3]])
        
        # 4. 构建系统提示 - 包含完整的训练案例
        system_prompt = f"""你是一位专业的塔罗师，请直接学习以下真实的解读案例，掌握其中的解读风格、深度和专业术语运用。

以下是 {len(relevant_examples)} 个真实的解读案例，请仔细学习其解读思路、语言风格和专业水准：

"""
        
        for i, example in enumerate(relevant_examples, 1):
            cards_str = ' | '.join(example['cards']) if example['cards'] else '未记录'
            system_prompt += f"""
【案例 {i}】
咨询者：{example['person']}
问题：{example['question']}
牌阵：{example['spread']}
抽到的牌：{cards_str}

专业解读：
{example['content']}

---
"""
        
        system_prompt += """
请完全按照以上案例的风格、深度和专业水准来进行解读。注意：
- 使用相同的专业术语（如"锚定"、"动能"、"消耗态"、"元素"、"层面"等）
- 保持相同的分析深度和洞察力
- 结合占星学、能量工作等多重维度
- 区分正位/逆位的精准含义
- 关注人格层面和灵魂层面的双重显现
"""
        
        # 5. 构建用户提示
        cards_str = ' | '.join(cards)
        recent_context = ""
        if recent_readings:
            recent_context = f"\n{person}的最近解读历史：\n"
            for i, reading in enumerate(recent_readings, 1):
                recent_cards = ' | '.join(reading['cards']) if reading['cards'] else '未记录'
                recent_context += f"{i}. {reading['question']} - {recent_cards}\n"
        
        user_prompt = f"""请为以下咨询提供专业的塔罗解读：

咨询者：{person}
问题：{question}
牌阵：{spread}  
抽到的牌：{cards_str}
{recent_context}
请按照你学习的案例风格，提供深度专业的解读。"""
        
        # 6. 调用DeepSeek API
        print("🤖 使用 DeepSeek R1 基于完整训练数据生成解读...")
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9
            )
            
            reading_result = response.choices[0].message.content.strip()
            
            return f"""🔮 {person}的专业塔罗解读

📋 咨询信息：
   问题：{question}
   牌阵：{spread}
   抽到的牌：{cards_str}

👤 个人Context：
   历史解读：{history_count} 次
   参考案例：{len(relevant_examples)} 个相关解读

🎯 专业解读：
{reading_result}

🧠 解读基础：
   基于 {len(self.training_examples)} 个完整训练案例直接学习
   重点参考同咨询者历史案例和相似牌组案例
   由 DeepSeek R1 深度推理生成"""
            
        except Exception as e:
            return f"❌ 生成解读失败: {e}"

def main():
    print("✅ 依赖库导入成功")
    print("🔮 Direct Learning 塔罗AI系统")
    print("=" * 50)
    
    # 初始化系统
    ai_system = DirectLearningTarotAI()
    
    if not ai_system.client:
        print("❌ DeepSeek 客户端初始化失败")
        return
    
    # 构建知识库
    ai_system.build_knowledge_base()
    
    # 加载训练案例
    ai_system.load_all_training_examples()
    
    # 测试案例
    print("\n🧪 测试 Direct Learning 解读...\n")
    print("=" * 60)
    print("测试案例 1")
    print("=" * 60)
    
    result = ai_system.generate_direct_learning_reading(
        person="Mel",
        question="当前的内在成长状态",
        cards=["愚人(正位)", "力量(正位)", "星币十(正位)"],
        spread="内在探索牌阵"
    )
    
    print(result)
    
    print("\n" + "=" * 60)
    
    # 交互模式
    while True:
        choice = input("是否进入交互模式？(y/n): ").strip().lower()
        if choice != 'y':
            break
            
        print("\n📝 请输入解读信息：")
        person = input("咨询者: ").strip()
        question = input("问题: ").strip()
        cards_input = input("牌组 (用|分隔): ").strip()
        spread = input("牌阵: ").strip() or "自由牌阵"
        
        if not all([person, question, cards_input]):
            print("⚠️ 请填写完整信息")
            continue
            
        cards = [card.strip() for card in cards_input.split('|')]
        
        print("\n🔮 生成解读中...\n")
        result = ai_system.generate_direct_learning_reading(person, question, cards, spread)
        print(result)
        print("\n" + "=" * 60)
    
    print("\n🎉 Direct Learning 塔罗AI 测试完成!")

if __name__ == "__main__":
    main() 