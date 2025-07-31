#!/usr/bin/env python3
"""
Self-Learning Tarot AI系统 - 让DeepSeek自己学习语料
"""
import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import re
import openai

class SelfLearningTarotAI:
    """让AI自己学习的塔罗系统"""
    
    def __init__(self):
        self.client = None
        self.db_path = "data/deepseek_tarot_knowledge.db"
        self.init_deepseek()
        self.init_database()
    
    def init_deepseek(self):
        """初始化DeepSeek客户端"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
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
    
    def build_knowledge_base(self):
        """构建知识库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM readings')
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"✅ 知识库已有 {count} 条记录")
            conn.close()
            return
        
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
                    
                    instruction = sample['instruction']
                    question_match = re.search(r'问题：([^\n]+)', instruction)
                    question = question_match.group(1) if question_match else ''
                    
                    cursor.execute('''
                        INSERT INTO readings (person, question, cards, spread, content, embedding, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (person, question, ';'.join(cards), spread, content, None, source_file))
                    
                    count += 1
                        
                except Exception as e:
                    print(f"⚠️ 处理记录时出错: {e}")
        
        conn.commit()
        conn.close()
        print(f"✅ 知识库构建完成，共 {count} 条记录")
    
    def get_learning_materials(self, person: str, cards: List[str], limit: int = 15):
        """获取学习材料"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        learning_materials = []
        
        # 1. 获取目标人物的所有历史解读
        cursor.execute('''
            SELECT person, question, cards, content 
            FROM readings 
            WHERE person = ?
            ORDER BY rowid DESC
        ''', (person,))
        
        person_readings = cursor.fetchall()
        
        # 2. 获取包含相同牌的其他解读（用于丰富牌意理解）
        query_cards_clean = [card.replace('(正位)', '').replace('(逆位)', '').strip() for card in cards]
        card_readings = []
        
        for card in query_cards_clean[:2]:  # 只取前2张牌避免太多数据
            cursor.execute('''
                SELECT person, question, cards, content 
                FROM readings 
                WHERE cards LIKE ? 
                ORDER BY rowid DESC
                LIMIT 5
            ''', (f'%{card}%',))
            card_readings.extend(cursor.fetchall())
        
        # 3. 随机获取一些其他解读作为背景理解
        cursor.execute('''
            SELECT person, question, cards, content 
            FROM readings 
            ORDER BY RANDOM()
            LIMIT 5
        ''', ())
        
        background_readings = cursor.fetchall()
        
        conn.close()
        
        return {
            'person_readings': person_readings,
            'card_readings': card_readings,
            'background_readings': background_readings
        }
    
    def generate_self_learning_reading(self, person: str, question: str, 
                                     cards: List[str], spread: str = "自由牌阵") -> str:
        """让DeepSeek自己学习后生成解读"""
        
        if not self.client:
            return "❌ DeepSeek 客户端未初始化"
        
        print(f"📚 为 {person} 准备学习材料...")
        materials = self.get_learning_materials(person, cards)
        
        print(f"   找到 {len(materials['person_readings'])} 个个人案例")
        print(f"   找到 {len(materials['card_readings'])} 个相关牌组案例")
        print(f"   找到 {len(materials['background_readings'])} 个背景案例")
        
        # 构建学习提示
        system_prompt = f"""你是一位专业的塔罗师。现在有丰富的塔罗解读案例供你学习，请你：

1. 自己分析和理解这些解读案例中的智慧
2. 学习每张牌在不同情境下的深层含义
3. 理解不同人的特征和模式
4. 基于你的学习和理解，为当前咨询提供解读

学习材料：

## {person}的历史解读案例：
"""
        
        # 添加个人案例
        for i, (p, q, c, content) in enumerate(materials['person_readings'], 1):
            cards_display = c.replace(';', ' | ') if c else '未记录'
            system_prompt += f"""
案例{i}：
问题：{q}
牌组：{cards_display}
解读：{content}

---
"""
        
        # 添加牌组参考案例
        if materials['card_readings']:
            system_prompt += f"\n## 相关牌组的解读案例（用于丰富牌意理解）：\n"
            
            for i, (p, q, c, content) in enumerate(materials['card_readings'][:8], 1):
                cards_display = c.replace(';', ' | ') if c else '未记录'
                system_prompt += f"""
案例{i}：
咨询者：{p}
问题：{q} 
牌组：{cards_display}
解读：{content[:400]}...

---
"""
        
        # 添加背景案例
        if materials['background_readings']:
            system_prompt += f"\n## 其他解读案例（用于理解整体风格）：\n"
            
            for i, (p, q, c, content) in enumerate(materials['background_readings'], 1):
                cards_display = c.replace(';', ' | ') if c else '未记录'
                system_prompt += f"""
案例{i}：
咨询者：{p}
问题：{q}
牌组：{cards_display}
解读：{content[:300]}...

---
"""
        
        system_prompt += f"""
现在，请基于以上所有案例：
1. 分析{person}这个人的特征、核心议题和模式
2. 理解当前牌组中每张牌的深层含义
3. 保持你原有的专业解牌能力
4. 结合学到的个人化理解，提供深度解读

不要机械引用案例，而是内化这些智慧后自然地解读。
"""
        
        # 构建用户提示
        cards_str = ' | '.join(cards)
        user_prompt = f"""请为以下咨询提供塔罗解读：

咨询者：{person}
问题：{question}
牌阵：{spread}
抽到的牌：{cards_str}

请基于你从学习材料中获得的理解进行解读。"""
        
        print("🤖 DeepSeek正在学习和分析...")
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2500,
                temperature=0.7,
                top_p=0.9
            )
            
            reading_result = response.choices[0].message.content.strip()
            
            return f"""🔮 {person}的塔罗解读

📋 咨询信息：
   问题：{question}
   牌阵：{spread}
   抽到的牌：{cards_str}

🎯 解读：
{reading_result}

🧠 解读基础：
   基于{len(materials['person_readings'])}个个人案例、{len(materials['card_readings'])}个牌组案例的自主学习
   由 DeepSeek R1 分析理解后生成"""
            
        except Exception as e:
            return f"❌ 生成解读失败: {e}"

def main():
    print("✅ 依赖库导入成功")
    print("🔮 Self-Learning Tarot AI系统")
    print("=" * 50)
    
    ai_system = SelfLearningTarotAI()
    
    if not ai_system.client:
        print("❌ DeepSeek 客户端初始化失败")
        return
    
    ai_system.build_knowledge_base()
    
    print("\n🧪 测试自主学习解读...\n")
    print("=" * 60)
    print("测试案例 1")
    print("=" * 60)
    
    result = ai_system.generate_self_learning_reading(
        person="Mel",
        question="当前的内在成长状态",
        cards=["愚人(正位)", "力量(正位)", "星币十(正位)"],
        spread="内在探索牌阵"
    )
    
    print(result)
    
    print("\n" + "=" * 60)
    
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
        
        print("\n🔮 DeepSeek正在学习和生成解读中...\n")
        result = ai_system.generate_self_learning_reading(person, question, cards, spread)
        print(result)
        print("\n" + "=" * 60)
    
    print("\n🎉 Self-Learning Tarot AI 测试完成!")

if __name__ == "__main__":
    main() 