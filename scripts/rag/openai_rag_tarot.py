#!/usr/bin/env python3
"""
OpenAI + RAG 塔罗AI系统 - 快速验证方案
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict
import re

try:
    import openai
    from sentence_transformers import SentenceTransformer
    import numpy as np
    print("✅ 依赖库导入成功")
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    print("请安装: pip install openai sentence-transformers")
    exit(1)

class OpenAITarotRAG:
    """OpenAI + RAG 塔罗AI系统"""
    
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.db_path = "data/openai_tarot_knowledge.db"
        self.init_openai()
        self.init_database()
        
    def init_openai(self):
        """初始化OpenAI客户端"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ 请设置 OPENAI_API_KEY 环境变量")
            print("export OPENAI_API_KEY='your-api-key'")
            return False
        
        self.client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI 客户端初始化成功")
        return True
    
    def init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
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
    
    def load_embedding_model(self):
        """加载嵌入模型"""
        if self.embedding_model is None:
            print("📥 加载嵌入模型...")
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ 嵌入模型加载成功")
        return self.embedding_model
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("📚 构建知识库...")
        
        # 检查是否已有数据
        conn = sqlite3.connect(self.db_path)
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
        
        embedding_model = self.load_embedding_model()
        count = 0
        
        with open(data_file, 'r', encoding='utf-8') as f:
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
                    
                    # 生成嵌入
                    query_text = f"咨询者:{person} 问题:{question} 牌:{';'.join(cards)} 牌阵:{spread}"
                    embedding = embedding_model.encode(query_text)
                    
                    # 存储到数据库
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO readings (person, question, cards, spread, content, embedding, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (person, question, ';'.join(cards), spread, content, 
                          embedding.tobytes(), source_file))
                    
                    conn.commit()
                    conn.close()
                    
                    count += 1
                    if count % 20 == 0:
                        print(f"   已处理 {count} 条记录...")
                        
                except Exception as e:
                    print(f"⚠️ 处理记录时出错: {e}")
        
        print(f"✅ 知识库构建完成，共 {count} 条记录")
    
    def search_similar_readings(self, person: str, question: str, 
                              cards: List[str], spread: str, top_k: int = 3) -> List[Dict]:
        """搜索相似解读"""
        embedding_model = self.load_embedding_model()
        
        # 生成查询嵌入
        query_text = f"咨询者:{person} 问题:{question} 牌:{';'.join(cards)} 牌阵:{spread}"
        query_embedding = embedding_model.encode(query_text)
        
        # 从数据库检索
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM readings')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
        
        # 计算相似度
        similarities = []
        for row in rows:
            stored_embedding = np.frombuffer(row[6], dtype=np.float32)
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            similarities.append({
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
    
    def generate_reading(self, person: str, question: str, 
                        cards: List[str], spread: str = "自由牌阵") -> str:
        """生成解读"""
        
        if not self.client:
            return "❌ OpenAI 客户端未初始化"
        
        print("🔍 检索相似解读...")
        similar_readings = self.search_similar_readings(person, question, cards, spread, top_k=3)
        
        print("🤖 使用 OpenAI 生成解读...")
        
        # 构建系统提示
        system_prompt = """你是一位经验丰富的专业塔罗师，具有深厚的占星学、心理学和灵性智慧背景。你的解读风格深刻、洞察力强，能够透过塔罗牌看见咨询者的内在世界和人生议题。

你的解读特点：
1. 深度分析每张牌的含义及其在当前情境中的意义
2. 探讨牌与牌之间的关系和能量流动
3. 结合咨询者的个人特质给出针对性建议
4. 语言富有洞察力和启发性，不流于表面

请基于提供的参考案例，学习并保持这种深度专业的解读风格。"""
        
        # 构建用户提示
        user_prompt = f"""请为以下咨询提供专业塔罗解读：

咨询者：{person}
问题：{question}
牌阵：{spread}
抽到的牌：{' | '.join(cards)}

"""
        
        # 添加参考案例
        if similar_readings:
            user_prompt += "参考你以往的解读风格：\n\n"
            for i, reading in enumerate(similar_readings, 1):
                if reading['similarity'] > 0.3:
                    user_prompt += f"参考案例{i}：\n"
                    user_prompt += f"咨询者: {reading['person']}\n"
                    user_prompt += f"问题: {reading['question']}\n"
                    user_prompt += f"牌组: {reading['cards']}\n"
                    
                    # 截取核心解读部分
                    content = reading['content']
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    
                    user_prompt += f"解读风格:\n{content}\n\n"
        
        user_prompt += f"""基于以上参考风格，请为{person}提供深度专业的塔罗解读。

要求：
1. 逐一分析抽到的每张牌：{' | '.join(cards)}
2. 探讨牌组的整体信息和能量流向
3. 结合{person}的个人特质和当前问题
4. 提供具体的指导建议
5. 保持深刻洞察和专业水准

请开始解读："""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 或 gpt-4o，更强但更贵
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_content = response.choices[0].message.content
            
            # 格式化最终输出
            final_reading = f"""🔮 {person}的塔罗解读

📋 咨询信息：
   问题：{question}
   牌阵：{spread}
   抽到的牌：{' | '.join(cards)}

🎯 专业解读：
{generated_content}

📚 参考信息：
   本次解读基于 {len([r for r in similar_readings if r['similarity'] > 0.3])} 个相似案例
"""
            
            return final_reading
            
        except Exception as e:
            return f"❌ 生成解读时出错: {e}"

def main():
    print("🔮 OpenAI + RAG 塔罗AI系统")
    print("="*50)
    
    # 初始化系统
    rag_ai = OpenAITarotRAG()
    
    if not rag_ai.client:
        print("❌ OpenAI 初始化失败，请检查 API Key")
        return
    
    # 构建知识库
    rag_ai.build_knowledge_base()
    
    # 测试案例
    print("\n🧪 测试 OpenAI + RAG 解读...")
    
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
        
        reading = rag_ai.generate_reading(
            case["person"], case["question"], 
            case["cards"], case["spread"]
        )
        
        print(reading)
        
        if i < len(test_cases):
            input("\n按Enter继续...")
    
    # 交互模式
    print(f"\n{'='*60}")
    choice = input("是否进入交互模式？(y/n): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        print("\n🎯 进入 OpenAI RAG 交互模式 (输入 'quit' 退出)")
        
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
                
                print("\n🔮 正在生成 OpenAI 解读...")
                reading = rag_ai.generate_reading(person, question, cards, spread)
                
                print(f"\n{reading}")
                
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用 OpenAI RAG 塔罗AI!")
                break
            except Exception as e:
                print(f"❌ 生成失败: {e}")
    
    print("\n🎉 OpenAI RAG 塔罗AI 测试完成!")

if __name__ == "__main__":
    main() 