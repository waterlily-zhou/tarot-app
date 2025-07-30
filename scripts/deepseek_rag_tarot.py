#!/usr/bin/env python3
"""
DeepSeek R1 + RAG 塔罗AI系统 - 性价比最优方案
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

class DeepSeekTarotRAG:
    """DeepSeek R1 + RAG 塔罗AI系统"""
    
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self.db_path = "data/deepseek_tarot_knowledge.db"
        self.init_deepseek()
        self.init_database()
        
    def init_deepseek(self):
        """初始化DeepSeek客户端"""
        # 首先尝试从环境变量读取
        api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # 如果环境变量没有，尝试从 .env.local 文件读取
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
            print("❌ 请设置 DEEPSEEK_API_KEY")
            print("方法1: export DEEPSEEK_API_KEY='your-api-key'")
            print("方法2: 在 .env.local 文件中添加 DEEPSEEK_API_KEY=your-api-key")
            print("🔗 获取API Key: https://platform.deepseek.com/api_keys")
            return False
        
        # 使用DeepSeek的API端点
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        print("✅ DeepSeek R1 客户端初始化成功")
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
                              cards: List[str], spread: str, top_k: int = 5) -> List[Dict]:
        """搜索相似解读 - 优化版"""
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
        
        # 计算增强相似度
        similarities = []
        query_cards_set = set([card.strip().replace('(正位)', '').replace('(逆位)', '') 
                              for card in cards])
        
        for row in rows:
            stored_embedding = np.frombuffer(row[6], dtype=np.float32)
            
            # 基础语义相似度
            semantic_similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            # 计算牌组重叠度
            stored_cards = row[3].split(';') if row[3] else []
            stored_cards_set = set([card.strip().replace('(正位)', '').replace('(逆位)', '') 
                                   for card in stored_cards])
            
            card_overlap = len(query_cards_set & stored_cards_set)
            total_cards = len(query_cards_set | stored_cards_set)
            card_similarity = card_overlap / max(total_cards, 1) if total_cards > 0 else 0
            
            # 同咨询者加权
            person_bonus = 0.2 if row[1] == person else 0
            
            # 同牌阵类型加权
            spread_bonus = 0.1 if row[4] == spread else 0
            
            # 综合相似度计算
            final_similarity = (
                semantic_similarity * 0.5 +  # 语义相似度权重50%
                card_similarity * 0.3 +      # 牌组相似度权重30%
                person_bonus +               # 同人加权20%
                spread_bonus                 # 同牌阵加权10%
            )
            
            similarities.append({
                'person': row[1],
                'question': row[2],
                'cards': row[3],
                'spread': row[4],
                'content': row[5],
                'source_file': row[7],
                'similarity': final_similarity,
                'semantic_sim': semantic_similarity,
                'card_overlap': card_overlap,
                'card_similarity': card_similarity
            })
        
        # 按综合相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def generate_reading(self, person: str, question: str, 
                        cards: List[str], spread: str = "自由牌阵") -> str:
        """生成解读"""
        
        if not self.client:
            return "❌ DeepSeek 客户端未初始化"
        
        print("🔍 检索相似解读...")
        similar_readings = self.search_similar_readings(person, question, cards, spread, top_k=5)
        
        print("🤖 使用 DeepSeek R1 生成解读...")
        
        # 构建系统提示 - 针对DeepSeek R1优化
        system_prompt = """你是一位具有深厚专业背景的塔罗师，拥有丰富的占星学、心理学和灵性智慧。你的解读风格深刻、富有洞察力，能够通过塔罗牌透视咨询者的内在世界和人生议题。

你的解读特色：
1. 对每张牌进行深度分析，阐释其在当前情境中的意义
2. 探讨牌与牌之间的关系和能量流动
3. 结合咨询者的个人特质提供针对性建议
4. 语言具有启发性和洞察力，不流于表面
5. 运用中文的表达优势，体现东方智慧与西方塔罗的融合

请基于提供的参考案例，学习并保持这种专业深度的解读风格。"""
        
        # 构建用户提示
        user_prompt = f"""请为以下咨询提供专业塔罗解读：

咨询者：{person}
问题：{question}
牌阵：{spread}
抽到的牌：{' | '.join(cards)}

"""
        
        # 添加参考案例 - 优化版
        if similar_readings:
            user_prompt += "参考你以往的解读风格和案例：\n\n"
            
            # 分类展示参考案例
            high_sim_readings = [r for r in similar_readings if r['similarity'] > 0.4]
            card_match_readings = [r for r in similar_readings if r['card_overlap'] > 0]
            same_person_readings = [r for r in similar_readings if r['person'] == person]
            
            case_count = 1
            
            # 优先显示高相似度案例
            for reading in high_sim_readings[:2]:
                user_prompt += f"高相似度参考{case_count}：\n"
                user_prompt += f"咨询者: {reading['person']}\n"
                user_prompt += f"问题: {reading['question']}\n"
                user_prompt += f"牌组: {reading['cards']}\n"
                user_prompt += f"相似度: {reading['similarity']:.2f} (牌重叠:{reading['card_overlap']}张)\n"
                
                content = reading['content']
                if len(content) > 1000:
                    content = content[:1000] + "..."
                user_prompt += f"解读风格:\n{content}\n\n"
                case_count += 1
            
            # 显示同牌参考
            for reading in card_match_readings[:2]:
                if reading not in high_sim_readings:
                    user_prompt += f"同牌参考{case_count}：\n"
                    user_prompt += f"咨询者: {reading['person']}\n"
                    user_prompt += f"问题: {reading['question']}\n"
                    user_prompt += f"牌组: {reading['cards']} (重叠{reading['card_overlap']}张)\n"
                    
                    content = reading['content']
                    if len(content) > 800:
                        content = content[:800] + "..."
                    user_prompt += f"解读参考:\n{content}\n\n"
                    case_count += 1
            
            # 显示同人参考
            for reading in same_person_readings[:1]:
                if reading not in high_sim_readings and reading not in card_match_readings:
                    user_prompt += f"{person}的历史参考：\n"
                    user_prompt += f"问题: {reading['question']}\n"
                    user_prompt += f"牌组: {reading['cards']}\n"
                    
                    content = reading['content']
                    if len(content) > 600:
                        content = content[:600] + "..."
                    user_prompt += f"个人特质参考:\n{content}\n\n"
        
        user_prompt += f"""基于以上参考风格，请为{person}提供深度专业的塔罗解读。

要求：
1. 逐一深入分析抽到的每张牌：{' | '.join(cards)}
2. 探讨牌组的整体信息和能量流向
3. 结合{person}的个人特质和当前问题
4. 提供具体的人生指导建议
5. 保持深刻洞察和专业水准
6. 体现中文表达的深度和美感

请开始解读："""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",  # 使用推理模型
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9
            )
            
            generated_content = response.choices[0].message.content
            
            # 格式化最终输出
            final_reading = f"""🔮 {person}的专业塔罗解读

📋 咨询信息：
   问题：{question}
   牌阵：{spread}
   抽到的牌：{' | '.join(cards)}

🎯 DeepSeek R1 深度解读：
{generated_content}

 📚 参考信息：
    高相似度案例: {len([r for r in similar_readings if r['similarity'] > 0.4])} 个
    同牌组案例: {len([r for r in similar_readings if r['card_overlap'] > 0])} 个
    同咨询者案例: {len([r for r in similar_readings if r['person'] == person])} 个
    由 DeepSeek R1 推理模型生成，结合了传统塔罗智慧与现代AI洞察

💰 成本优势：
   相比传统方案节省约80%费用，同时保持专业水准
"""
            
            return final_reading
            
        except Exception as e:
            return f"❌ 生成解读时出错: {e}"
    
    def check_usage_and_cost(self):
        """检查使用情况和成本"""
        try:
            # DeepSeek暂不提供余额查询API，但可以估算成本
            print("💰 DeepSeek R1 成本优势:")
            print("   - 输入: $0.50/M tokens (vs OpenAI $10/M)")
            print("   - 输出: $2.18/M tokens (vs OpenAI $30/M)")
            print("   - 节省: ~80%的费用")
            print("   - 推理能力: 接近GPT-4水平")
            
        except Exception as e:
            print(f"⚠️ 无法获取使用信息: {e}")

def main():
    print("🔮 DeepSeek R1 + RAG 塔罗AI系统")
    print("="*50)
    
    # 初始化系统
    rag_ai = DeepSeekTarotRAG()
    
    if not rag_ai.client:
        print("❌ DeepSeek 初始化失败，请检查 API Key")
        return
    
    # 显示成本优势
    rag_ai.check_usage_and_cost()
    
    # 构建知识库
    rag_ai.build_knowledge_base()
    
    # 测试案例
    print("\n🧪 测试 DeepSeek R1 + RAG 解读...")
    
    test_cases = [
        {
            "person": "Mel",
            "question": "事业发展方向",
            "cards": ["愚人(正位)", "力量(正位)", "星币十(正位)"],
            "spread": "三张牌解读"
        },
        {
            "person": "测试者",
            "question": "2024年运势",
            "cards": ["太阳(正位)", "世界(正位)", "圣杯十(正位)"],
            "spread": "年度牌阵"
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
        print("\n🎯 进入 DeepSeek R1 RAG 交互模式 (输入 'quit' 退出)")
        
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
                
                print("\n🔮 正在生成 DeepSeek R1 解读...")
                reading = rag_ai.generate_reading(person, question, cards, spread)
                
                print(f"\n{reading}")
                
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用 DeepSeek R1 RAG 塔罗AI!")
                break
            except Exception as e:
                print(f"❌ 生成失败: {e}")
    
    print("\n🎉 DeepSeek R1 RAG 塔罗AI 测试完成!")
    print("💡 推荐：DeepSeek R1 提供了接近GPT-4的能力，但成本仅为其1/5")

if __name__ == "__main__":
    main() 