#!/usr/bin/env python3
"""
RAG增强塔罗AI系统 - 结合检索和生成
"""

import os
import sys
import torch
import json
import numpy as np
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
import re

# 设置环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    print("✅ 依赖库导入成功")
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    print("请安装: pip install sentence-transformers")
    sys.exit(1)

class TarotKnowledgeBase:
    """塔罗知识库 - 用于检索相关解读"""
    
    def __init__(self, db_path: str = "data/tarot_knowledge.db"):
        self.db_path = db_path
        self.embedding_model = None
        self.init_database()
        
    def init_database(self):
        """初始化数据库"""
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
                embedding BLOB,
                source_file TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ 知识库数据库初始化完成: {self.db_path}")
    
    def load_embedding_model(self):
        """加载嵌入模型"""
        if self.embedding_model is None:
            print("📥 加载中文嵌入模型...")
            try:
                # 使用中文嵌入模型
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✅ 嵌入模型加载成功")
            except Exception as e:
                print(f"⚠️ 嵌入模型加载失败，使用备用方案: {e}")
                # 备用：使用更简单的模型
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return self.embedding_model
    
    def add_reading(self, person: str, question: str, cards: List[str], 
                   spread: str, content: str, source_file: str):
        """添加解读到知识库"""
        # 生成嵌入向量
        embedding_model = self.load_embedding_model()
        
        # 组合查询文本用于嵌入
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
    
    def search_similar_readings(self, person: str, question: str, 
                              cards: List[str], spread: str, top_k: int = 3) -> List[Dict]:
        """搜索相似的解读"""
        embedding_model = self.load_embedding_model()
        
        # 生成查询嵌入
        query_text = f"咨询者:{person} 问题:{question} 牌:{';'.join(cards)} 牌阵:{spread}"
        query_embedding = embedding_model.encode(query_text)
        
        # 从数据库检索所有记录
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

class RAGTarotAI:
    """RAG增强的塔罗AI"""
    
    def __init__(self):
        self.knowledge_base = TarotKnowledgeBase()
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def load_model(self):
        """加载训练好的模型"""
        print("🤖 加载RAG增强塔罗AI...")
        
        # 检查设备
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("✅ 使用 Apple Silicon MPS")
        else:
            self.device = "cpu"
            print("⚠️ 使用 CPU")
        
        model_path = "./models/qwen-tarot-24gb"
        base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
        
        try:
            # 修复LoRA配置
            adapter_config_path = Path(model_path) / "adapter_config.json"
            with open(adapter_config_path, 'r') as f:
                config = json.load(f)
            config["inference_mode"] = False
            with open(adapter_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # 加载模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model = PeftModel.from_pretrained(base_model, model_path)
            
            if self.device == "mps":
                self.model = self.model.to("mps")
            
            print("✅ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("📚 构建塔罗知识库...")
        
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
                    
                    # 添加到知识库
                    self.knowledge_base.add_reading(
                        person, question, cards, spread, content, source_file
                    )
                    
                    count += 1
                    if count % 10 == 0:
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
        
        # 4. 后处理和格式化
        final_reading = self._format_final_reading(
            person, question, cards, spread, generated, similar_readings
        )
        
        return final_reading
    
    def _build_enhanced_prompt(self, person: str, question: str, cards: List[str], 
                             spread: str, similar_readings: List[Dict]) -> str:
        """构建增强prompt"""
        
        prompt = f"""塔罗解读：
咨询者：{person}
问题：{question}
牌阵：{spread}
牌：{';'.join(cards)}

参考相似解读：
"""
        
        # 添加相似解读作为参考
        for i, reading in enumerate(similar_readings, 1):
            similarity_score = reading['similarity']
            if similarity_score > 0.3:  # 只使用相似度较高的
                prompt += f"\n参考{i} (相似度: {similarity_score:.2f}):\n"
                prompt += f"咨询者: {reading['person']}\n"
                prompt += f"问题: {reading['question']}\n"
                prompt += f"牌: {reading['cards']}\n"
                
                # 截取参考内容的关键部分 (前500字符)
                ref_content = reading['content'][:500]
                prompt += f"解读摘要: {ref_content}...\n"
        
        prompt += "\n基于以上参考和你的专业知识，请提供专业解读："
        
        return prompt
    
    def _generate_with_model(self, prompt: str) -> str:
        """使用模型生成解读"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = full_response[len(prompt):].strip()
            
            return generated
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return "生成解读时遇到技术问题，请重试。"
    
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

📚 相关参考：
"""
        
        # 添加参考信息
        for i, reading in enumerate(similar_readings[:2], 1):  # 只显示前2个参考
            if reading['similarity'] > 0.3:
                final_reading += f"   参考{i}: {reading['person']}的{reading['question']} (相似度: {reading['similarity']:.1%})\n"
        
        return final_reading

def main():
    print("🔮 RAG增强塔罗AI系统")
    print("="*50)
    
    # 初始化系统
    rag_ai = RAGTarotAI()
    
    # 加载模型
    if not rag_ai.load_model():
        return
    
    # 构建知识库
    rag_ai.build_knowledge_base()
    
    # 测试案例
    print("\n🧪 测试RAG增强解读...")
    
    test_cases = [
        {
            "person": "Mel",
            "question": "事业发展方向",
            "cards": ["愚人(正位)", "力量(正位)", "星币十(正位)"],
            "spread": "三张牌解读"
        },
        {
            "person": "测试者",
            "question": "感情运势",
            "cards": ["恋人(正位)", "圣杯二(正位)", "圣杯十(正位)"],
            "spread": "感情牌阵"
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
        
        if i < len(test_cases):
            input("\n按Enter继续下一个测试...")
    
    # 进入交互模式
    print(f"\n{'='*60}")
    choice = input("是否进入交互模式？(y/n): ").strip().lower()
    
    if choice in ['y', 'yes', '是']:
        print("\n🎯 进入RAG增强交互模式 (输入 'quit' 退出)")
        
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
                
                print("\n🔮 正在生成RAG增强解读...")
                reading = rag_ai.generate_enhanced_reading(person, question, cards, spread)
                
                print(f"\n{reading}")
                
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用RAG增强塔罗AI!")
                break
            except Exception as e:
                print(f"❌ 生成失败: {e}")
    
    print("\n🎉 RAG增强塔罗AI系统测试完成!")

if __name__ == "__main__":
    main() 