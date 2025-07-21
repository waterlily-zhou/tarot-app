#!/usr/bin/env python3
"""
塔罗AI系统
整合卡牌识别、RAG检索和本地LLM，提供完整的AI解牌服务
"""

import json
import requests
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rag_system import TarotRAGSystem
import time

class TarotAISystem:
    def __init__(self, 
                 llm_model: str = "qwen2.5:1.5b",
                 ollama_url: str = "http://localhost:11434"):
        
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        
        print("🔮 初始化塔罗AI系统...")
        
        # 初始化RAG系统
        print("📚 加载知识库...")
        self.rag = TarotRAGSystem()
        
        # 测试Ollama连接
        self._test_llm_connection()
        
        print("✅ 塔罗AI系统初始化完成！")
    
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
            print("请确保Ollama服务已启动: ollama serve")
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """调用本地LLM"""
        try:
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"LLM调用失败，状态码: {response.status_code}"
                
        except Exception as e:
            return f"LLM调用出错: {e}"
    
    def identify_cards_from_text(self, card_text: str) -> List[str]:
        """从文本中识别卡牌（临时方案，未来替换为视觉识别）"""
        # 读取已知卡牌列表
        card_list_file = Path("data/processed/all_cards/card_list.json")
        if card_list_file.exists():
            with open(card_list_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_cards = data.get('all_cards', [])
        else:
            all_cards = []
        
        # 简单的卡牌识别逻辑
        identified_cards = []
        for card in all_cards:
            if card in card_text:
                identified_cards.append(card)
        
        return identified_cards
    
    def generate_reading(self, 
                        cards: List[str], 
                        question: str = None,
                        user_id: str = None,
                        spread_type: str = "general") -> Dict:
        """生成塔罗解牌"""
        
        print(f"🎴 开始解牌...")
        print(f"卡牌: {cards}")
        print(f"问题: {question}")
        
        # 1. 生成查询文本
        if question:
            query = f"{question} {' '.join(cards)}"
        else:
            query = f"塔罗解牌 {' '.join(cards)}"
        
        # 2. 使用RAG获取相关上下文
        print("🔍 检索相关知识...")
        context = self.rag.generate_context_for_query(query, user_id)
        
        # 3. 构建系统提示词
        system_prompt = """你是一位专业的塔罗占卜师，拥有深厚的塔罗牌解读能力。
你的解牌风格特点：
- 深入细致，富有洞察力
- 结合心理学和灵性智慧
- 关注内在成长和指导意义
- 语言优美，富有诗意

请基于提供的卡牌和背景知识，进行专业的塔罗解读。
解读应该包含：
1. 卡牌组合的整体能量
2. 每张主要卡牌的含义
3. 对当前情况的洞察
4. 未来的建议和指导

请用中文回答，语气温和而充满智慧。"""
        
        # 4. 构建用户提示词
        user_prompt = f"""请为以下塔罗牌阵进行解读：

卡牌：{', '.join(cards)}
问题：{question if question else '综合运势'}
牌阵类型：{spread_type}

相关背景知识：
{context}

请进行详细而深入的解读："""
        
        # 5. 调用LLM生成解读
        print("🤖 AI正在解牌...")
        start_time = time.time()
        reading_text = self._call_llm(user_prompt, system_prompt)
        generation_time = time.time() - start_time
        
        # 6. 整理结果
        result = {
            "cards": cards,
            "question": question,
            "spread_type": spread_type,
            "reading": reading_text,
            "context_used": context[:500] + "..." if len(context) > 500 else context,
            "user_id": user_id,
            "timestamp": time.time(),
            "generation_time": generation_time,
            "model_used": self.llm_model
        }
        
        # 7. 保存到用户上下文（如果有用户ID）
        if user_id:
            self.rag.add_user_context(user_id, {
                "type": "reading",
                "summary": f"解读了 {', '.join(cards[:3])} 等卡牌",
                "cards": cards,
                "themes": [question] if question else ["综合运势"],
                "notes": f"生成时间: {generation_time:.2f}秒"
            })
        
        print(f"✅ 解牌完成 (用时 {generation_time:.2f}秒)")
        return result
    
    def analyze_image(self, image_path: str) -> List[str]:
        """分析图片中的卡牌（占位函数）"""
        print(f"📸 分析图片: {image_path}")
        
        # TODO: 这里将来整合视觉识别模型
        # 目前返回一些示例卡牌
        sample_cards = ["力量", "魔法师", "皇后"]
        
        print(f"识别到卡牌: {sample_cards}")
        return sample_cards
    
    def interactive_reading(self):
        """交互式解牌"""
        print("\n🔮 欢迎使用塔罗AI解牌系统")
        print("=" * 50)
        
        while True:
            print("\n选择输入方式：")
            print("1. 手动输入卡牌")
            print("2. 分析图片 (暂未实现)")
            print("3. 退出")
            
            choice = input("\n请选择 (1-3): ").strip()
            
            if choice == "1":
                self._manual_card_input()
            elif choice == "2":
                print("📸 图片分析功能正在开发中...")
                self._image_analysis_demo()
            elif choice == "3":
                print("👋 感谢使用塔罗AI系统！")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def _manual_card_input(self):
        """手动输入卡牌"""
        print("\n🎴 手动输入卡牌")
        print("请输入卡牌名称，用逗号分隔 (例如: 愚者,魔法师,皇后)")
        
        card_input = input("卡牌: ").strip()
        if not card_input:
            print("❌ 未输入卡牌")
            return
        
        cards = [card.strip() for card in card_input.split(',') if card.strip()]
        
        question = input("问题 (可选): ").strip()
        user_id = input("用户ID (可选): ").strip()
        
        if not user_id:
            user_id = "anonymous"
        
        # 生成解读
        result = self.generate_reading(cards, question, user_id)
        
        # 显示结果
        self._display_reading_result(result)
    
    def _image_analysis_demo(self):
        """图片分析演示"""
        print("\n📸 图片分析演示")
        
        image_path = input("请输入图片路径: ").strip()
        if not image_path or not Path(image_path).exists():
            print("❌ 图片路径无效")
            return
        
        # 模拟卡牌识别
        cards = self.analyze_image(image_path)
        
        question = input("问题 (可选): ").strip()
        user_id = input("用户ID (可选): ").strip()
        
        if not user_id:
            user_id = "anonymous"
        
        # 生成解读
        result = self.generate_reading(cards, question, user_id)
        
        # 显示结果
        self._display_reading_result(result)
    
    def _display_reading_result(self, result: Dict):
        """显示解读结果"""
        print("\n" + "="*60)
        print("🔮 塔罗解读结果")
        print("="*60)
        
        print(f"\n🎴 卡牌: {', '.join(result['cards'])}")
        if result['question']:
            print(f"❓ 问题: {result['question']}")
        print(f"🕐 生成时间: {result['generation_time']:.2f}秒")
        print(f"🤖 使用模型: {result['model_used']}")
        
        print(f"\n📖 解读:")
        print("-" * 40)
        print(result['reading'])
        print("-" * 40)
        
        # 询问是否保存
        save = input("\n💾 是否保存此次解读？(y/n): ").strip().lower()
        if save == 'y':
            self._save_reading(result)
    
    def _save_reading(self, result: Dict):
        """保存解读结果"""
        readings_dir = Path("data/ai_readings")
        readings_dir.mkdir(exist_ok=True)
        
        timestamp = int(result['timestamp'])
        filename = f"reading_{timestamp}.json"
        filepath = readings_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 解读已保存: {filepath}")
    
    def get_system_stats(self) -> Dict:
        """获取系统统计信息"""
        rag_stats = self.rag.get_database_stats()
        
        # 检查AI解读数量
        ai_readings_dir = Path("data/ai_readings")
        ai_readings_count = len(list(ai_readings_dir.glob("*.json"))) if ai_readings_dir.exists() else 0
        
        return {
            "knowledge_base": rag_stats,
            "ai_readings": ai_readings_count,
            "llm_model": self.llm_model
        }

def main():
    """主函数"""
    print("🌟 塔罗AI系统启动")
    
    try:
        # 初始化系统
        tarot_ai = TarotAISystem()
        
        # 显示系统状态
        print("\n📊 系统状态:")
        stats = tarot_ai.get_system_stats()
        print(f"知识库: {stats['knowledge_base']}")
        print(f"AI解读记录: {stats['ai_readings']} 条")
        print(f"LLM模型: {stats['llm_model']}")
        
        # 开始交互
        tarot_ai.interactive_reading()
        
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        print("请检查：")
        print("1. Ollama服务是否启动: ollama serve")
        print("2. 模型是否已下载: ollama pull qwen2.5:1.5b")
        print("3. RAG系统是否已初始化")

if __name__ == "__main__":
    main() 