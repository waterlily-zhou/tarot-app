#!/usr/bin/env python3
"""
轻量个人化DeepSeek R1测试 - 在保持R1推理能力基础上加入个人信息
"""
import os
import openai
import sqlite3
from pathlib import Path

def init_deepseek():
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
        print("❌ 请设置 DEEPSEEK_API_KEY")
        return None
    
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    print("✅ DeepSeek R1 客户端初始化成功")
    return client

def get_light_person_context(person: str, cards: list):
    """轻量获取个人背景信息"""
    db_path = "data/deepseek_tarot_knowledge.db"
    
    if not Path(db_path).exists():
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 只获取少量关键信息
    cursor.execute('''
        SELECT question, cards, content 
        FROM readings 
        WHERE person = ?
        ORDER BY rowid DESC
        LIMIT 3
    ''', (person,))
    
    recent_readings = cursor.fetchall()
    
    # 获取当前牌组的一些相关解读（不要太多）
    query_cards_clean = [card.replace('(正位)', '').replace('(逆位)', '').strip() for card in cards]
    card_examples = []
    
    for card in query_cards_clean[:2]:  # 只取前2张牌
        cursor.execute('''
            SELECT content 
            FROM readings 
            WHERE cards LIKE ? 
            LIMIT 2
        ''', (f'%{card}%',))
        card_examples.extend(cursor.fetchall())
    
    conn.close()
    
    return {
        'recent_readings': recent_readings,
        'card_examples': card_examples
    }

def test_light_personalized_r1(client):
    """测试轻量个人化的R1解读"""
    
    person = "Mel"
    cards = ["愚人(正位)", "力量(正位)", "星币十(正位)"]
    
    print(f"🔮 测试轻量个人化DeepSeek R1解读（{person}）...")
    
    # 获取轻量背景信息
    context = get_light_person_context(person, cards)
    
    # 构建简洁的个人化提示
    system_prompt = """你是一位专业的塔罗师。请为咨询提供深度的塔罗解读。"""
    
    user_prompt = f"""请为以下咨询提供塔罗解读：

咨询者：{person}
问题：当前的内在成长状态
牌阵：内在探索牌阵
抽到的牌：{' | '.join(cards)}"""

    # 如果有背景信息，轻量地加入
    if context and context['recent_readings']:
        user_prompt += f"""

背景信息：{person}近期的一些关注点："""
        for i, (question, _, _) in enumerate(context['recent_readings'], 1):
            user_prompt += f"\n- {question}"
        user_prompt += "\n\n请结合这些背景适当个人化解读。"
    
    user_prompt += "\n\n请进行专业解读。"
    
    try:
        print("🤖 调用轻量个人化DeepSeek R1...")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.7,
            top_p=0.9
        )
        
        result = response.choices[0].message.content.strip()
        
        print("🎯 轻量个人化解读结果：")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        # 显示thinking过程
        if hasattr(response.choices[0].message, 'reasoning_content'):
            print("\n🧠 R1 Thinking过程：")
            print("-" * 40)
            print(response.choices[0].message.reasoning_content)
            print("-" * 40)
        
        return result
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        return None

def test_pure_r1_thinking(client):
    """测试纯净的R1 thinking能力"""
    
    print("🔮 测试纯净DeepSeek R1的塔罗解读能力...")
    
    # 最简单的提示
    system_prompt = """你是一位专业的塔罗师。请为咨询提供深度的塔罗解读。"""
    
    user_prompt = """请为以下咨询提供塔罗解读：

咨询者：Mel
问题：当前的内在成长状态
牌阵：内在探索牌阵
抽到的牌：愚人(正位) | 力量(正位) | 星币十(正位)

请进行专业解读。"""
    
    try:
        print("🤖 调用DeepSeek R1 (纯净版)...")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.7,
            top_p=0.9
        )
        
        result = response.choices[0].message.content.strip()
        return result
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        return None

def main():
    print("🔮 轻量个人化DeepSeek R1测试")
    print("=" * 50)
    
    client = init_deepseek()
    if not client:
        return
    
    # 测试纯净版
    print("\n1️⃣ 纯净版本...")
    pure_result = test_pure_r1_thinking(client)
    
    print("\n" + "="*60)
    
    # 测试轻量个人化版
    print("\n2️⃣ 轻量个人化版本...")
    personalized_result = test_light_personalized_r1(client)
    
    print("\n💭 对比分析：")
    print("- 个人化版本是否保持了R1的推理质量？")
    print("- 是否增加了有用的个人化信息？")
    print("- 背景信息是否干扰了原始解读能力？")

if __name__ == "__main__":
    main() 