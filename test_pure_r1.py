#!/usr/bin/env python3
"""
深度学习版DeepSeek R1测试 - 让R1真正学会解牌思维
"""
import os
import openai
import sqlite3
from pathlib import Path
import json

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

def get_card_meanings(cards: list):
    """获取牌意笔记"""
    card_meanings = {}
    
    for card_name in cards:
        # 清理牌名
        clean_card = card_name.replace('(正位)', '').replace('(逆位)', '').strip()
        
        # 查找对应的MD文件
        card_file = Path(f"data/card_meanings/{clean_card}.md")
        
        if card_file.exists():
            try:
                with open(card_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    card_meanings[clean_card] = content
                    print(f"✅ 找到 {clean_card} 的牌意笔记")
            except Exception as e:
                print(f"⚠️ 读取 {clean_card} 牌意失败: {e}")
        else:
            print(f"🔍 未找到 {clean_card} 的牌意笔记")
    
    return card_meanings

def extract_reading_wisdom(person: str, cards: list):
    """从历史解读中提炼智慧而非原文"""
    db_path = "data/deepseek_tarot_knowledge.db"
    
    if not Path(db_path).exists():
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取个人解读历史
    cursor.execute('''
        SELECT question, cards, content 
        FROM readings 
        WHERE person = ?
        ORDER BY rowid DESC
        LIMIT 5
    ''', (person,))
    
    person_readings = cursor.fetchall()
    
    # 获取相同牌的解读
    query_cards_clean = [card.replace('(正位)', '').replace('(逆位)', '').strip() for card in cards]
    card_readings = []
    
    for card in query_cards_clean[:2]:
        cursor.execute('''
            SELECT question, cards, content 
            FROM readings 
            WHERE cards LIKE ? 
            LIMIT 3
        ''', (f'%{card}%',))
        card_readings.extend(cursor.fetchall())
    
    conn.close()
    
    return {
        'person_readings': person_readings,
        'card_readings': card_readings
    }

def create_learning_framework(client, person: str, historical_data):
    """让R1先学习解牌框架，再应用到具体解读"""
    
    # 构建学习提示
    learning_prompt = f"""你是一位资深塔罗师。现在需要你分析以下塔罗解读案例，提炼出这位塔罗师的核心解牌智慧和方法论。

请仔细分析以下 {person} 的解读案例，总结出：

1. **个人特征模式** - 这个人的核心议题、能量特质、常见模式
2. **解牌思维框架** - 这位塔罗师如何分析牌意、如何连接牌与牌、如何个人化解读
3. **牌意理解深度** - 每张牌的独特解读角度和深层含义

## {person}的解读案例：
"""
    
    # 添加个人案例（精选，不要太多）
    if historical_data and historical_data['person_readings']:
        for i, (question, cards_str, content) in enumerate(historical_data['person_readings'][:3], 1):
            learning_prompt += f"""
### 案例{i}：
问题：{question}
牌组：{cards_str}
解读要点：{content[:500]}...

"""
    
    # 添加牌组参考（用于理解牌意）
    if historical_data and historical_data['card_readings']:
        learning_prompt += f"\n## 相关牌组解读片段（用于理解牌意深度）：\n"
        for i, (question, cards_str, content) in enumerate(historical_data['card_readings'][:3], 1):
            learning_prompt += f"""
片段{i}：{content[:300]}...
"""
    
    learning_prompt += f"""

现在请你：
1. **总结 {person} 的个人特征和核心议题**
2. **提炼这位塔罗师的解牌思维和方法论**  
3. **理解不同牌的深层含义和独特视角**

不要复述案例内容，而是要抽象出智慧精华和方法论。请以结构化的方式总结你的学习成果。
"""
    
    try:
        print("🧠 R1正在学习解牌思维框架...")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": learning_prompt}],
            max_tokens=1500,
            temperature=0.3  # 更低的温度确保分析的准确性
        )
        
        framework = response.choices[0].message.content.strip()
        return framework
        
    except Exception as e:
        print(f"❌ 学习框架生成失败: {e}")
        return None

def test_deep_learning_r1(client):
    """测试深度学习版R1"""
    
    person = "Mel"
    cards = ["愚人(正位)", "力量(正位)", "皇后(正位)"]
    
    print(f"🎓 测试深度学习版DeepSeek R1解读（{person}）...")
    
    # 1. 获取牌意笔记
    card_meanings = get_card_meanings(cards)
    
    # 2. 提炼历史智慧
    historical_data = extract_reading_wisdom(person, cards)
    
    # 3. 让R1先学习框架
    framework = create_learning_framework(client, person, historical_data)
    
    if not framework:
        print("❌ 无法生成学习框架")
        return None
    
    print("📚 R1学习成果：")
    print("-" * 40)
    print(framework)
    print("-" * 40)
    
    # 4. 基于学习的框架进行解读
    system_prompt = f"""你是一位专业的塔罗师。你已经通过学习掌握了以下解牌智慧：

{framework}

现在请运用这些学到的智慧进行塔罗解读。"""
    
    # 如果有牌意笔记，加入系统提示
    if card_meanings:
        system_prompt += "\n\n你还掌握了以下牌意笔记：\n"
        for card, meaning in card_meanings.items():
            system_prompt += f"\n{card}：\n{meaning}\n"
    
    system_prompt += "\n\n请运用你学到的解牌思维和个人化理解进行专业解读。"
    
    user_prompt = f"""请为以下咨询提供塔罗解读：

咨询者：{person}
问题：当前的内在成长状态和2024年发展方向
牌阵：内在探索牌阵
抽到的牌：{' | '.join(cards)}

请运用你刚才学到的解牌框架和对{person}的理解进行深度解读。"""
    
    try:
        print("🔮 基于深度学习的R1解读...")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        
        print("🎯 深度学习版解读结果：")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")
        return None

def test_card_meaning_enhanced_r1(client):
    """测试加入牌意笔记的R1解读"""
    
    person = "Mel"
    cards = ["愚人(正位)", "力量(正位)", "皇后(正位)"]
    
    print(f"🔮 测试牌意笔记增强版DeepSeek R1解读（{person}）...")
    
    # 获取牌意笔记
    card_meanings = get_card_meanings(cards)
    
    # 获取轻量背景信息
    context = get_light_person_context(person, cards)
    
    # 构建增强的系统提示
    system_prompt = """你是一位专业的塔罗师。请为咨询提供深度的塔罗解读。"""
    
    # 如果有牌意笔记，加入系统提示
    if card_meanings:
        system_prompt += "\n\n以下是一些牌意笔记供参考：\n"
        for card, meaning in card_meanings.items():
            system_prompt += f"\n{card}：\n{meaning}\n"
        system_prompt += "\n请结合这些牌意深化理解进行解读。"
    
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
        print("🤖 调用牌意笔记增强版DeepSeek R1...")
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
        
        print("🎯 牌意笔记增强版解读结果：")
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
    cards = ["愚人(正位)", "力量(正位)", "皇后(正位)"]
    
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
抽到的牌：愚人(正位) | 力量(正位) | 皇后(正位)

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
    print("🎓 深度学习版DeepSeek R1测试")
    print("=" * 50)
    
    client = init_deepseek()
    if not client:
        return
    
    # 测试深度学习版本（新增）
    print("\n🎯 深度学习版本...")
    deep_learning_result = test_deep_learning_r1(client)
    
    print("\n" + "="*60)
    
    # 测试牌意笔记增强版
    print("\n📚 牌意笔记增强版本...")
    enhanced_result = test_card_meaning_enhanced_r1(client)
    
    print("\n" + "="*60)
    
    # 测试轻量个人化版
    print("\n💡 轻量个人化版本...")
    personalized_result = test_light_personalized_r1(client)
    
    print("\n💭 深度分析：")
    print("- 深度学习版是否真正掌握了解牌思维？")
    print("- 是否能抽象出你的个人化方法论？")
    print("- 解读质量是否有显著提升？")
    print("- 是否减少了机械引用，增加了理解深度？")

if __name__ == "__main__":
    main() 