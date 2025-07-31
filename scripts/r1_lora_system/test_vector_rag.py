#!/usr/bin/env python3
"""
向量RAG验证脚本 - 验证向量检索是否真的比SQL检索更好
"""
import sqlite3
from pathlib import Path
import json

def install_dependencies():
    """安装必要的依赖"""
    print("📦 安装向量检索依赖...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "chromadb", "sentence-transformers"])
        print("✅ 依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def setup_real_vector_db():
    """建立真实的向量数据库"""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        print("✅ ChromaDB导入成功")
    except ImportError:
        print("❌ 请先安装依赖: pip install chromadb sentence-transformers")
        if install_dependencies():
            import chromadb
            from sentence_transformers import SentenceTransformer
        else:
            return None
    
    # 检查数据库是否存在
    db_path = "data/deepseek_tarot_knowledge.db"
    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}")
        return None
    
    print("🔧 设置向量数据库...")
    
    # 初始化ChromaDB
    client = chromadb.Client()
    try:
        collection = client.create_collection("tarot_knowledge")
    except Exception:
        # 如果collection已存在，先删除再创建
        try:
            client.delete_collection("tarot_knowledge")
            collection = client.create_collection("tarot_knowledge")
        except Exception as e:
            print(f"❌ 创建collection失败: {e}")
            return None
    
    # 加载嵌入模型
    print("📥 加载嵌入模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 从数据库读取历史解读
    print("📚 读取历史解读数据...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT person, question, cards, content FROM readings")
    readings = cursor.fetchall()
    conn.close()
    
    if not readings:
        print("❌ 没有找到历史解读数据")
        return None
    
    print(f"📊 找到 {len(readings)} 条历史解读")
    
    # 创建文档和嵌入
    documents = []
    metadatas = []
    embeddings = []
    ids = []
    
    for i, (person, question, cards, content) in enumerate(readings):
        # 构建文档文本
        doc_text = f"咨询者：{person}\n问题：{question}\n牌组：{cards}\n解读：{content}"
        documents.append(doc_text)
        
        metadatas.append({
            "person": person,
            "question": question, 
            "cards": cards
        })
        
        # 生成嵌入
        embedding = model.encode(doc_text).tolist()
        embeddings.append(embedding)
        ids.append(f"reading_{i}")
    
    # 添加到向量数据库
    print("💾 添加数据到向量数据库...")
    try:
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ 成功添加 {len(documents)} 条记录到向量数据库")
        return collection
    except Exception as e:
        print(f"❌ 添加数据失败: {e}")
        return None

def test_sql_vs_vector_retrieval():
    """对比SQL检索和向量检索的效果"""
    
    # 设置真实的向量数据库
    collection = setup_real_vector_db()
    
    # 测试查询
    test_queries = [
        ("Mel", ["愚人", "力量"], "内在成长"),
        ("Mel", ["皇后", "恶魔"], "关系"),
        ("Mel", ["星币十"], "事业"),
    ]
    
    print("🔍 向量RAG vs SQL检索对比测试")
    print("=" * 50)
    
    overall_results = []
    
    for person, cards, theme in test_queries:
        print(f"\n📋 测试查询：{person} | {', '.join(cards)} | {theme}")
        
        # SQL检索（现有方法）
        sql_results = sql_retrieve(person, cards)
        
        # 向量检索（真实方法）
        vector_results = vector_retrieve(person, cards, theme, collection)
        
        # 简单的相关性评估
        relevance_score = evaluate_relevance(sql_results, vector_results, person, cards)
        overall_results.append(relevance_score)
    
    # 总结
    print(f"\n" + "="*60)
    print("📊 总体测试结果:")
    
    sql_wins = sum(1 for r in overall_results if r['winner'] == 'SQL')
    vector_wins = sum(1 for r in overall_results if r['winner'] == 'Vector')
    ties = sum(1 for r in overall_results if r['winner'] == 'Tie')
    
    print(f"  - SQL获胜: {sql_wins}次")
    print(f"  - 向量获胜: {vector_wins}次") 
    print(f"  - 平局: {ties}次")
    
    avg_sql_score = sum(r['sql_score'] for r in overall_results) / len(overall_results)
    avg_vector_score = sum(r['vector_score'] for r in overall_results) / len(overall_results)
    
    print(f"  - SQL平均分: {avg_sql_score:.1f}")
    print(f"  - 向量平均分: {avg_vector_score:.1f}")
    
    print(f"\n💡 结论:")
    if vector_wins > sql_wins:
        print("✅ 向量检索在当前数据集上表现更好，建议进行向量RAG升级")
    elif sql_wins > vector_wins:
        print("❌ SQL检索仍然更有效，暂不建议向量RAG升级")
    else:
        print("🤔 两种方法各有优势，需要进一步评估具体使用场景")

def sql_retrieve(person: str, cards: list):
    """现有的SQL检索方法"""
    db_path = "data/deepseek_tarot_knowledge.db"
    
    if not Path(db_path).exists():
        return {'person_readings': [], 'card_readings': []}
    
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

def vector_retrieve(person: str, cards: list, theme: str, collection):
    """真实的向量检索方法"""
    if not collection:
        return {'semantic_matches': [], 'error': 'Vector DB not available'}
    
    # 构建查询文本
    query_text = f"咨询者：{person} 牌组：{', '.join(cards)} 主题：{theme}"
    
    try:
        # 执行向量检索
        results = collection.query(
            query_texts=[query_text],
            n_results=5  # 检索更多结果
        )
        
        matches = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                matches.append({
                    'content': doc,
                    'person': metadata.get('person', 'unknown'),
                    'question': metadata.get('question', 'unknown'),
                    'cards': metadata.get('cards', 'unknown')
                })
        
        return {'semantic_matches': matches}
    except Exception as e:
        return {'semantic_matches': [], 'error': str(e)}

def evaluate_relevance(sql_results, vector_results, person, cards):
    """评估真实的相关性"""
    
    print(f"\n📊 详细结果分析：")
    
    # 分析SQL结果
    sql_person_count = len(sql_results.get('person_readings', []))
    sql_card_count = len(sql_results.get('card_readings', []))
    
    print(f"📈 SQL检索:")
    print(f"  - 个人历史解读: {sql_person_count}条")
    print(f"  - 相关牌组解读: {sql_card_count}条")
    
    if sql_person_count > 0:
        print(f"  - 个人解读示例: {sql_results['person_readings'][0][0][:50]}...")
    
    # 分析向量结果
    vector_matches = vector_results.get('semantic_matches', [])
    vector_count = len(vector_matches)
    
    print(f"🎯 向量检索:")
    print(f"  - 语义匹配结果: {vector_count}条")
    
    if vector_results.get('error'):
        print(f"  - 错误: {vector_results['error']}")
    elif vector_count > 0:
        print(f"  - 匹配示例: {vector_matches[0]['question'][:50]}...")
        
        # 分析匹配质量
        person_matches = sum(1 for m in vector_matches if m['person'] == person)
        card_matches = sum(1 for m in vector_matches 
                          if any(card.replace('(正位)', '').replace('(逆位)', '').strip() 
                                in m['cards'] for card in cards))
        
        print(f"  - 相同咨询者匹配: {person_matches}条")
        print(f"  - 相关牌组匹配: {card_matches}条")
    
    # 计算综合评分
    sql_score = sql_person_count * 0.4 + sql_card_count * 0.2
    vector_score = vector_count * 0.3
    
    if vector_count > 0 and not vector_results.get('error'):
        # 奖励向量检索的语义理解能力
        person_bonus = sum(1 for m in vector_matches if m['person'] == person) * 0.2
        card_bonus = sum(1 for m in vector_matches 
                        if any(card.replace('(正位)', '').replace('(逆位)', '').strip() 
                              in m['cards'] for card in cards)) * 0.1
        vector_score += person_bonus + card_bonus
    
    print(f"\n🏆 评分对比:")
    print(f"  - SQL总分: {sql_score:.1f}")
    print(f"  - 向量总分: {vector_score:.1f}")
    print(f"  - 胜者: {'Vector' if vector_score > sql_score else 'SQL' if sql_score > vector_score else 'Tie'}")
    
    return {
        'sql_score': sql_score,
        'vector_score': vector_score,
        'sql_person_count': sql_person_count,
        'sql_card_count': sql_card_count,
        'vector_count': vector_count,
        'winner': 'Vector' if vector_score > sql_score else 'SQL' if sql_score > vector_score else 'Tie'
    }

if __name__ == "__main__":
    print("🧪 向量RAG效果验证")
    print("这个脚本用于验证向量检索是否真的比SQL检索更好")
    print()
    
    # 运行对比测试
    test_sql_vs_vector_retrieval() 