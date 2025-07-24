#!/usr/bin/env python3
"""
RAG系统测试脚本
"""

from rag_system import TarotRAGSystem

def test_rag_system():
    print("🧪 测试塔罗AI RAG系统")
    print("=" * 40)
    
    # 初始化RAG系统
    rag = TarotRAGSystem()
    
    # 测试查询
    test_queries = [
        "皇后牌的含义",
        "心轮能量",
        "巨蟹座的特点",
        "2025年运程"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        
        # 搜索知识库
        results = rag.search_knowledge_base(query, n_results=2)
        
        print("📚 搜索结果:")
        for collection_name, items in results.items():
            if items:
                print(f"\n--- {collection_name} ---")
                for i, item in enumerate(items[:1]):  # 只显示第一个结果
                    content = item['content'][:150] + "..." if len(item['content']) > 150 else item['content']
                    print(f"{i+1}. {content}")
        
        # 生成上下文
        context = rag.generate_context_for_query(query, "mel")
        print(f"\n🎯 生成的上下文长度: {len(context)} 字符")
        
        print("-" * 40)

if __name__ == "__main__":
    test_rag_system() 