#!/usr/bin/env python3
"""
塔罗AI RAG (检索增强生成) 系统
管理塔罗知识库、用户上下文，并提供语义检索功能
"""

import json
import os
import chromadb
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import re

class TarotRAGSystem:
    def __init__(self, data_dir: str = "data", embedding_model: str = "BAAI/bge-small-zh-v1.5"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.db_dir = self.data_dir / "vector_db"
        self.db_dir.mkdir(exist_ok=True)
        
        print(f"🤖 加载嵌入模型: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ 嵌入模型加载完成")
        
        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # 创建不同的向量集合
        self.collections = {
            "course_notes": self.chroma_client.get_or_create_collection("tarot_course_notes"),
            "readings": self.chroma_client.get_or_create_collection("tarot_readings"),
            "birthcharts": self.chroma_client.get_or_create_collection("birthcharts"),
            "user_context": self.chroma_client.get_or_create_collection("user_context")
        }
        
        print(f"🗃️  向量数据库初始化完成")
        
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量嵌入"""
        return self.embedding_model.encode(text).tolist()
    
    def chunk_text(self, text: str, max_length: int = 500, overlap: int = 50) -> List[str]:
        """将长文本分割成块"""
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # 如果当前段落很长，进一步分割
            if len(paragraph) > max_length:
                sentences = re.split(r'[。！？]', paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    sentence += "。"
                    
                    if len(current_chunk) + len(sentence) > max_length:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += sentence
            else:
                if len(current_chunk) + len(paragraph) > max_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def index_course_notes(self):
        """索引课程笔记"""
        print("📚 索引课程笔记...")
        
        notes_dir = self.processed_dir / "course_notes"
        if not notes_dir.exists():
            print("❌ 课程笔记目录不存在")
            return
        
        indexed_count = 0
        for json_file in notes_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                note_data = json.load(f)
            
            card_name = note_data['card_name']
            content = note_data['content']
            keywords = note_data.get('keywords', [])
            
            # 分块处理长文本
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{note_data['id']}_chunk_{i}"
                
                # 添加到向量数据库
                self.collections["course_notes"].add(
                    documents=[chunk],
                    metadatas=[{
                        "card_name": card_name,
                        "keywords": ",".join(keywords),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_file": str(json_file),
                        "type": "course_note"
                    }],
                    ids=[chunk_id]
                )
            
            indexed_count += 1
            print(f"✅ 已索引: {card_name} ({len(chunks)} 个文本块)")
        
        print(f"📚 课程笔记索引完成: {indexed_count} 个文件")
    
    def index_readings(self):
        """索引解牌记录"""
        print("🔮 索引解牌记录...")
        
        readings_dir = self.processed_dir / "readings"
        if not readings_dir.exists():
            print("❌ 解牌记录目录不存在")
            return
        
        indexed_count = 0
        for json_file in readings_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                reading_data = json.load(f)
            
            title = reading_data['title']
            interpretation = reading_data['interpretation']
            cards = reading_data.get('cards', [])
            date = reading_data.get('date', '')
            
            # 创建卡牌列表字符串
            card_list = ", ".join([card['name'] for card in cards])
            
            # 分块处理解牌内容
            chunks = self.chunk_text(interpretation)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{reading_data['id']}_chunk_{i}"
                
                self.collections["readings"].add(
                    documents=[chunk],
                    metadatas=[{
                        "title": title,
                        "cards": card_list,
                        "date": date,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_file": str(json_file),
                        "type": "reading"
                    }],
                    ids=[chunk_id]
                )
            
            indexed_count += 1
            print(f"✅ 已索引: {title} ({len(chunks)} 个文本块)")
        
        print(f"🔮 解牌记录索引完成: {indexed_count} 个文件")
    
    def index_birthcharts(self):
        """索引星盘数据"""
        print("⭐ 索引星盘数据...")
        
        charts_dir = self.processed_dir / "birthcharts"
        if not charts_dir.exists():
            print("❌ 星盘目录不存在")
            return
        
        indexed_count = 0
        for json_file in charts_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                chart_data = json.load(f)
            
            person_name = chart_data['person_name']
            content = chart_data['content']
            sections = chart_data.get('sections', {})
            
            # 索引完整内容
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{chart_data['id']}_chunk_{i}"
                
                self.collections["birthcharts"].add(
                    documents=[chunk],
                    metadatas=[{
                        "person_name": person_name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_file": str(json_file),
                        "type": "birthchart"
                    }],
                    ids=[chunk_id]
                )
            
            # 单独索引各个章节
            for section_name, section_content in sections.items():
                if section_content.strip():
                    section_id = f"{chart_data['id']}_section_{hashlib.md5(section_name.encode()).hexdigest()[:8]}"
                    
                    self.collections["birthcharts"].add(
                        documents=[section_content],
                        metadatas=[{
                            "person_name": person_name,
                            "section_name": section_name,
                            "source_file": str(json_file),
                            "type": "birthchart_section"
                        }],
                        ids=[section_id]
                    )
            
            indexed_count += 1
            print(f"✅ 已索引: {person_name} ({len(chunks)} 个文本块 + {len(sections)} 个章节)")
        
        print(f"⭐ 星盘数据索引完成: {indexed_count} 个文件")
    
    def add_user_context(self, user_id: str, context_data: Dict):
        """添加用户上下文信息"""
        context_id = f"user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 将上下文转换为可搜索的文本
        context_text = f"用户: {user_id}\n"
        if 'summary' in context_data:
            context_text += f"摘要: {context_data['summary']}\n"
        if 'cards' in context_data:
            context_text += f"相关卡牌: {', '.join(context_data['cards'])}\n"
        if 'themes' in context_data:
            context_text += f"主题: {', '.join(context_data['themes'])}\n"
        if 'notes' in context_data:
            context_text += f"备注: {context_data['notes']}\n"
        
        # 处理metadata，确保所有值都是标量类型
        metadata = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "context_type": context_data.get('type', 'general')
        }
        
        # 将列表转换为字符串
        for key, value in context_data.items():
            if isinstance(value, list):
                metadata[key] = ','.join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
            else:
                metadata[key] = str(value)
        
        self.collections["user_context"].add(
            documents=[context_text],
            metadatas=[metadata],
            ids=[context_id]
        )
        
        print(f"✅ 用户上下文已添加: {user_id}")
    
    def search_knowledge_base(self, query: str, collection_names: List[str] = None, n_results: int = 5) -> Dict:
        """搜索知识库"""
        if collection_names is None:
            collection_names = ["course_notes", "readings", "birthcharts"]
        
        results = {}
        
        for collection_name in collection_names:
            if collection_name in self.collections:
                try:
                    search_results = self.collections[collection_name].query(
                        query_texts=[query],
                        n_results=n_results
                    )
                    
                    # 格式化结果
                    formatted_results = []
                    if search_results['documents'][0]:  # 确保有结果
                        for i in range(len(search_results['documents'][0])):
                            formatted_results.append({
                                "content": search_results['documents'][0][i],
                                "metadata": search_results['metadatas'][0][i],
                                "distance": search_results['distances'][0][i] if 'distances' in search_results else None
                            })
                    
                    results[collection_name] = formatted_results
                    
                except Exception as e:
                    print(f"❌ 搜索 {collection_name} 时出错: {e}")
                    results[collection_name] = []
        
        return results
    
    def get_user_context(self, user_id: str, limit: int = 10) -> List[Dict]:
        """获取用户历史上下文"""
        try:
            search_results = self.collections["user_context"].query(
                query_texts=[f"用户: {user_id}"],
                n_results=limit,
                where={"user_id": user_id}
            )
            
            formatted_results = []
            if search_results['documents'][0]:
                for i in range(len(search_results['documents'][0])):
                    formatted_results.append({
                        "content": search_results['documents'][0][i],
                        "metadata": search_results['metadatas'][0][i]
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ 获取用户上下文时出错: {e}")
            return []
    
    def generate_context_for_query(self, query: str, user_id: str = None, max_context_length: int = 2000) -> str:
        """为查询生成相关上下文"""
        context_parts = []
        
        # 1. 搜索知识库
        knowledge_results = self.search_knowledge_base(query, n_results=3)
        
        # 添加课程笔记
        if knowledge_results.get("course_notes"):
            context_parts.append("=== 相关课程笔记 ===")
            for result in knowledge_results["course_notes"][:2]:
                card_name = result['metadata'].get('card_name', '未知')
                content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
                context_parts.append(f"【{card_name}】{content}")
        
        # 添加历史解牌
        if knowledge_results.get("readings"):
            context_parts.append("\n=== 相关解牌记录 ===")
            for result in knowledge_results["readings"][:2]:
                title = result['metadata'].get('title', '未知')
                content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
                context_parts.append(f"【{title}】{content}")
        
        # 2. 添加用户上下文
        if user_id:
            user_context = self.get_user_context(user_id, limit=3)
            if user_context:
                context_parts.append("\n=== 用户历史信息 ===")
                for ctx in user_context:
                    content = ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['content']
                    context_parts.append(content)
        
        # 组合上下文，确保不超过长度限制
        full_context = "\n".join(context_parts)
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "..."
        
        return full_context
    
    def get_database_stats(self) -> Dict:
        """获取数据库统计信息"""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = count
            except Exception as e:
                stats[name] = f"错误: {e}"
        
        return stats
    
    def clear_database(self, collection_name: str = None):
        """清空数据库"""
        if collection_name and collection_name in self.collections:
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass  # 集合可能不存在
            self.collections[collection_name] = self.chroma_client.get_or_create_collection(collection_name)
            print(f"✅ 已清空集合: {collection_name}")
        else:
            for name in self.collections.keys():
                try:
                    self.chroma_client.delete_collection(name)
                except Exception:
                    pass  # 集合可能不存在
                self.collections[name] = self.chroma_client.get_or_create_collection(name)
            print("✅ 已清空所有集合")

def main():
    """主函数 - 建立知识库"""
    rag = TarotRAGSystem()
    
    print("🚀 开始建立塔罗AI知识库...")
    print("=" * 50)
    
    # 检查现有数据库状态
    print("📊 当前数据库状态:")
    stats = rag.get_database_stats()
    for name, count in stats.items():
        print(f"  {name}: {count} 条记录")
    
    # 询问是否重新索引
    if any(isinstance(count, int) and count > 0 for count in stats.values()):
        choice = input("\n⚠️  检测到已有数据，是否重新索引？(y/n): ")
        if choice.lower() == 'y':
            rag.clear_database()
    
    # 开始索引
    print("\n🔄 开始索引数据...")
    
    # 索引各类数据
    rag.index_course_notes()
    rag.index_readings()
    rag.index_birthcharts()
    
    # 添加示例用户上下文
    print("\n👤 添加示例用户上下文...")
    rag.add_user_context("mel", {
        "type": "profile",
        "summary": "对塔罗和星座学习有深度兴趣，特别关注心轮能量和大爱主题",
        "cards": ["皇后", "星星", "圣杯国王"],
        "themes": ["心轮通道", "大爱", "灵性成长"],
        "notes": "M4 MacBook用户，希望本地运行AI模型"
    })
    
    # 显示最终统计
    print("\n📊 索引完成后的数据库状态:")
    final_stats = rag.get_database_stats()
    for name, count in final_stats.items():
        print(f"  {name}: {count} 条记录")
    
    print("\n✅ 塔罗AI知识库建立完成！")
    print("💡 你可以使用以下方法测试RAG系统:")
    print("  - rag.search_knowledge_base('皇后牌的含义')")
    print("  - rag.generate_context_for_query('心轮能量', 'mel')")
    
    return rag

if __name__ == "__main__":
    rag_system = main()
    
    # 简单的交互测试
    print("\n🧪 RAG系统测试")
    print("输入查询来测试知识库检索 (输入 'quit' 退出):")
    
    while True:
        query = input("\n🔍 查询: ").strip()
        if query.lower() in ['quit', 'exit', '退出']:
            break
        
        if query:
            print("📚 搜索结果:")
            results = rag_system.search_knowledge_base(query, n_results=2)
            
            for collection_name, items in results.items():
                if items:
                    print(f"\n--- {collection_name} ---")
                    for i, item in enumerate(items[:2]):
                        content = item['content'][:200] + "..." if len(item['content']) > 200 else item['content']
                        print(f"{i+1}. {content}")
            
            print("\n🎯 生成的上下文:")
            context = rag_system.generate_context_for_query(query, "mel")
            print(context[:500] + "..." if len(context) > 500 else context) 