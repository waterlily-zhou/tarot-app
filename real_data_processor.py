#!/usr/bin/env python3
"""
真实塔罗数据处理器
处理用户的真实数据：Notion导出的MD文件 + 图片
"""

import json
import os
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import shutil

class RealTarotDataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # 创建处理后的目录结构
        (self.processed_dir / "readings").mkdir(exist_ok=True)
        (self.processed_dir / "course_notes").mkdir(exist_ok=True)
        (self.processed_dir / "birthcharts").mkdir(exist_ok=True)
        (self.processed_dir / "all_cards").mkdir(exist_ok=True)
    
    def extract_cards_from_text(self, text: str) -> List[str]:
        """从文本中提取卡牌名称"""
        # 匹配 "Cards: " 后面的内容
        cards_match = re.search(r'Cards:\s*(.+)', text)
        if cards_match:
            cards_text = cards_match.group(1)
            # 按逗号分割并清理
            cards = [card.strip() for card in cards_text.split(',') if card.strip()]
            return cards
        return []
    
    def extract_date_from_text(self, text: str) -> Optional[str]:
        """从文本中提取日期"""
        date_match = re.search(r'Date:\s*(.+)', text)
        if date_match:
            return date_match.group(1).strip()
        return None
    
    def extract_title_from_text(self, text: str) -> str:
        """从markdown文本中提取标题"""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "未知标题"
    
    def process_reading_folder(self, folder_path: Path) -> Optional[Dict]:
        """处理单个解牌文件夹"""
        try:
            # 查找MD文件和图片文件
            md_files = list(folder_path.glob("*.md"))
            image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
            
            if not md_files:
                print(f"警告：文件夹 {folder_path} 中没有找到MD文件")
                return None
            
            md_file = md_files[0]  # 取第一个MD文件
            image_file = image_files[0] if image_files else None
            
            # 读取MD文件内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取信息
            title = self.extract_title_from_text(content)
            cards = self.extract_cards_from_text(content)
            date = self.extract_date_from_text(content)
            
            # 分离解牌内容（去掉元数据部分）
            content_lines = content.split('\n')
            content_start = 0
            for i, line in enumerate(content_lines):
                if line.strip().startswith('![') or (line.strip() and not line.startswith('#') and not line.startswith('Cards:') and not line.startswith('Date:')):
                    content_start = i
                    break
            
            interpretation = '\n'.join(content_lines[content_start:]).strip()
            
            # 创建处理后的数据结构
            processed_data = {
                "id": f"reading_{hashlib.md5(f'{title}{date}'.encode()).hexdigest()[:8]}",
                "title": title,
                "date": date,
                "cards": [{"name": card, "position": f"pos_{i+1}"} for i, card in enumerate(cards)],
                "interpretation": interpretation,
                "original_folder": str(folder_path),
                "image_path": str(image_file) if image_file else None,
                "metadata": {
                    "processed_at": datetime.datetime.now().isoformat(),
                    "card_count": len(cards),
                    "has_image": image_file is not None
                }
            }
            
            return processed_data
            
        except Exception as e:
            print(f"处理文件夹 {folder_path} 时出错: {e}")
            return None
    
    def process_course_note(self, file_path: Path) -> Optional[Dict]:
        """处理课程笔记文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取标题（卡牌名称）
            title = self.extract_title_from_text(content)
            
            # 提取关键字
            keywords = []
            keywords_match = re.search(r'关键字:\s*(.+)', content)
            if keywords_match:
                keywords_text = keywords_match.group(1)
                keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            
            # 提取序号
            sequence_match = re.search(r'序号:\s*(\d+)', content)
            sequence = int(sequence_match.group(1)) if sequence_match else None
            
            processed_data = {
                "id": f"note_{title.replace(' ', '_').lower()}",
                "card_name": title,
                "content": content,
                "keywords": keywords,
                "sequence": sequence,
                "original_file": str(file_path),
                "metadata": {
                    "processed_at": datetime.datetime.now().isoformat(),
                    "content_length": len(content)
                }
            }
            
            return processed_data
            
        except Exception as e:
            print(f"处理课程笔记 {file_path} 时出错: {e}")
            return None
    
    def process_birthchart(self, file_path: Path) -> Optional[Dict]:
        """处理星盘文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            title = self.extract_title_from_text(content)
            
            # 提取主要章节
            sections = {}
            current_section = None
            current_content = []
            
            for line in content.split('\n'):
                if line.startswith('## '):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = line[3:].strip()
                    current_content = []
                elif current_section:
                    current_content.append(line)
            
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            
            processed_data = {
                "id": f"birthchart_{title.replace(' ', '_').lower()}",
                "person_name": title,
                "content": content,
                "sections": sections,
                "original_file": str(file_path),
                "metadata": {
                    "processed_at": datetime.datetime.now().isoformat(),
                    "content_length": len(content),
                    "section_count": len(sections)
                }
            }
            
            return processed_data
            
        except Exception as e:
            print(f"处理星盘文件 {file_path} 时出错: {e}")
            return None
    
    def save_processed_data(self, data: Dict, category: str):
        """保存处理后的数据"""
        filename = f"{data['id']}.json"
        filepath = self.processed_dir / category / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"{category} 数据已保存: {filepath}")
    
    def process_all_readings(self):
        """处理所有解牌记录"""
        readings_dir = self.data_dir / "readings"
        if not readings_dir.exists():
            print("readings 目录不存在")
            return
        
        processed_count = 0
        for folder in readings_dir.iterdir():
            if folder.is_dir():
                processed_data = self.process_reading_folder(folder)
                if processed_data:
                    self.save_processed_data(processed_data, "readings")
                    processed_count += 1
        
        print(f"共处理了 {processed_count} 个解牌记录")
    
    def process_all_course_notes(self):
        """处理所有课程笔记"""
        notes_dir = self.data_dir / "course_notes"
        if not notes_dir.exists():
            print("course_notes 目录不存在")
            return
        
        processed_count = 0
        for file in notes_dir.glob("*.md"):
            processed_data = self.process_course_note(file)
            if processed_data:
                self.save_processed_data(processed_data, "course_notes")
                processed_count += 1
        
        print(f"共处理了 {processed_count} 个课程笔记")
    
    def process_all_birthcharts(self):
        """处理所有星盘文件"""
        charts_dir = self.data_dir / "birthcharts"
        if not charts_dir.exists():
            print("birthcharts 目录不存在")
            return
        
        processed_count = 0
        for file in charts_dir.glob("*.md"):
            processed_data = self.process_birthchart(file)
            if processed_data:
                self.save_processed_data(processed_data, "birthcharts")
                processed_count += 1
        
        print(f"共处理了 {processed_count} 个星盘文件")
    
    def collect_all_cards(self):
        """收集所有出现过的卡牌名称"""
        all_cards = set()
        
        # 从解牌记录中收集
        readings_dir = self.processed_dir / "readings"
        for file in readings_dir.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for card in data.get('cards', []):
                all_cards.add(card['name'])
        
        # 从课程笔记中收集
        notes_dir = self.processed_dir / "course_notes"
        for file in notes_dir.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_cards.add(data['card_name'])
        
        # 保存卡牌列表
        cards_list = sorted(list(all_cards))
        cards_data = {
            "all_cards": cards_list,
            "total_count": len(cards_list),
            "created_at": datetime.datetime.now().isoformat()
        }
        
        with open(self.processed_dir / "all_cards" / "card_list.json", 'w', encoding='utf-8') as f:
            json.dump(cards_data, f, ensure_ascii=False, indent=2)
        
        print(f"共发现 {len(cards_list)} 种不同的卡牌")
        return cards_list
    
    def generate_summary_report(self):
        """生成处理摘要报告"""
        report = {
            "processed_at": datetime.datetime.now().isoformat(),
            "readings": len(list((self.processed_dir / "readings").glob("*.json"))),
            "course_notes": len(list((self.processed_dir / "course_notes").glob("*.json"))),
            "birthcharts": len(list((self.processed_dir / "birthcharts").glob("*.json"))),
            "total_files": 0
        }
        
        report["total_files"] = report["readings"] + report["course_notes"] + report["birthcharts"]
        
        with open(self.processed_dir / "processing_summary.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("\n=== 数据处理摘要 ===")
        print(f"解牌记录: {report['readings']} 个")
        print(f"课程笔记: {report['course_notes']} 个") 
        print(f"星盘文件: {report['birthcharts']} 个")
        print(f"总计: {report['total_files']} 个文件")
        
        return report

if __name__ == "__main__":
    processor = RealTarotDataProcessor()
    
    print("🔮 塔罗AI真实数据处理器")
    print("=" * 50)
    
    # 处理所有数据
    print("1. 处理解牌记录...")
    processor.process_all_readings()
    
    print("\n2. 处理课程笔记...")
    processor.process_all_course_notes()
    
    print("\n3. 处理星盘文件...")
    processor.process_all_birthcharts()
    
    print("\n4. 收集所有卡牌...")
    cards = processor.collect_all_cards()
    
    print("\n5. 生成摘要报告...")
    processor.generate_summary_report()
    
    print(f"\n✅ 数据处理完成！处理后的数据保存在: {processor.processed_dir}")
    print("\n📋 发现的卡牌:")
    for i, card in enumerate(cards[:10]):  # 只显示前10个
        print(f"  {i+1}. {card}")
    if len(cards) > 10:
        print(f"  ... 还有 {len(cards) - 10} 个卡牌") 