#!/usr/bin/env python3
"""
塔罗AI数据准备脚本
将解牌记录和课程笔记转换为结构化格式
"""

import json
import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

class TarotDataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.data_dir / "readings").mkdir(exist_ok=True)
        (self.data_dir / "course_notes").mkdir(exist_ok=True)
        (self.data_dir / "card_images").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
    
    def process_reading_record(self, 
                             theme: str,
                             cards_info: List[Dict],  # [{"position": "past", "card": "愚者", "reversed": False}]
                             interpretation: str,
                             date: Optional[str] = None,
                             reader: Optional[str] = None,
                             querent_context: Optional[str] = None) -> Dict:
        """
        处理单次解牌记录
        """
        if date is None:
            date = datetime.datetime.now().isoformat()
        
        # 生成唯一ID
        content_hash = hashlib.md5(f"{theme}{date}{interpretation}".encode()).hexdigest()[:8]
        
        record = {
            "id": f"reading_{content_hash}",
            "theme": theme,
            "date": date,
            "reader": reader,
            "querent_context": querent_context,
            "cards": cards_info,
            "interpretation": interpretation,
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "card_count": len(cards_info),
                "spread_type": self._detect_spread_type(cards_info)
            }
        }
        
        return record
    
    def process_course_note(self,
                          card_name: str,
                          content: str,
                          category: str = "general",  # general, love, career, etc.
                          keywords: List[str] = None) -> Dict:
        """
        处理课程笔记
        """
        if keywords is None:
            keywords = []
        
        note = {
            "id": f"note_{card_name.replace(' ', '_').lower()}_{category}",
            "card_name": card_name,
            "category": category,
            "content": content,
            "keywords": keywords,
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "content_length": len(content)
            }
        }
        
        return note
    
    def _detect_spread_type(self, cards_info: List[Dict]) -> str:
        """根据牌的数量和位置检测牌阵类型"""
        count = len(cards_info)
        
        if count == 1:
            return "single_card"
        elif count == 3:
            return "three_card"
        elif count == 5:
            return "five_card"
        elif count == 10:
            return "celtic_cross"
        else:
            return f"custom_{count}_card"
    
    def save_reading(self, reading_data: Dict):
        """保存解牌记录"""
        filename = f"{reading_data['id']}.json"
        filepath = self.data_dir / "readings" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(reading_data, f, ensure_ascii=False, indent=2)
        
        print(f"解牌记录已保存: {filepath}")
    
    def save_course_note(self, note_data: Dict):
        """保存课程笔记"""
        filename = f"{note_data['id']}.json"
        filepath = self.data_dir / "course_notes" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(note_data, f, ensure_ascii=False, indent=2)
        
        print(f"课程笔记已保存: {filepath}")
    
    def batch_process_from_text(self, input_file: str):
        """
        从文本文件批量处理数据
        期待格式：每个记录用 "---" 分隔
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 这里需要根据你的具体数据格式来调整
        # 现在先提供一个示例框架
        sections = content.split('---')
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # 解析每个section的内容
            # 你需要根据实际数据格式来修改这部分
            lines = section.split('\n')
            print(f"处理section: {lines[0][:50]}...")
    
    def create_sample_data(self):
        """创建示例数据，帮助理解格式"""
        # 示例解牌记录
        sample_reading = self.process_reading_record(
            theme="2025年运程",
            cards_info=[
                {"position": "past", "card": "圣杯国王", "reversed": False},
                {"position": "present", "card": "宝剑三", "reversed": False},
                {"position": "future", "card": "星币七", "reversed": True}
            ],
            interpretation="过去的情感稳定为现在的挑战做了准备，未来需要注意投资谨慎...",
            reader="示例占卜师"
        )
        self.save_reading(sample_reading)
        
        # 示例课程笔记
        sample_note = self.process_course_note(
            card_name="皇后",
            content="皇后代表心轮，女神，造梦，在心轮的绿色波段中...",
            category="心轮通道",
            keywords=["心轮", "女神", "创造", "母性"]
        )
        self.save_course_note(sample_note)
        
        print("示例数据已创建完成！")

if __name__ == "__main__":
    processor = TarotDataProcessor()
    
    print("塔罗AI数据准备工具")
    print("请选择操作：")
    print("1. 创建示例数据")
    print("2. 从文件批量处理")
    print("3. 手动输入单条记录")
    
    choice = input("请输入选择 (1-3): ")
    
    if choice == "1":
        processor.create_sample_data()
    elif choice == "2":
        input_file = input("请输入源文件路径: ")
        if os.path.exists(input_file):
            processor.batch_process_from_text(input_file)
        else:
            print("文件不存在！")
    elif choice == "3":
        # 手动输入的交互式界面
        theme = input("解牌主题: ")
        interpretation = input("解牌内容: ")
        # 简化版，实际可以做更复杂的输入
        cards_info = [{"position": "center", "card": "示例牌", "reversed": False}]
        
        reading = processor.process_reading_record(theme, cards_info, interpretation)
        processor.save_reading(reading)
    else:
        print("无效选择！") 