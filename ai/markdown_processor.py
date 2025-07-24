#!/usr/bin/env python3
"""
塔罗AI Markdown数据处理器
专门处理现有的Markdown格式课程笔记和星盘数据
"""

import json
import os
import re
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

class MarkdownTarotProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
    def parse_course_note(self, file_path: str) -> Dict:
        """
        解析课程笔记Markdown文件
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取文件名中的牌名
        filename = Path(file_path).stem
        card_name_match = re.match(r'^(.+?)\s+\w+', filename)
        card_name = card_name_match.group(1) if card_name_match else filename
        
        # 解析头部信息
        lines = content.split('\n')
        title = ""
        keywords = []
        card_number = None
        
        for i, line in enumerate(lines):
            if line.startswith('# '):
                title = line[2:].strip()
            elif line.startswith('关键字:'):
                keywords_text = line.split(':', 1)[1].strip()
                keywords = [k.strip() for k in keywords_text.split(',')]
            elif line.startswith('序号:'):
                number_text = line.split(':', 1)[1].strip()
                try:
                    card_number = int(number_text)
                except ValueError:
                    pass
        
        # 提取章节内容
        sections = self._extract_sections(content)
        
        # 生成唯一ID
        content_hash = hashlib.md5(f"{card_name}{title}".encode()).hexdigest()[:8]
        
        processed_data = {
            "id": f"course_{content_hash}",
            "type": "course_note",
            "card_name": card_name,
            "title": title,
            "card_number": card_number,
            "keywords": keywords,
            "sections": sections,
            "raw_content": content,
            "metadata": {
                "source_file": str(file_path),
                "processed_at": datetime.datetime.now().isoformat(),
                "content_length": len(content),
                "sections_count": len(sections)
            }
        }
        
        return processed_data
    
    def parse_birthchart(self, file_path: str) -> Dict:
        """
        解析星盘文件
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取文件名中的人名
        filename = Path(file_path).stem
        person_name_match = re.match(r'^(.+?)的星盘', filename)
        person_name = person_name_match.group(1) if person_name_match else filename
        
        # 提取主要星座信息
        astrological_info = self._extract_astrological_info(content)
        
        # 提取章节内容
        sections = self._extract_sections(content)
        
        # 生成唯一ID
        content_hash = hashlib.md5(f"{person_name}{content[:100]}".encode()).hexdigest()[:8]
        
        processed_data = {
            "id": f"chart_{content_hash}",
            "type": "birth_chart",
            "person_name": person_name,
            "astrological_info": astrological_info,
            "sections": sections,
            "raw_content": content,
            "metadata": {
                "source_file": str(file_path),
                "processed_at": datetime.datetime.now().isoformat(),
                "content_length": len(content),
                "sections_count": len(sections)
            }
        }
        
        return processed_data
    
    def _extract_sections(self, content: str) -> List[Dict]:
        """
        提取Markdown中的章节信息
        """
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith('## '):
                # 保存上一个章节
                if current_section:
                    sections.append({
                        "title": current_section,
                        "content": '\n'.join(current_content).strip(),
                        "length": len('\n'.join(current_content))
                    })
                
                # 开始新章节
                current_section = line[3:].strip()
                current_content = []
            
            elif current_section:
                current_content.append(line)
        
        # 添加最后一个章节
        if current_section:
            sections.append({
                "title": current_section,
                "content": '\n'.join(current_content).strip(),
                "length": len('\n'.join(current_content))
            })
        
        return sections
    
    def _extract_astrological_info(self, content: str) -> Dict:
        """
        从星盘内容中提取主要占星信息
        """
        info = {
            "main_signs": [],
            "houses": [],
            "aspects": [],
            "planets": []
        }
        
        # 使用正则表达式提取星座和宫位信息
        # 例如: "巨蟹五宫", "双鱼十二宫", "金星金牛二宫"
        sign_house_pattern = r'(\w+)([一二三四五六七八九十]{1,2})宫'
        planet_sign_house_pattern = r'(\w+)(\w+)([一二三四五六七八九十]{1,2})宫'
        
        for match in re.finditer(sign_house_pattern, content):
            sign = match.group(1)
            house = match.group(2)
            info["houses"].append({"sign": sign, "house": house})
        
        for match in re.finditer(planet_sign_house_pattern, content):
            planet = match.group(1)
            sign = match.group(2)
            house = match.group(3)
            info["planets"].append({"planet": planet, "sign": sign, "house": house})
        
        # 提取合相信息
        conjunction_pattern = r'(\w+)合(\w+)|(\w+)、(\w+)合相'
        for match in re.finditer(conjunction_pattern, content):
            if match.group(1) and match.group(2):
                info["aspects"].append({
                    "type": "conjunction",
                    "bodies": [match.group(1), match.group(2)]
                })
            elif match.group(3) and match.group(4):
                info["aspects"].append({
                    "type": "conjunction", 
                    "bodies": [match.group(3), match.group(4)]
                })
        
        return info
    
    def process_all_course_notes(self) -> List[Dict]:
        """
        处理所有课程笔记
        """
        course_notes_dir = self.data_dir / "course_notes"
        processed_notes = []
        
        if not course_notes_dir.exists():
            print(f"课程笔记目录不存在: {course_notes_dir}")
            return processed_notes
        
        for md_file in course_notes_dir.glob("*.md"):
            print(f"处理课程笔记: {md_file.name}")
            try:
                processed_note = self.parse_course_note(str(md_file))
                processed_notes.append(processed_note)
                
                # 保存处理后的数据
                output_file = self.processed_dir / f"{processed_note['id']}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_note, f, ensure_ascii=False, indent=2)
                
                print(f"已处理并保存: {output_file}")
                
            except Exception as e:
                print(f"处理文件 {md_file} 时出错: {e}")
        
        return processed_notes
    
    def process_all_birthcharts(self) -> List[Dict]:
        """
        处理所有星盘文件
        """
        birthcharts_dir = self.data_dir / "birthcharts"
        processed_charts = []
        
        if not birthcharts_dir.exists():
            print(f"星盘目录不存在: {birthcharts_dir}")
            return processed_charts
        
        for md_file in birthcharts_dir.glob("*.md"):
            print(f"处理星盘文件: {md_file.name}")
            try:
                processed_chart = self.parse_birthchart(str(md_file))
                processed_charts.append(processed_chart)
                
                # 保存处理后的数据
                output_file = self.processed_dir / f"{processed_chart['id']}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_chart, f, ensure_ascii=False, indent=2)
                
                print(f"已处理并保存: {output_file}")
                
            except Exception as e:
                print(f"处理文件 {md_file} 时出错: {e}")
        
        return processed_charts
    
    def create_embeddings_dataset(self) -> Dict:
        """
        为向量数据库创建嵌入数据集
        """
        embeddings_data = {
            "documents": [],
            "metadatas": [],
            "ids": []
        }
        
        # 处理所有JSON文件
        for json_file in self.processed_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data['type'] == 'course_note':
                # 为每个章节创建独立的文档
                for section in data['sections']:
                    embeddings_data["documents"].append(
                        f"牌名: {data['card_name']}\n"
                        f"章节: {section['title']}\n"
                        f"内容: {section['content']}"
                    )
                    embeddings_data["metadatas"].append({
                        "type": "course_note",
                        "card_name": data['card_name'],
                        "section_title": section['title'],
                        "keywords": data['keywords'],
                        "source_id": data['id']
                    })
                    embeddings_data["ids"].append(f"{data['id']}_{section['title']}")
                
            elif data['type'] == 'birth_chart':
                # 为每个章节创建独立的文档
                for section in data['sections']:
                    embeddings_data["documents"].append(
                        f"人物: {data['person_name']}\n"
                        f"星盘章节: {section['title']}\n"
                        f"内容: {section['content']}"
                    )
                    embeddings_data["metadatas"].append({
                        "type": "birth_chart",
                        "person_name": data['person_name'],
                        "section_title": section['title'],
                        "source_id": data['id']
                    })
                    embeddings_data["ids"].append(f"{data['id']}_{section['title']}")
        
        # 保存嵌入数据集
        embeddings_file = self.processed_dir / "embeddings_dataset.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
        
        print(f"嵌入数据集已保存: {embeddings_file}")
        print(f"总计文档数: {len(embeddings_data['documents'])}")
        
        return embeddings_data
    
    def generate_summary_report(self):
        """
        生成数据处理摘要报告
        """
        processed_files = list(self.processed_dir.glob("*.json"))
        
        course_notes = []
        birth_charts = []
        
        for json_file in processed_files:
            if json_file.name == "embeddings_dataset.json":
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data['type'] == 'course_note':
                course_notes.append(data)
            elif data['type'] == 'birth_chart':
                birth_charts.append(data)
        
        print("\n" + "="*50)
        print("数据处理摘要报告")
        print("="*50)
        print(f"课程笔记数量: {len(course_notes)}")
        print(f"星盘数量: {len(birth_charts)}")
        print(f"处理的文件总数: {len(processed_files) - 1}")  # 减去embeddings_dataset.json
        
        if course_notes:
            print("\n课程笔记详情:")
            for note in course_notes:
                print(f"  - {note['card_name']}: {len(note['sections'])}个章节, {note['metadata']['content_length']}字符")
        
        if birth_charts:
            print("\n星盘详情:")
            for chart in birth_charts:
                print(f"  - {chart['person_name']}: {len(chart['sections'])}个章节, {chart['metadata']['content_length']}字符")

if __name__ == "__main__":
    processor = MarkdownTarotProcessor()
    
    print("塔罗AI Markdown数据处理器")
    print("开始处理现有数据...")
    
    # 处理课程笔记
    course_notes = processor.process_all_course_notes()
    
    # 处理星盘文件
    birth_charts = processor.process_all_birthcharts()
    
    # 创建嵌入数据集
    embeddings_data = processor.create_embeddings_dataset()
    
    # 生成摘要报告
    processor.generate_summary_report()
    
    print("\n数据处理完成！")
    print("下一步：运行向量数据库设置脚本") 