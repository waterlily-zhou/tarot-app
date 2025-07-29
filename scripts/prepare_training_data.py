#!/usr/bin/env python3
"""
最终修复版数据处理脚本 - 确保token长度符合要求
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

class TarotDataProcessorFinal:
    def __init__(self, readings_dir: str, max_chars: int = 6000):  # 大幅降低字符限制
        self.readings_dir = Path(readings_dir)
        self.output_dir = Path("data/finetune")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars  # 6000字符大约对应4000-5000个中文token
        
    def parse_md_file(self, md_path: Path) -> Optional[Dict]:
        """解析单个MD文件"""
        content = md_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # 提取标题 - 确保总是有标题
        title = lines[0].replace('# ', '').strip() if lines else md_path.stem
        
        metadata = {}
        content_start = 0
        
        # 解析多牌阵情况
        cards_data = {}
        spread_data = {}
        
        for i, line in enumerate(lines[1:], 1):
            if line.startswith('Cards'):
                if ':' in line:
                    cards_key, cards_content = line.split(':', 1)
                    cards_key = cards_key.strip()
                    cards_data[cards_key] = self._parse_cards(cards_content.strip())
                    
            elif line.startswith('Card '):
                if ':' in line:
                    cards_key, cards_content = line.split(':', 1)
                    cards_key = cards_key.strip().replace('Card ', 'Cards ')
                    cards_data[cards_key] = self._parse_cards(cards_content.strip())
                    
            elif line.startswith('Spread'):
                if ':' in line:
                    spread_key, spread_content = line.split(':', 1)
                    spread_key = spread_key.strip()
                    spread_data[spread_key] = spread_content.strip()
                    
            elif line.startswith('Person'):
                metadata['person'] = line.split(':', 1)[1].strip()
            elif line.startswith('Date'):
                metadata['date'] = line.split(':', 1)[1].strip()
            elif line.startswith('Reader'):
                metadata['reader'] = line.split(':', 1)[1].strip()
            elif line.strip() == '' and i > 8:
                if not content_start:
                    content_start = i + 1
        
        if not content_start:
            content_start = 10
        
        # 提取正文
        content_lines = []
        for line in lines[content_start:]:
            if not line.startswith('![') and line.strip():
                content_lines.append(line)
        
        analysis = '\n'.join(content_lines).strip()
        
        if not analysis:
            return None
        
        # 合并所有牌阵数据
        all_cards = []
        all_spreads = []
        
        if cards_data:
            for key in sorted(cards_data.keys()):
                all_cards.extend(cards_data[key])
        
        if spread_data:
            for key in sorted(spread_data.keys()):
                all_spreads.append(spread_data[key])
        
        return {
            'title': title,
            'person': metadata.get('person', ''),
            'cards': all_cards,
            'cards_detail': cards_data,
            'spread': '; '.join(all_spreads) if all_spreads else '',
            'spread_detail': spread_data,
            'date': metadata.get('date', ''),
            'reader': metadata.get('reader', ''),
            'analysis': analysis,
            'has_structured_cards': bool(cards_data),
            'has_structured_spreads': bool(spread_data),
            'source_file': md_path.name
        }
    
    def _parse_cards(self, cards_text: str) -> List[str]:
        """解析卡牌信息"""
        if not cards_text:
            return []
            
        cards = re.split(r'[,，;；、]', cards_text)
        
        cleaned_cards = []
        for card in cards:
            card = card.strip()
            if card and not card.startswith('Card') and not card.startswith('Spread'):
                cleaned_cards.append(card)
        
        return cleaned_cards
    
    def split_long_text_aggressive(self, text: str, max_chars: int = 6000) -> List[str]:
        """更激进的文本拆分策略"""
        if len(text) <= max_chars:
            return [text]
        
        print(f"🔄 拆分文本 ({len(text)} 字符 → 目标 ≤{max_chars})")
        
        # 1. 首先尝试按二级标题拆分
        sections = re.split(r'\n## ', text)
        if len(sections) > 1:
            parts = []
            current_part = sections[0]
            
            for section in sections[1:]:
                section = '## ' + section
                
                # 如果单个section就超过限制，需要进一步拆分
                if len(section) > max_chars:
                    # 先保存当前部分
                    if current_part.strip():
                        parts.append(current_part.strip())
                    
                    # 拆分这个超长section
                    subsections = self._split_by_paragraphs(section, max_chars)
                    parts.extend(subsections)
                    current_part = ""
                elif len(current_part) + len(section) <= max_chars:
                    current_part += '\n' + section
                else:
                    if current_part.strip():
                        parts.append(current_part.strip())
                    current_part = section
            
            if current_part.strip():
                parts.append(current_part.strip())
            
            # 验证所有部分都符合长度要求
            final_parts = []
            for part in parts:
                if len(part) <= max_chars:
                    final_parts.append(part)
                else:
                    # 进一步拆分
                    sub_parts = self._split_by_paragraphs(part, max_chars)
                    final_parts.extend(sub_parts)
            
            return final_parts
        
        # 2. 如果没有二级标题，按段落拆分
        return self._split_by_paragraphs(text, max_chars)
    
    def _split_by_paragraphs(self, text: str, max_chars: int) -> List[str]:
        """按段落拆分文本"""
        paragraphs = text.split('\n\n')
        parts = []
        current_part = ""
        
        for para in paragraphs:
            # 如果单个段落就超过限制，强制按句子拆分
            if len(para) > max_chars:
                if current_part.strip():
                    parts.append(current_part.strip())
                    current_part = ""
                
                # 按句子拆分
                sentences = re.split(r'[。！？\n]', para)
                temp_part = ""
                for sentence in sentences:
                    if sentence.strip():
                        sentence = sentence.strip() + '。'
                        if len(temp_part) + len(sentence) <= max_chars:
                            temp_part += sentence
                        else:
                            if temp_part.strip():
                                parts.append(temp_part.strip())
                            temp_part = sentence
                
                if temp_part.strip():
                    parts.append(temp_part.strip())
                    
            elif len(current_part) + len(para) + 2 <= max_chars:
                current_part += "\n\n" + para if current_part else para
            else:
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = para
        
        if current_part.strip():
            parts.append(current_part.strip())
        
        return parts if parts else [text[:max_chars]]
    
    def create_training_samples(self, parsed_data: List[Dict]) -> List[Dict]:
        """创建训练样本"""
        samples = []
        
        # 按person分组
        person_readings = {}
        for reading in parsed_data:
            person = reading['person']
            if person not in person_readings:
                person_readings[person] = []
            person_readings[person].append(reading)
        
        for reading in parsed_data:
            title = reading.get('title', reading.get('source_file', 'unknown'))
            analysis_length = len(reading['analysis'])
            
            if analysis_length > self.max_chars:
                print(f"📄 拆分长文本: {title} ({analysis_length} 字符)")
                
                # 使用更激进的拆分策略
                text_parts = self.split_long_text_aggressive(reading['analysis'], self.max_chars)
                
                for i, part in enumerate(text_parts):
                    part_title = f"{title} (第{i+1}部分)"
                    
                    # 验证拆分后的长度
                    if len(part) > self.max_chars:
                        print(f"⚠️ 拆分后仍过长: {part_title} ({len(part)} 字符)，强制截断")
                        part = part[:self.max_chars] + "..."
                    
                    instruction = self._create_instruction(reading, is_part=True, part_num=i+1, total_parts=len(text_parts))
                    
                    sample = {
                        "instruction": instruction,
                        "response": part,
                        "metadata": {
                            "person": reading['person'],
                            "cards": reading['cards'],
                            "spread": reading['spread'],
                            "title": part_title,
                            "original_title": title,
                            "is_split": True,
                            "part_num": i+1,
                            "total_parts": len(text_parts),
                            "source_file": reading['source_file']
                        }
                    }
                    samples.append(sample)
            else:
                # 正常长度的文本
                instruction = self._create_instruction(reading)
                
                sample = {
                    "instruction": instruction,
                    "response": reading['analysis'],
                    "metadata": {
                        "person": reading['person'],
                        "cards": reading['cards'],
                        "spread": reading['spread'],
                        "title": title,
                        "is_split": False,
                        "source_file": reading['source_file']
                    }
                }
                samples.append(sample)
        
        return samples
    
    def _create_instruction(self, reading: Dict, is_part: bool = False, part_num: int = 1, total_parts: int = 1) -> str:
        """创建指令 - 简化版本减少token消耗"""
        title = reading.get('title', reading.get('source_file', 'unknown'))
        
        # 简化指令以减少token消耗
        if reading['cards']:
            cards_str = "；".join(reading['cards'][:5])  # 最多显示5张牌
            spread_str = reading['spread'] if reading['spread'] else "牌阵未指定"
            
            instruction = f"""塔罗解读：
咨询者：{reading['person']}
问题：{title}
牌阵：{spread_str}
牌：{cards_str}"""
        else:
            instruction = f"""塔罗解读：
咨询者：{reading['person']}
问题：{title}"""

        if is_part:
            instruction += f"\n(第{part_num}/{total_parts}部分)"
        
        instruction += "\n\n请提供专业解读："
        
        return instruction
    
    def process_all_files(self) -> str:
        """处理所有MD文件"""
        parsed_data = []
        
        for md_file in self.readings_dir.glob("*.md"):
            try:
                data = self.parse_md_file(md_file)
                if data:
                    parsed_data.append(data)
                    print(f"✅ 处理: {md_file.name} - '{data['title']}' ({len(data['analysis'])} 字符)")
                else:
                    print(f"⏭️ 跳过: {md_file.name} (无有效内容)")
            except Exception as e:
                print(f"❌ 错误: {md_file.name} - {e}")
        
        # 创建训练样本
        samples = self.create_training_samples(parsed_data)
        
        # 保存为JSONL
        output_file = self.output_dir / "tarot_readings.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n🎉 处理完成！")
        print(f"📊 总计解读: {len(parsed_data)} 条")
        print(f"📝 训练样本: {len(samples)} 条")
        print(f"💾 输出文件: {output_file}")
        
        return str(output_file)

if __name__ == "__main__":
    processor = TarotDataProcessorFinal("data/Readings 6c05b8c41bcf40f9be4f7dd503141fd2", max_chars=6000)
    processor.process_all_files()