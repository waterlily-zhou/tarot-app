#!/usr/bin/env python3
"""
找出标题为 unknown 的具体样本
"""

import json
from pathlib import Path

def find_unknown_samples():
    data_file = Path("data/finetune/tarot_readings.jsonl")
    
    if not data_file.exists():
        print("❌ 训练数据文件不存在")
        return
    
    print("🔍 查找 unknown 标题的样本...")
    
    unknown_samples = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)
                title = sample.get('metadata', {}).get('title', 'unknown')
                
                if title == 'unknown':
                    instruction = sample.get('instruction', '')
                    response = sample.get('response', '')
                    person = sample.get('metadata', {}).get('person', '')
                    
                    unknown_samples.append({
                        'line_number': i + 1,
                        'person': person,
                        'instruction_preview': instruction[:200] + "..." if len(instruction) > 200 else instruction,
                        'response_preview': response[:200] + "..." if len(response) > 200 else response,
                        'instruction_length': len(instruction),
                        'response_length': len(response)
                    })
                    
            except json.JSONDecodeError as e:
                print(f"❌ 第{i+1}行JSON解析错误: {e}")
    
    print(f"\n📊 找到 {len(unknown_samples)} 个 unknown 标题的样本:")
    print("=" * 80)
    
    for i, sample in enumerate(unknown_samples):
        print(f"\n🔍 Unknown 样本 #{i+1} (第{sample['line_number']}行):")
        print(f"   👤 咨询者: {sample['person']}")
        print(f"   📏 指令长度: {sample['instruction_length']} 字符")
        print(f"   📏 回答长度: {sample['response_length']} 字符")
        print(f"   📝 指令预览: {sample['instruction_preview']}")
        print(f"   📄 回答预览: {sample['response_preview']}")
        print("-" * 60)
    
    return unknown_samples

if __name__ == "__main__":
    find_unknown_samples() 