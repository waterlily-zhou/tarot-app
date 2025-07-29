#!/usr/bin/env python3
"""
诊断训练数据中的问题样本
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

def diagnose_training_data():
    data_file = Path("data/finetune/tarot_readings.jsonl")
    
    if not data_file.exists():
        print("❌ 训练数据文件不存在")
        return
    
    print("🔍 诊断训练数据...")
    
    # 加载分词器来测试长度
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
    except:
        print("⚠️ 无法加载分词器，仅分析字符长度")
        tokenizer = None
    
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)
                samples.append((i, sample))
            except json.JSONDecodeError as e:
                print(f"❌ 第{i+1}行JSON解析错误: {e}")
    
    print(f"📊 总样本数: {len(samples)}")
    
    issues = []
    
    for i, (line_num, sample) in enumerate(samples):
        try:
            instruction = sample.get('instruction', '')
            response = sample.get('response', '')
            
            # 检查数据类型
            if not isinstance(instruction, str):
                issues.append(f"第{line_num+1}行: instruction不是字符串，类型: {type(instruction)}")
                continue
                
            if not isinstance(response, str):
                issues.append(f"第{line_num+1}行: response不是字符串，类型: {type(response)}")
                continue
            
            # 检查长度
            char_len = len(instruction) + len(response)
            
            if tokenizer:
                # 模拟完整的训练格式
                full_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
                try:
                    tokens = tokenizer.encode(full_text)
                    token_len = len(tokens)
                    
                    print(f"样本 {i+1:2d}: 字符={char_len:5d}, Token={token_len:5d} - {sample.get('metadata', {}).get('title', 'unknown')[:50]}")
                    
                    if token_len > 8192:  # 超长样本
                        issues.append(f"第{line_num+1}行: Token长度过长 ({token_len}), 标题: {sample.get('metadata', {}).get('title', 'unknown')}")
                        
                except Exception as e:
                    issues.append(f"第{line_num+1}行: 分词失败 - {e}")
            else:
                print(f"样本 {i+1:2d}: 字符={char_len:5d} - {sample.get('metadata', {}).get('title', 'unknown')[:50]}")
                
                if char_len > 20000:  # 超长样本（字符级别）
                    issues.append(f"第{line_num+1}行: 字符长度过长 ({char_len}), 标题: {sample.get('metadata', {}).get('title', 'unknown')}")
            
            # 检查空数据
            if not instruction.strip() or not response.strip():
                issues.append(f"第{line_num+1}行: 包含空的instruction或response")
                
        except Exception as e:
            issues.append(f"第{line_num+1}行: 处理时出错 - {e}")
    
    if issues:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ 数据格式检查通过")
    
    return issues

if __name__ == "__main__":
    diagnose_training_data() 