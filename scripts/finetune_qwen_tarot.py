#!/usr/bin/env python3
"""
优化版Qwen微调脚本 - 针对24GB内存MacBook Air M4
"""

import os
import sys
import torch
import warnings
import json
from pathlib import Path
import gc
warnings.filterwarnings("ignore")

# 设置环境变量优化内存使用
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # 禁用MPS内存上限

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    print("✅ 所有依赖库已成功导入")
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    sys.exit(1)

def check_system_info():
    """检查系统信息"""
    if torch.backends.mps.is_available():
        print("🚀 使用 Apple Silicon MPS")
        device = "mps"
    else:
        print("💻 使用 CPU")
        device = "cpu"
    
    # 检查可用内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 总内存: {memory.total / (1024**3):.1f} GB")
        print(f"💾 可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"💾 使用率: {memory.percent}%")
        
        if memory.available / (1024**3) < 10:
            print("⚠️ 可用内存较少，建议关闭其他应用")
        else:
            print("✅ 内存充足，可以运行大模型")
            
    except ImportError:
        print("💡 安装 psutil 可查看详细内存信息: pip install psutil")
    
    return device

def clean_memory():
    """清理内存"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("🧹 内存清理完成")

def validate_and_clean_data(data_file: str, max_length: int = 4000):
    """验证和清理训练数据"""
    print(f"🔍 验证训练数据 (最大长度: {max_length} tokens)...")
    
    model_name = "Qwen/Qwen1.5-7B-Chat"
    print(f"📥 加载分词器: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    valid_samples = []
    rejected_samples = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)
                
                # 构建完整文本用于token计算
                full_text = sample['instruction'] + '\n\n' + sample['response']
                
                # 计算token长度
                tokens = tokenizer.encode(full_text, add_special_tokens=True)
                token_length = len(tokens)
                
                if token_length <= max_length:
                    valid_samples.append(sample)
                else:
                    # 智能截断而不是拒绝
                    instruction_tokens = tokenizer.encode(sample['instruction'], add_special_tokens=True)
                    max_response_tokens = max_length - len(instruction_tokens) - 20  # 预留特殊token空间
                    
                    if max_response_tokens > 200:  # 确保有足够空间给回答
                        response_tokens = tokenizer.encode(sample['response'], add_special_tokens=False)
                        if len(response_tokens) > max_response_tokens:
                            # 截断到句子边界
                            truncated_tokens = response_tokens[:max_response_tokens]
                            truncated_response = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                            
                            # 找到最后一个句号或者段落边界来优雅截断
                            last_period = max(
                                truncated_response.rfind('。'),
                                truncated_response.rfind('\n\n'),
                                truncated_response.rfind('！'),
                                truncated_response.rfind('？')
                            )
                            
                            if last_period > len(truncated_response) * 0.7:  # 如果句号位置合理
                                truncated_response = truncated_response[:last_period + 1]
                            
                            sample['response'] = truncated_response + "\n\n(解读内容已截断)"
                            valid_samples.append(sample)
                            print(f"✂️ 优雅截断样本 {i+1}: {sample.get('metadata', {}).get('title', 'unknown')}")
                        else:
                            valid_samples.append(sample)
                    else:
                        print(f"❌ 样本 {i+1} 指令过长，无法容纳: {sample.get('metadata', {}).get('title', 'unknown')}")
                        rejected_samples.append({
                            'line': i+1,
                            'token_length': token_length,
                            'title': sample.get('metadata', {}).get('title', 'unknown'),
                            'reason': 'instruction_too_long'
                        })
                    
            except Exception as e:
                print(f"❌ 样本 {i+1} 解析错误: {e}")
                rejected_samples.append({'line': i+1, 'error': str(e)})
    
    print(f"✅ 有效样本: {len(valid_samples)}")
    print(f"❌ 拒绝样本: {len(rejected_samples)}")
    
    if rejected_samples and len(rejected_samples) <= 5:
        print("\n被拒绝的样本详情:")
        for r in rejected_samples:
            if 'token_length' in r:
                print(f"  第{r['line']}行: {r['title']} ({r['token_length']} tokens)")
            else:
                print(f"  第{r['line']}行: {r.get('error', 'unknown error')}")
    
    return valid_samples, tokenizer

def preprocess_function(examples, tokenizer, max_length=4000):
    """预处理函数"""
    inputs = []
    labels = []
    
    for instruction, response in zip(examples['instruction'], examples['response']):
        # 构建对话格式
        text = f"{instruction}\n\n{response}"
        
        # 分词，使用严格的截断
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        inputs.append(tokenized['input_ids'])
        labels.append(tokenized['input_ids'].copy())
    
    return {
        'input_ids': inputs,
        'labels': labels
    }

def main():
    # 检查系统信息
    device = check_system_info()
    
    # 清理内存
    clean_memory()
    
    # 验证数据
    data_file = "data/finetune/tarot_readings.jsonl"
    if not Path(data_file).exists():
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行数据处理脚本")
        return
    
    # 验证和清理数据
    valid_samples, tokenizer = validate_and_clean_data(data_file, max_length=4000)
    
    if len(valid_samples) < 10:
        print("❌ 有效样本太少，无法进行训练")
        return
    
    print(f"\n🚀 开始微调，使用 {len(valid_samples)} 个样本")
    
    # 转换为Dataset格式
    dataset_dict = {
        'instruction': [s['instruction'] for s in valid_samples],
        'response': [s['response'] for s in valid_samples]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # 预处理数据
    def preprocess_wrapper(examples):
        return preprocess_function(examples, tokenizer, max_length=4000)
    
    dataset = dataset.map(
        preprocess_wrapper,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ 数据预处理完成，样本数: {len(dataset)}")
    
    # 清理内存
    clean_memory()
    
    # 加载模型 - 优化内存使用
    print(f"📥 加载 Qwen1.5-7B 模型...")
    model_name = "Qwen/Qwen1.5-7B-Chat"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if device == "mps" else torch.float16,  # MPS使用float32
            device_map=None,  # 先不自动分配，手动管理
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # 加载时减少CPU内存使用
            attn_implementation="eager"  # 使用传统注意力机制，更稳定
        )
        
        # 手动移动到MPS
        if device == "mps":
            model = model.to("mps")
        
        print(f"✅ 模型加载成功！参数量: {model.num_parameters():,}")
        print(f"📊 模型大小估算: ~{model.num_parameters() * 2 / (1024**3):.1f}GB (fp16)")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 尝试解决方案:")
        print("  1. 重启Terminal释放内存")
        print("  2. 设置 export PYTORCH_ENABLE_MPS_FALLBACK=1")
        print("  3. 临时关闭其他应用")
        return
    
    # 配置LoRA - 使用适中的参数
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # 适中的rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 全部注意力矩阵
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 清理内存
    clean_memory()
    
    # 训练参数 - 针对24GB内存优化
    training_args = TrainingArguments(
        output_dir="./models/qwen-tarot-24gb",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # 减小batch size以适应MPS内存限制
        gradient_accumulation_steps=8,  # 增加梯度累积补偿小batch size
        warmup_steps=10,
        learning_rate=5e-5,
        fp16=False,  # MPS不支持fp16混合精度
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no",  # 修复：使用新的参数名
        dataloader_num_workers=0,  # MPS不支持多进程
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        save_total_limit=2,  # 只保留最近2个checkpoint
        prediction_loss_only=True
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print("🏋️ 开始训练...")
    print("💡 训练期间请勿运行其他大型应用，确保内存充足")
    
    try:
        trainer.train()
        
        # 保存模型
        print("💾 保存模型...")
        trainer.save_model()
        tokenizer.save_pretrained("./models/qwen-tarot-24gb")
        
        print("🎉 训练完成！")
        print(f"📁 模型保存在: ./models/qwen-tarot-24gb")
        print(f"🎯 下一步: 运行 python scripts/test_qwen_tarot.py 测试模型")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("🔧 故障排除建议:")
        print("  1. 检查是否有其他大型应用占用内存")
        print("  2. 重启Terminal和Python环境")
        print("  3. 降低batch_size到1")
        print("  4. 检查MPS是否正常工作")

if __name__ == "__main__":
    main()