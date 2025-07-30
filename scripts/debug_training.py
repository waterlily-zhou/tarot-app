#!/usr/bin/env python3
"""
调试版微调脚本 - 增加详细进度输出
"""

import os
import sys
import torch
import time
import json
from pathlib import Path

print("🚀 开始调试版训练...")
print(f"⏰ 开始时间: {time.strftime('%H:%M:%S')}")

# 设置环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    print("✅ 所有依赖库导入成功")
except ImportError as e:
    print(f"❌ 依赖库导入失败: {e}")
    sys.exit(1)

def debug_system():
    """调试系统信息"""
    print("\n📊 系统信息:")
    
    # 检查设备
    if torch.backends.mps.is_available():
        print("  ✅ MPS 可用")
        device = "mps"
    else:
        print("  ⚠️ MPS 不可用，使用 CPU")
        device = "cpu"
    
    # 检查内存
    import psutil
    memory = psutil.virtual_memory()
    print(f"  💾 总内存: {memory.total / (1024**3):.1f} GB")
    print(f"  💾 可用内存: {memory.available / (1024**3):.1f} GB")
    print(f"  💾 使用率: {memory.percent}%")
    
    return device

def load_and_check_data():
    """加载并检查数据"""
    print("\n📄 加载训练数据...")
    
    data_file = "data/finetune/tarot_readings.jsonl"
    if not Path(data_file).exists():
        print(f"❌ 数据文件不存在: {data_file}")
        return None
    
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # 只加载前5个样本进行测试
                break
            samples.append(json.loads(line))
    
    print(f"✅ 加载了 {len(samples)} 个样本 (测试用)")
    for i, sample in enumerate(samples):
        title = sample.get('metadata', {}).get('title', 'unknown')
        print(f"  样本 {i+1}: {title}")
    
    return samples

def test_model_loading():
    """测试模型加载"""
    print("\n🤖 测试模型加载...")
    
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    print(f"📥 加载分词器: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ 分词器加载成功")
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        return None, None
    
    print(f"📥 加载模型: {model_name}")
    start_time = time.time()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        load_time = time.time() - start_time
        print(f"✅ 模型加载成功，耗时: {load_time:.1f}秒")
        print(f"📊 模型参数量: {model.num_parameters():,}")
        
        # 移动到MPS
        device = debug_system()
        if device == "mps":
            print("🔄 移动模型到MPS...")
            start_time = time.time()
            model = model.to("mps")
            move_time = time.time() - start_time
            print(f"✅ 模型移动到MPS成功，耗时: {move_time:.1f}秒")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def test_data_processing(samples, tokenizer):
    """测试数据处理"""
    print("\n🔧 测试数据处理...")
    
    dataset_dict = {
        'instruction': [s['instruction'] for s in samples],
        'response': [s['response'] for s in samples]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    print(f"✅ 创建数据集，样本数: {len(dataset)}")
    
    # 测试预处理
    def preprocess_function(examples):
        inputs = []
        labels = []
        
        for instruction, response in zip(examples['instruction'], examples['response']):
            text = f"{instruction}\n\n{response}"
            
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=512,  # 使用较小的长度进行测试
                padding=False,
                return_tensors=None
            )
            
            inputs.append(tokenized['input_ids'])
            labels.append(tokenized['input_ids'].copy())
        
        return {'input_ids': inputs, 'labels': labels}
    
    print("🔄 处理数据...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    print(f"✅ 数据处理完成，处理后样本数: {len(processed_dataset)}")
    
    # 打印第一个样本的token长度
    first_sample = processed_dataset[0]
    print(f"📊 第一个样本token长度: {len(first_sample['input_ids'])}")
    
    return processed_dataset

def test_lora_setup(model):
    """测试LoRA设置"""
    print("\n⚙️ 测试LoRA设置...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # 更小的rank用于测试
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # 只微调部分模块
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    print("✅ LoRA配置成功")
    model.print_trainable_parameters()
    
    return model

def test_trainer_setup(model, tokenizer, dataset):
    """测试Trainer设置"""
    print("\n🏋️ 测试Trainer设置...")
    
    training_args = TrainingArguments(
        output_dir="./models/debug-test",
        num_train_epochs=1,  # 只训练1个epoch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=1,
        learning_rate=5e-5,
        fp16=False,
        logging_steps=1,  # 每步都输出日志
        save_strategy="no",  # 不保存checkpoint，只测试
        eval_strategy="no",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        max_grad_norm=1.0,
        dataloader_pin_memory=False
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("✅ Trainer设置成功")
    return trainer

def main():
    """主函数"""
    print("🎯 开始调试流程...")
    
    # 1. 检查系统
    device = debug_system()
    
    # 2. 加载数据
    samples = load_and_check_data()
    if not samples:
        return
    
    # 3. 测试模型加载
    model, tokenizer = test_model_loading()
    if model is None:
        return
    
    # 4. 测试数据处理
    dataset = test_data_processing(samples, tokenizer)
    
    # 5. 测试LoRA
    model = test_lora_setup(model)
    
    # 6. 测试Trainer
    trainer = test_trainer_setup(model, tokenizer, dataset)
    
    # 7. 开始训练测试
    print("\n🚀 开始训练测试...")
    print("注意：这只是一个短时间的训练测试")
    
    try:
        trainer.train()
        print("🎉 训练测试完成！")
        
        # 检查输出目录
        output_dir = Path("./models/debug-test")
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"📁 生成了 {len(files)} 个文件")
            for f in files[:5]:  # 只显示前5个文件
                print(f"  - {f.name}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 