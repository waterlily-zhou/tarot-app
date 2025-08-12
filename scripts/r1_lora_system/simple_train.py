#!/usr/bin/env python3
"""
简化版 DeepSeek R1 70B QLoRA 训练脚本
避免预加载模型的问题
"""
import os
import sys
import torch
import json
from pathlib import Path
import gc
from datetime import datetime

def detect_environment():
    print("🔍 检测环境...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} | 显存: {total_memory:.1f}GB")
        return True
    print("❌ 需要CUDA GPU环境")
    return False

def load_training_data(data_path="training_data.jsonl"):
    print(f"📂 加载训练数据: {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    try:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if 'messages' in item:
                    data.append(item)
        print(f"✅ 加载完成: {len(data)} 条训练数据")
        return data
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def train_70b_qlora():
    print("🌟 DeepSeek R1 70B QLoRA 塔罗牌微调系统")
    print("🔗 专为Lambda GPU云平台优化")
    print()
    print("🎯 开始DeepSeek R1 70B QLoRA微调")
    print("=" * 60)
    
    # 检测环境
    if not detect_environment():
        print("❌ 训练失败，请检查错误信息")
        return False
    
    # 加载训练数据
    training_data = load_training_data()
    if training_data is None:
        print("❌ 训练失败，请检查错误信息")
        return False
    
    try:
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM, 
            BitsAndBytesConfig,
            TrainingArguments
        )
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        from datasets import Dataset
        
        # 模型名称
        model_name = "deepseek-ai/DeepSeek-R1-0528"
        print(f"✅ 使用模型: {model_name}")
        
        # 加载tokenizer
        print("📥 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 4bit量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 加载模型 - 使用更稳定的参数
        print("📥 加载70B模型 (4bit量化)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory={0: "80GB"}
        )
        
        # LoRA配置
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query_key_value", "dense"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 应用LoRA
        print("🔧 应用LoRA配置...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 准备数据集
        print("📊 准备训练数据集...")
        dataset = Dataset.from_list(training_data)
        
        # 格式化函数
        def format_chat(example):
            messages = example['messages']
            text = ""
            for msg in messages:
                if msg['role'] == 'user':
                    text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg['role'] == 'assistant':
                    text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            return {"text": text}
        
        # 应用格式化
        dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir="./deepseek-r1-tarot-lora",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            warmup_steps=100,
            evaluation_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None,  # 禁用wandb
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # 创建训练器
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=2048,
            packing=False,
        )
        
        # 开始训练
        print("🚀 开始训练...")
        trainer.train()
        
        # 保存模型
        print("💾 保存模型...")
        trainer.save_model()
        
        print("✅ 训练完成！")
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    train_70b_qlora() 