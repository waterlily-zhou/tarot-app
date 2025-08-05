#!/usr/bin/env python3
"""
DeepSeek R1 70B QLoRA 微调脚本 - 优化版
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
                # 保留原始messages结构 ✅
                if 'messages' in item:
                    data.append(item)
        print(f"✅ 加载完成: {len(data)} 条训练数据")
        return data
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def setup_70b_model():
    print("🚀 设置DeepSeek R1 70B模型...")
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
        
        # 使用官方模型名称 ✅
        model_name = "deepseek-ai/DeepSeek-R1-0528"
        print(f"✅ 使用模型: {model_name}")
        
        # 加载tokenizer
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
            bnb_4bit_compute_dtype=torch.bfloat16  # 更稳定 ✅
        )
        
        # 加载模型
        print("📥 加载70B模型 (4bit量化)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # 更稳定 ✅
            attn_implementation="flash_attention_2",  # 加速20% ✅
            use_cache=False
        )
        
        # 正确的LoRA配置 ✅
        lora_config = LoraConfig(
            r=8,                    # 降低rank节省显存
            lora_alpha=16,
            target_modules=["query_key_value", "dense"],  # GPT-NeoX架构
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("💡 请安装: pip install transformers peft bitsandbytes trl")
        return None, None
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def format_chat(example):
    """使用tokenizer内置的聊天模板格式化数据 ✅"""
    messages = example["messages"]
    return {"text": tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )}

def train_70b_qlora():
    print("🎯 开始DeepSeek R1 70B QLoRA微调")
    print("="*60)
    
    # 1. 环境检测
    if not detect_environment():
        return False
    
    # 2. 加载数据
    training_data = load_training_data()
    if not training_data:
        return False
    
    # 3. 设置模型
    model, tokenizer = setup_70b_model()
    if model is None:
        return False
    # 3.1  将tokenizer提升为全局，供 format_chat 使用
    globals()['tokenizer'] = tokenizer
    
    try:
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import Dataset
        
        # 4. 准备数据集
        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(format_chat)  # 正确格式化 ✅
        print(f"📊 格式化数据集: {len(dataset)} 条数据")
        
        # 5. 优化的训练参数 ✅
        training_args = TrainingArguments(
            output_dir="./deepseek_r1_70b_tarot_lora",
            num_train_epochs=3,
            per_device_train_batch_size=2,  # H100可承受
            gradient_accumulation_steps=16,  # 合理值
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            # 学习率配置
            learning_rate=2e-5,  # 小数据集用低学习率
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            
            # 日志与保存
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            report_to="tensorboard",
            
            # 优化
            optim="paged_adamw_8bit",  # 内存优化
            max_grad_norm=0.3,
            bf16=True,  # H100支持
            
            # 其他
            dataloader_num_workers=4,
            remove_unused_columns=True,
            push_to_hub=False,
        )
        
        # 6. 创建训练器
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=2048,  # 支持长解读 ✅
            packing=True,  # 高效利用上下文
        )
        
        # 7. 开始训练
        print("🏋️ 开始70B QLoRA训练...")
        print("⏱️ 预计时间: 5-7小时")
        print("💰 预计成本: $12-18 (Lambda H100 PCIe)")
        
        # 训练
        trainer.train()
        
        # 8. 保存模型
        print("💾 保存LoRA权重...")
        trainer.save_model()
        
        # 9. 测试
        print("🧪 测试模型...")
        test_messages = [
            {"role": "user", "content": "请解读塔罗牌：愚人(正位)"}
        ]
        inputs = tokenizer.apply_chat_template(
            test_messages,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("📝 测试输出:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        print("✅ 70B QLoRA训练完成！")
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌟 DeepSeek R1 70B QLoRA 塔罗牌微调系统")
    print("🔗 专为Lambda GPU云平台优化")
    print()
    
    success = train_70b_qlora()
    if success:
        print("🎉 训练成功完成！")
        print("📁 模型保存在: ./deepseek_r1_70b_tarot_lora/")
    else:
        print("❌ 训练失败，请检查错误信息")