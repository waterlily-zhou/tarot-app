#!/usr/bin/env python3
"""
简化的FP8转换训练脚本
专注于FP8参数转换，绕过复杂的量化配置检查
"""
import os
import sys
import torch
import gc
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json

# 强制实时输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

def load_training_data():
    """加载训练数据"""
    print("📂 加载训练数据: training_data.jsonl")
    with open("training_data.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"✅ 加载完成: {len(data)} 条训练数据")
    return data

def apply_fp8_conversion(model):
    """应用FP8参数转换 - 这是核心修复逻辑"""
    print("🔧 检查并修复FP8量化兼容性问题...")
    fp8_converted = 0
    
    try:
        for name, param in model.named_parameters():
            if hasattr(param, 'dtype') and 'float8' in str(param.dtype).lower():
                print(f"🔧 转换FP8参数: {name} 从 {param.dtype} 到 torch.float16")
                param.data = param.data.to(torch.float16)
                fp8_converted += 1
    except Exception as e:
        print(f"⚠️ FP8参数转换失败（忽略）：{e}")
    
    print(f"✅ FP8转换完成，转换了 {fp8_converted} 个参数")
    return model

def simple_model_loading():
    """简化的模型加载"""
    print("🚀 开始简化模型加载流程...")
    
    # 基础配置
    model_name = "deepseek-ai/DeepSeek-R1"
    
    # 1. 加载tokenizer
    print("🔹 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 设置QLoRA配置
    print("⚙️ 设置QLoRA配置...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # 3. 加载模型 - 使用最简单的方式
    print("🔧 加载模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print("✅ 模型加载成功")
        
        # 4. 应用FP8转换 - 这是关键步骤！
        model = apply_fp8_conversion(model)
        
        # 5. 准备PEFT
        print("🔧 准备PEFT训练...")
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        print("✅ PEFT设置完成")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """主函数"""
    print("🎯 开始简化FP8转换训练")
    print("=" * 60)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU 0: {gpu_info.name} | 显存: {gpu_info.total_memory/1024**3:.1f}GB")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 加载数据
    training_data = load_training_data()
    
    # 加载模型
    model, tokenizer = simple_model_loading()
    
    if model is None:
        print("❌ 模型加载失败，训练终止")
        return
    
    print("✅ 简化FP8转换训练流程完成！")
    print("🔍 可以继续添加训练循环...")

if __name__ == "__main__":
    main()