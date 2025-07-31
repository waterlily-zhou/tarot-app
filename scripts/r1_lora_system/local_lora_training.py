#!/usr/bin/env python3
"""
本地R1-Distill LoRA训练脚本 - MacBook Air M4优化版
保护数据隐私，所有训练在本地完成
"""
import os
import sys
import torch
import json
import sqlite3
from pathlib import Path
import gc
from datetime import datetime

# 设置MPS优化
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def check_m4_compatibility():
    """检查M4硬件兼容性"""
    print("🔍 检查MacBook M4硬件兼容性...")
    
    # 检查MPS可用性
    if not torch.backends.mps.is_available():
        print("❌ MPS不可用，无法使用M4 GPU加速")
        return False
    
    print("✅ MPS可用，可以使用M4 GPU加速")
    
    # 检查内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"💾 总内存: {total_gb:.1f}GB")
        print(f"💾 可用内存: {available_gb:.1f}GB")
        
        # 内存建议
        if available_gb < 8:
            print("❌ 可用内存不足8GB，无法训练")
            return False
        elif available_gb < 12:
            print("⚠️ 内存有限，建议使用1.5B模型 + 4bit量化")
            return "small"
        elif available_gb < 16:
            print("⚠️ 内存刚好够用，建议使用7B模型 + 4bit量化")
            return "medium"
        else:
            print("✅ 内存充足，可以训练7B模型")
            return "large"
            
    except ImportError:
        print("💡 建议安装psutil: pip install psutil")
    
    return True

def choose_model_config(memory_status):
    """根据内存情况选择模型配置"""
    if memory_status == "small":
        return {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
            "use_4bit": True,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "max_seq_length": 512,
            "gradient_checkpointing": True
        }
    elif memory_status == "medium":
        return {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "use_4bit": True, 
            "batch_size": 1,
            "gradient_accumulation": 4,
            "max_seq_length": 1024,
            "gradient_checkpointing": True
        }
    else:
        return {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "use_4bit": False,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "max_seq_length": 1024,
            "gradient_checkpointing": False
        }

def setup_4bit_config():
    """设置4bit量化配置"""
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    except ImportError:
        print("❌ 需要安装bitsandbytes: pip install bitsandbytes")
        return None

def install_dependencies():
    """安装LoRA训练依赖"""
    print("📦 安装LoRA训练依赖...")
    
    dependencies = [
        "transformers>=4.36.0",
        "peft>=0.7.0", 
        "trl>=0.7.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",  # MPS兼容版本
        "torch>=2.1.0",
    ]
    
    import subprocess
    for dep in dependencies:
        try:
            print(f"  安装 {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep, "--quiet"
            ])
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败 {dep}: {e}")
            return False
    
    print("✅ 依赖安装完成")
    return True

def prepare_local_dataset():
    """准备本地训练数据 - 不上传任何内容"""
    print("📚 准备本地训练数据...")
    
    # 读取本地数据库
    db_path = "data/deepseek_tarot_knowledge.db"
    if not Path(db_path).exists():
        print(f"❌ 本地数据库不存在: {db_path}")
        return None
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT person, question, cards, spread, content FROM readings")
    readings = cursor.fetchall()
    conn.close()
    
    if len(readings) < 50:
        print(f"⚠️ 数据量较少({len(readings)}条)，建议至少50条以上")
        
    print(f"📊 找到 {len(readings)} 条本地解读数据")
    
    # 构建训练数据
    training_data = []
    for person, question, cards, spread, content in readings:
        # 标准化输入格式
        instruction = f"""作为专业塔罗师，请为以下咨询提供深度解读：

咨询者：{person}
问题：{question}
牌阵：{spread or '自由牌阵'}
抽到的牌：{cards}

请运用你的专业知识和直觉进行解读。"""

        training_data.append({
            "instruction": instruction,
            "output": content,
            "metadata": {
                "person": person,
                "cards": cards,
                "length": len(content)
            }
        })
    
    # 保存到本地
    output_file = "data/local_training_data.json"
    os.makedirs("data", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 训练数据保存到: {output_file}")
    print(f"🔒 数据完全保存在本地，不会上传到任何服务器")
    
    return training_data

def setup_m4_lora_config():
    """M4优化的LoRA配置"""
    try:
        from peft import LoraConfig, TaskType
    except ImportError:
        print("❌ 请先安装PEFT: pip install peft")
        return None
    
    # 针对M4的优化配置
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        
        # 核心参数 - 平衡性能和质量
        r=16,              # 适中的rank，M4可以处理
        lora_alpha=32,     # 2倍的alpha
        lora_dropout=0.1,  # 适中的dropout
        
        # 目标模块 - 关注核心注意力层
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
            "gate_proj", "up_proj", "down_proj"       # FFN层
        ],
        
        # 其他配置
        bias="none",
        modules_to_save=None,  # 节省内存
    )
    
    print("⚙️ LoRA配置优化为M4硬件")
    return config

def train_local_lora():
    """本地LoRA训练主函数"""
    print("🚀 开始本地R1-Distill LoRA训练")
    print("="*50)
    
    # 1. 硬件检查
    memory_status = check_m4_compatibility()
    if memory_status == False:
        return False
    
    # 2. 选择模型配置
    config = choose_model_config(memory_status)
    print(f"\n⚙️ 选择配置：{config['model_name']}")
    print(f"💾 4bit量化：{'是' if config['use_4bit'] else '否'}")
    print(f"📦 批处理大小：{config['batch_size']}")
    
    # 3. 安装依赖
    if not install_dependencies():
        return False
    
    # 4. 准备数据
    training_data = prepare_local_dataset()
    if not training_data:
        return False
    
    try:
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer, DataCollatorForLanguageModeling
        )
        from peft import get_peft_model, TaskType
        from datasets import Dataset
        from trl import SFTTrainer
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    # 5. 加载基础模型
    print(f"📥 加载{config['model_name']}...")
    
    try:
        # 先加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'], 
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 激进内存优化
        aggressive_memory_optimization()
        
        # 设置量化配置
        quantization_config = None
        if config['use_4bit']:
            print("🔄 设置4bit量化以节省内存...")
            quantization_config = setup_4bit_config()
            if quantization_config is None:
                return False
        
        # 监控内存
        if not monitor_memory_usage():
            print("💡 请关闭更多应用释放内存")
            return False
        
        # 加载模型 - M4优化
        model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16,  # M4支持FP16
            device_map="auto" if config['use_4bit'] else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False             # 训练时关闭cache
        )
        
        # 启用gradient checkpointing
        if config['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
            print("✅ 启用梯度检查点以节省内存")
        
        # 如果没有量化，手动移动到MPS
        if not config['use_4bit']:
            print("🔄 将模型移动到M4 GPU...")
            model = model.to("mps")
        
        print(f"✅ 模型加载成功！参数量: {model.num_parameters():,}")
        
        # 清理内存并监控
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        monitor_memory_usage()
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 尝试关闭其他应用释放内存")
        return False
    
    # 6. 应用LoRA
    lora_config = setup_m4_lora_config()
    if not lora_config:
        return False
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 7. 准备数据集
    dataset = Dataset.from_list(training_data)
    
    def formatting_func(examples):
        """格式化训练数据"""
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = f"{instruction}\n\n{output}"
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_func, batched=True)
    
    # 8. 训练配置 - 内存优化
    training_args = TrainingArguments(
        output_dir="./models/local_tarot_lora",
        
        # 学习率
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        
        # 批次配置 - 根据内存动态调整
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation'],
        
        # 训练步数
        num_train_epochs=3,
        max_steps=300,  # 减少步数以节省时间
        
        # 保存策略
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,  # 只保留2个checkpoint
        
        # 优化器
        optim="adamw_torch",
        weight_decay=0.01,
        
        # MPS优化
        fp16=True,
        dataloader_num_workers=0,  # MPS不支持多进程
        dataloader_pin_memory=False,
        
        # 监控
        logging_steps=20,
        report_to="none",
        
        # 其他
        remove_unused_columns=False,
        push_to_hub=False,  # 不上传到云端
    )
    
    # 9. 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=config['max_seq_length'],  # 根据内存调整
        peft_config=lora_config,
        dataset_text_field="text",
        packing=False,
    )
    
    # 10. 开始训练
    print("🏋️ 开始本地训练...")
    print(f"📊 训练数据: {len(training_data)}条")
    print(f"🔒 所有数据保持在本地，绝不上传")
    print("💡 建议训练期间关闭浏览器等大内存应用")
    
    try:
        # 最后一次内存清理
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # 训练
        trainer.train()
        
        # 保存模型
        print("💾 保存本地LoRA模型...")
        trainer.save_model()
        tokenizer.save_pretrained("./models/local_tarot_lora")
        
        print("🎉 本地训练完成！")
        print(f"📁 模型保存在: ./models/local_tarot_lora")
        print("🔒 所有模型文件都在本地，未上传任何数据")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        print("💡 尝试解决方案:")
        print("  1. 关闭浏览器和其他应用")
        print("  2. 重启Terminal释放内存")
        print("  3. 选择更小的模型")
        return False

def test_local_model():
    """测试本地训练的模型"""
    print("🧪 测试本地训练的LoRA模型...")
    
    model_path = "./models/local_tarot_lora"
    if not Path(model_path).exists():
        print("❌ 本地模型不存在，请先训练")
        return
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        # 加载模型
        base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="mps",
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 测试解读
        test_prompt = """作为专业塔罗师，请为以下咨询提供深度解读：

咨询者：测试用户
问题：当前的人生方向
牌阵：三牌指引
抽到的牌：愚人(正位) | 力量(正位) | 星币十(正位)

请运用你的专业知识和直觉进行解读。"""

        inputs = tokenizer(test_prompt, return_tensors="pt").to("mps")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = result[len(test_prompt):].strip()
        
        print("🎯 本地模型解读结果：")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def estimate_training_time():
    """估算训练时间"""
    print("⏱️ 训练时间估算（M4 MacBook Air）：")
    print("  - 数据准备: 10-20分钟")
    print("  - 模型下载: 15-30分钟（首次）")
    print("  - LoRA训练: 2-4小时")
    print("  - 总耗时: 3-5小时")
    print("  - 电费成本: ~$1-2")

def aggressive_memory_optimization():
    """激进内存优化"""
    import os
    # 设置更激进的内存管理
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
    
    # 禁用一些不必要的功能
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("🔧 启用激进内存优化")

def monitor_memory_usage():
    """监控内存使用"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        used_percent = memory.percent
        
        if available_gb < 2:
            print(f"⚠️ 内存告急! 可用: {available_gb:.1f}GB")
            return False
        elif available_gb < 4:
            print(f"⚠️ 内存紧张: {available_gb:.1f}GB ({used_percent:.1f}%使用)")
        else:
            print(f"✅ 内存正常: {available_gb:.1f}GB ({used_percent:.1f}%使用)")
        
        return True
    except ImportError:
        return True

if __name__ == "__main__":
    print("🏠 本地R1-Distill LoRA训练系统")
    print("🔒 完全保护数据隐私，不上传任何内容")
    print()
    
    while True:
        print("\n选择操作：")
        print("1. 检查硬件兼容性")
        print("2. 估算训练时间")
        print("3. 开始本地训练")
        print("4. 测试本地模型")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == "1":
            check_m4_compatibility()
        elif choice == "2":
            estimate_training_time()
        elif choice == "3":
            train_local_lora()
        elif choice == "4":
            test_local_model()
        elif choice == "5":
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请重试") 