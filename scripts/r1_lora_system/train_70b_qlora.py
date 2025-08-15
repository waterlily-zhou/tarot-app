#!/usr/bin/env python3
"""
DeepSeek R1 70B QLoRA 微调脚本 - 全新优化版
设计原则：
1. 彻底解决FP8量化冲突
2. 优化设备映射避免meta设备问题
3. 简化模型加载流程
4. 增强错误处理和日志
"""
import os
import sys
import torch
import json
import gc
import logging
from datetime import datetime
from pathlib import Path

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

# 强制实时输出
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def detect_environment():
    """检测GPU环境并返回设备信息"""
    logger.info("🔍 检测环境...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} | 显存: {total_memory:.1f}GB")
        return True
    logger.error("❌ 需要CUDA GPU环境")
    return False

def load_training_data(data_path="training_data.jsonl"):
    """加载训练数据"""
    logger.info(f"📂 加载训练数据: {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"❌ 数据文件不存在: {data_path}")
        return None
    
    try:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if 'messages' in item:
                    data.append(item)
        logger.info(f"✅ 加载完成: {len(data)} 条训练数据")
        return data
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        return None

def load_model_safely(model_id):
    """安全加载模型，避免FP8量化冲突"""
    from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 创建干净的配置对象
    logger.info("🧹 创建干净配置对象...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # 清除所有量化相关属性
    for attr in ["quantization_config", "fp8", "use_fp8", "is_fp8"]:
        if hasattr(config, attr):
            delattr(config, attr)
            logger.info(f"  已删除属性: {attr}")
    
    # 在配置中设置use_cache而不是在from_pretrained中
    config.use_cache = False
    
    # 🔧 关键修复：先加载模型到CPU，然后处理FP8，最后量化
    logger.info("🔧 Step 1: 先加载模型到CPU（无量化）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map="cpu",  # 强制CPU加载
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用float16避免FP8问题
        low_cpu_mem_usage=True,
    )
    
    # 🔧 Step 2: FP8参数转换
    logger.info("🔧 Step 2: 进行FP8参数转换...")
    fp8_converted = 0
    
    try:
        for name, param in model.named_parameters():
            if hasattr(param, 'dtype') and 'float8' in str(param.dtype).lower():
                logger.info(f"🔧 转换FP8参数: {name} 从 {param.dtype} 到 torch.float16")
                param.data = param.data.to(torch.float16)
                fp8_converted += 1
                
    except Exception as e:
        logger.warning(f"⚠️ FP8参数转换遇到问题（继续）：{e}")
    
    logger.info(f"✅ FP8转换完成！转换了 {fp8_converted} 个参数")
    
    # 🔧 Step 3: 现在应用量化并移动到GPU
    logger.info("🔧 Step 3: 应用量化配置...")
    
    # 设置4-bit量化配置，启用CPU offload
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,  # 启用CPU offload
    )
    
    # 手动量化模型
    from transformers.quantizers import AutoHfQuantizer
    hf_quantizer = AutoHfQuantizer.from_config(bnb_config)
    model = hf_quantizer.quantize_model(model, device_map="auto")
    
    return model, bnb_config

def setup_70b_model():
    """设置DeepSeek R1 70B模型"""
    logger.info("🚀 设置DeepSeek R1 70B模型...")
    try:
        from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # 模型ID
        model_id = "deepseek-ai/DeepSeek-R1"
        
        # 临时禁用离线模式
        for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
            if var in os.environ:
                logger.info(f"🔓 临时禁用: {var}")
                os.environ.pop(var, None)
        
        # 加载tokenizer
        logger.info("🔹 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 安全加载模型
        model, bnb_config = load_model_safely(model_id)
        
        # 准备k-bit训练
        logger.info("🔧 准备模型进行QLoRA训练...")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        # 启用输入梯度
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 应用LoRA
        logger.info("🎯 应用LoRA适配器...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 全局化tokenizer供后续使用
        globals()['tokenizer'] = tokenizer
        
        return model, tokenizer, lora_config

    except ImportError as e:
        logger.error(f"❌ 缺少依赖: {e}")
        logger.info("💡 请安装: pip install transformers peft bitsandbytes accelerate")
        return None, None, None
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def format_chat(example):
    """使用tokenizer内置的聊天模板格式化数据"""
    messages = example["messages"]
    tk = globals().get("tokenizer", None)
    if tk is None:
        raise RuntimeError("tokenizer not initialized in globals()")
    return {"text": tk.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )}

def train_70b_qlora():
    """主训练函数"""
    logger.info("🎯 开始DeepSeek R1 70B QLoRA微调")
    logger.info("=" * 60)
    
    # 1. 环境检测
    if not detect_environment():
        return False
    
    # 2. 加载数据
    training_data = load_training_data()
    if not training_data:
        return False
    
    # 3. 设置模型
    model, tokenizer, lora_config = setup_70b_model()
    if model is None:
        return False
    
    # 确保tokenizer全局可用
    globals()['tokenizer'] = tokenizer
    
    try:
        from datasets import Dataset
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint
        
        # 4. 准备数据集
        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(format_chat)
        
        # 5. 分词
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=4096,
                padding=False,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        
        # 6. 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 7. 训练参数
        training_args = TrainingArguments(
            output_dir="./deepseek_r1_70b_tarot_lora",
            num_train_epochs=3,
            per_device_train_batch_size=1,  # 根据显存调整
            gradient_accumulation_steps=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            save_strategy="epoch",
            logging_steps=10,
            fp16=False,
            bf16=True,  # 使用bfloat16
            tf32=True,   # 在Ampere GPU上启用tf32
            gradient_checkpointing=True,
            report_to="none",
            optim="paged_adamw_8bit",  # 使用分页优化器
            max_grad_norm=0.3,
        )
        
        # 8. 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 9. 训练
        logger.info("🏋️ 开始训练...")
        trainer.train()
        
        # 10. 保存模型
        logger.info("💾 保存模型...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # 11. 测试模型
        logger.info("🧪 测试模型...")
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
        logger.info("📝 测试输出:")
        logger.info("-" * 50)
        logger.info(response)
        logger.info("-" * 50)
        
        logger.info("✅ 训练完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🌟 DeepSeek R1 70B QLoRA 塔罗牌微调系统")
    print("=" * 60)
    
    # 清除CUDA缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    success = train_70b_qlora()
    if success:
        print("🎉 训练成功完成！")
        print("📁 模型保存在: ./deepseek_r1_70b_tarot_lora/")
    else:
        print("❌ 训练失败，请检查日志文件 training.log")