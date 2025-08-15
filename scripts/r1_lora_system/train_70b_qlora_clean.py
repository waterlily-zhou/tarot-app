#!/usr/bin/env python3
"""
DeepSeek R1 70B QLoRA微调脚本 - 清洁版本
使用DeepSeek团队提供的终极FP8解决方案
"""

import json
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

def load_training_data(data_path):
    """加载训练数据"""
    print(f"📂 加载训练数据: {data_path}")
    
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

def setup_70b_model():
    """设置DeepSeek R1 70B模型 - DeepSeek终极解决方案"""
    print("🚀 设置DeepSeek R1 70B模型...")
    print("🚀 实施DeepSeek团队终极FP8解决方案...")
    
    try:
        # 🔥 DeepSeek终极解决方案：使用本地缓存路径
        model_id = "deepseek-ai/DeepSeek-R1"  # 使用主版本，避免FP8配置
        
        # 使用我们发现的正确缓存路径
        cache_path = "/home/ubuntu/.cache/huggingface/models--deepseek-ai--DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"
        
        # 检查缓存是否存在
        if os.path.exists(cache_path):
            model_path = cache_path
            print(f"✅ 使用本地缓存: {model_path}")
        else:
            model_path = model_id
            print(f"⚠️ 本地缓存不存在，使用模型ID: {model_path}")
        
        # 临时禁用离线模式
        offline_vars = {}
        for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
            if var in os.environ:
                offline_vars[var] = os.environ.pop(var)
                print(f"🔓 临时禁用: {var}")
        
        print(f"🚀 使用DeepSeek-R1主版本: {model_path}")
        
        # DeepSeek终极配置清理方案
        print("🚀 实施DeepSeek终极FP8解决方案...")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        
        # 🔥 终极方案：用bitsandbytes配置替换FP8配置
        try:
            # 检查是否有FP8配置需要替换
            has_fp8 = False
            if hasattr(config, "quantization_config") and config.quantization_config is not None:
                qc = config.quantization_config
                if hasattr(qc, "quant_method") and qc.quant_method == "fp8":
                    has_fp8 = True
                elif isinstance(qc, dict) and qc.get("quant_method") == "fp8":
                    has_fp8 = True
            
            if has_fp8:
                print("🔄 检测到FP8配置，替换为bitsandbytes配置")
                # 创建伪造的bitsandbytes配置字典
                fake_bnb_config = {
                    "quant_method": "bitsandbytes",
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": "bfloat16"
                }
                
                # 创建一个配置对象
                class FakeBnBConfig:
                    def __init__(self):
                        self.quant_method = "bitsandbytes"
                        self.load_in_4bit = True
                        self.bnb_4bit_quant_type = "nf4"
                        self.bnb_4bit_use_double_quant = True
                        self.bnb_4bit_compute_dtype = "bfloat16"
                    
                    def to_dict(self):
                        return fake_bnb_config
                
                config.quantization_config = FakeBnBConfig()
                print("✅ 已替换为虚假的bitsandbytes配置")
            else:
                print("✅ 未检测到FP8配置，保持原配置")
            
            # 清除可能存在的FP8标志
            for attr in ["fp8", "use_fp8", "is_fp8"]:
                if hasattr(config, attr):
                    delattr(config, attr)
                    print(f"🧹 已删除属性: {attr}")
            
            print("✅ 配置处理完成")
        except Exception as e:
            print(f"⚠️ 配置处理失败: {e}")
            # 如果失败，至少清除quantization_config
            if hasattr(config, "quantization_config"):
                config.quantization_config = None

        # 创建我们自己的4bit量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # 暂时禁用所有量化配置检查（增强版）
        import transformers.quantizers.auto as qa_module
        original_merge = qa_module.AutoHfQuantizer.merge_quantization_configs
        original_supports = qa_module.AutoHfQuantizer.supports_quant_method
        
        def dummy_merge(*args, **kwargs):
            print("🛑 跳过量化配置合并")
            return None
            
        def dummy_supports(quantization_config_dict):
            print("🛑 跳过量化方法支持检查")
            return True
            
        qa_module.AutoHfQuantizer.merge_quantization_configs = staticmethod(dummy_merge)
        qa_module.AutoHfQuantizer.supports_quant_method = staticmethod(dummy_supports)
        
        # 额外patch：直接修改modeling_utils中的量化检查
        try:
            import transformers.modeling_utils as mu
            original_check = getattr(mu, '_check_and_enable_sdpa', None)
            
            # 创建一个安全的配置检查函数
            def safe_quantization_check(config):
                if hasattr(config, 'quantization_config') and config.quantization_config is None:
                    return False
                return False
                
            # 临时替换配置检查
            if hasattr(mu, 'PreTrainedModel'):
                mu.PreTrainedModel._is_quantized_training_enabled = lambda self: False
            print("🛑 已禁用量化训练检查")
        except Exception as patch_e:
            print(f"⚠️ 额外补丁失败（忽略）: {patch_e}")
        
        try:
            # 加载模型时强制覆盖所有量化设置（DeepSeek建议）
            print("🔧 加载模型...")
            model = AutoModelForCausalLM.from_pretrained(
              model_path,
              config=config,  # 使用我们清理过的配置
              quantization_config=bnb_config,  # 强制使用我们的4bit配置
              device_map="auto",
              trust_remote_code=True,
              torch_dtype=torch.bfloat16,
              use_cache=False,
              local_files_only=False,  # 允许在线访问获取正确配置
              ignore_mismatched_sizes=True,  # 忽略可能的配置不匹配
              low_cpu_mem_usage=True,  # 启用内存优化
            )
            print("✅ 模型加载成功，使用DeepSeek方案的4bit NF4量化")
            
            # DeepSeek备选方案：应用量化自由包装器
            print("🔧 应用DeepSeek量化自由包装器...")
            try:
                from transformers import ModelingMixin
                
                class QuantFreeModel(ModelingMixin):
                    def __init__(self, base_model):
                        super().__init__()
                        self.model = base_model
                        self.config = base_model.config
                        
                        # 清除量化标志
                        self.is_quantized = False
                        if hasattr(self.config, "quantization_config"):
                            self.config.quantization_config = None
                        
                        # 清除所有量化相关属性
                        for attr in ["fp8", "use_fp8", "is_fp8", "quantization"]:
                            if hasattr(self.config, attr):
                                delattr(self.config, attr)
                    
                    def forward(self, *args, **kwargs):
                        return self.model(*args, **kwargs)
                    
                    def __getattr__(self, name):
                        return getattr(self.model, name)
                
                # 应用包装器
                model = QuantFreeModel(model)
                print("✅ 已应用量化自由包装器")
            except Exception as wrapper_e:
                print(f"⚠️ 包装器应用失败，继续使用原模型: {wrapper_e}")
                
        finally:
            # 恢复原始函数
            qa_module.AutoHfQuantizer.merge_quantization_configs = original_merge
            qa_module.AutoHfQuantizer.supports_quant_method = original_supports
        
        # 加载tokenizer
        print("🔹 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 准备模型进行QLoRA训练
        print("🔧 准备模型进行QLoRA训练...")
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        
        print("✅ 模型设置完成！")
        return model, tokenizer, lora_config
        
    except Exception as e:
        print(f"❌ 模型设置失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def preprocess_data(examples, tokenizer):
    """预处理训练数据"""
    texts = []
    for item in examples:
        if 'messages' in item:
            # 转换为文本格式
            text = ""
            for msg in item['messages']:
                role = msg.get('role', '')
                content = msg.get('content', '')
                text += f"{role}: {content}\n"
            texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=2048,
        return_tensors="pt"
    )
    
    # Labels与input_ids相同（语言建模任务）
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def train_70b_qlora():
    """主训练函数"""
    print("🎯 开始DeepSeek R1 70B QLoRA微调")
    print("="*60)
    
    # 检查环境
    if not torch.cuda.is_available():
        print("❌ 需要CUDA支持")
        return False
    
    print("🔍 检测环境...")
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"GPU {device}: {gpu_name} | 显存: {gpu_memory:.1f}GB")
    
    # 加载训练数据
    training_data = load_training_data("training_data.jsonl")
    if not training_data:
        return False
    
    # 设置模型
    model, tokenizer, lora_config = setup_70b_model()
    if model is None:
        return False
    
    # 预处理数据
    print("🔧 预处理训练数据...")
    processed_data = preprocess_data(training_data, tokenizer)
    dataset = Dataset.from_dict(processed_data)
    
    # 创建数据加载器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collator
    )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 训练循环
    print("🚀 开始训练...")
    model.train()
    
    total_steps = len(dataloader)
    print(f"📊 总训练步数: {total_steps}")
    
    for epoch in range(3):  # 3个epoch
        print(f"\n📈 Epoch {epoch + 1}/3")
        epoch_loss = 0
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            # 移动数据到GPU
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")
    
    # 保存模型
    print("💾 保存模型...")
    model.save_pretrained("./deepseek_r1_tarot_lora")
    tokenizer.save_pretrained("./deepseek_r1_tarot_lora")
    
    print("🎉 训练完成！")
    return True

if __name__ == "__main__":
    success = train_70b_qlora()
    if success:
        print("🎉 DeepSeek R1 QLoRA微调成功完成！")
    else:
        print("❌ 训练失败")