#!/usr/bin/env python3
"""
修复版塔罗AI测试脚本 - 确保LoRA权重被正确应用
"""

import os
import sys
import torch
import json
from pathlib import Path

# 设置环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig
    print("✅ 依赖库导入成功")
except ImportError as e:
    print(f"❌ 缺少依赖库: {e}")
    sys.exit(1)

def load_trained_model():
    """加载训练好的模型"""
    print("🤖 加载你的专属塔罗AI模型...")
    
    # 检查设备
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ 使用 Apple Silicon MPS")
    else:
        device = "cpu"
        print("⚠️ 使用 CPU")
    
    model_path = "./models/qwen-tarot-24gb"
    
    try:
        # 1. 首先修复adapter配置
        print("🔧 修复LoRA配置...")
        adapter_config_path = Path(model_path) / "adapter_config.json"
        
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
        
        # 关键修复：设置为训练模式以激活LoRA
        config["inference_mode"] = False
        
        with open(adapter_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ LoRA配置已修复")
        
        # 2. 加载分词器
        print("📥 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # 3. 加载基础模型
        base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
        print(f"📥 加载基础模型: {base_model_name}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 4. 加载LoRA适配器（现在应该会正确激活）
        print("📥 加载并激活你的个人化适配器...")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 确认LoRA是否激活
        print(f"🔍 LoRA状态检查:")
        print(f"   - 是否为PeftModel: {isinstance(model, PeftModel)}")
        print(f"   - 是否为推理模式: {getattr(model.peft_config['default'], 'inference_mode', 'unknown')}")
        
        # 5. 移动到设备
        if device == "mps":
            print("🔄 移动模型到MPS...")
            model = model.to("mps")
        
        print("✅ 模型加载成功！")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_simple_case(model, tokenizer, device):
    """测试一个简单案例"""
    print("\n🧪 快速测试...")
    
    # 使用训练数据中的格式
    prompt = """塔罗解读：
咨询者：Mel
问题：工作发展
牌阵：三张牌
牌：愚人(正位)；力量(正位)；星币十(正位)

请提供专业解读："""
    
    print("🎯 输入prompt:")
    print(prompt)
    print("\n🤖 AI解读:")
    print("-" * 60)
    
    try:
        # 生成回答
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                top_p=0.9
            )
        
        # 解码回答
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ai_response = full_response[len(prompt):].strip()
        
        print(ai_response)
        print("-" * 60)
        
        # 简单质量检查
        if len(ai_response) < 50:
            print("⚠️ 生成内容过短")
        elif "愚人" in ai_response or "力量" in ai_response or "星币" in ai_response:
            print("✅ 生成内容包含相关牌名")
        else:
            print("⚠️ 生成内容可能不够相关")
            
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔮 修复版塔罗AI测试程序")
    print("="*50)
    
    # 加载模型
    model, tokenizer, device = load_trained_model()
    if model is None:
        return
    
    # 快速测试
    test_simple_case(model, tokenizer, device)
    
    print("\n🎉 测试完成！")
    print("💡 如果结果仍然不理想，可能需要检查训练数据格式")

if __name__ == "__main__":
    main() 