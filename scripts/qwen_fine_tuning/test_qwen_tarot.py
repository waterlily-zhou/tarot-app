#!/usr/bin/env python3
"""
测试微调后的塔罗AI模型
"""

import os
import sys
import torch
from pathlib import Path

# 设置环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
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
    
    # 加载基础模型
    base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
    print(f"📥 加载基础模型: {base_model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 加载LoRA适配器
        print("📥 加载你的个人化适配器...")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 🔧 关键修复：确保LoRA权重被应用
        print("🔧 激活LoRA适配器...")
        model = model.merge_and_unload()  # 合并LoRA权重到基础模型
        
        # 移动到设备
        if device == "mps":
            model = model.to("mps")
        
        print("✅ 模型加载成功！")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None

def test_tarot_reading(model, tokenizer, device):
    """测试塔罗解读"""
    print("\n🔮 开始测试你的塔罗AI...")
    
    # 测试案例
    test_cases = [
        {
            "person": "Mel",
            "question": "我的事业发展如何？",
            "cards": "愚人(正位)；力量(正位)；星币十(正位)",
            "spread": "三张牌解读"
        },
        {
            "person": "测试者",
            "question": "感情运势",
            "cards": "恋人(正位)；圣杯二(正位)",
            "spread": "简单牌阵"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 测试案例 {i}:")
        print(f"   咨询者: {case['person']}")
        print(f"   问题: {case['question']}")
        print(f"   牌: {case['cards']}")
        print(f"   牌阵: {case['spread']}")
        
        # 构建prompt
        prompt = f"""塔罗解读：
咨询者：{case['person']}
问题：{case['question']}
牌阵：{case['spread']}
牌：{case['cards']}

请提供专业解读："""
        
        print(f"\n🤖 AI解读:")
        print("-" * 50)
        
        try:
            # 生成回答
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码回答
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取AI生成的部分（去掉prompt）
            ai_response = full_response[len(prompt):].strip()
            
            print(ai_response)
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
        
        print()

def interactive_mode(model, tokenizer, device):
    """交互模式"""
    print("\n🎯 进入交互模式 (输入 'quit' 退出)")
    print("现在你可以向你的塔罗AI提问了！")
    
    while True:
        print("\n" + "="*60)
        
        try:
            person = input("咨询者姓名: ").strip()
            if person.lower() == 'quit':
                break
                
            question = input("问题: ").strip()
            if question.lower() == 'quit':
                break
                
            cards = input("抽到的牌 (用；分隔): ").strip()
            if cards.lower() == 'quit':
                break
                
            spread = input("牌阵类型 (可选): ").strip() or "自由牌阵"
            if spread.lower() == 'quit':
                break
            
            # 构建prompt
            prompt = f"""塔罗解读：
咨询者：{person}
问题：{question}
牌阵：{spread}
牌：{cards}

请提供专业解读："""
            
            print(f"\n🔮 {person}的塔罗解读:")
            print("="*60)
            
            # 生成回答
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码回答
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ai_response = full_response[len(prompt):].strip()
            
            print(ai_response)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 感谢使用你的专属塔罗AI!")
            break
        except Exception as e:
            print(f"❌ 生成失败: {e}")

def main():
    print("🔮 塔罗AI测试程序")
    print("="*50)
    
    # 加载模型
    model, tokenizer, device = load_trained_model()
    if model is None:
        return
    
    # 运行测试
    test_tarot_reading(model, tokenizer, device)
    
    # 询问是否进入交互模式
    choice = input("是否进入交互模式？(y/n): ").strip().lower()
    if choice in ['y', 'yes', '是']:
        interactive_mode(model, tokenizer, device)
    
    print("\n🎉 测试完成！你的塔罗AI已经准备就绪！")

if __name__ == "__main__":
    main() 