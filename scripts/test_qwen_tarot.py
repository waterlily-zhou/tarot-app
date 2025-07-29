#!/usr/bin/env python3
"""
测试微调后的Qwen塔罗模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def load_model():
    """加载训练后的模型"""
    model_path = "./models/qwen-tarot-24gb"
    base_model_name = "Qwen/Qwen1.5-7B-Chat"
    
    print("🔮 加载塔罗AI模型...")
    
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 合并权重以提高推理速度
        model = model.merge_and_unload()
        
        print("✅ 模型加载成功！")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 请确保训练已完成且模型文件存在")
        return None, None

def generate_reading(model, tokenizer, prompt, max_length=500):
    """生成塔罗解读"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 如果使用MPS，移动到设备
    if torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除原始prompt，只返回生成的部分
    response = response[len(prompt):].strip()
    
    return response

def main():
    model, tokenizer = load_model()
    
    if model is None:
        return
    
    # 测试案例
    test_cases = [
        {
            "title": "工作发展测试",
            "prompt": """塔罗解读：
咨询者：Mel
问题：未来工作发展方向
牌阵：三张牌
牌：愚者；星币三；宝剑王后

请提供专业解读："""
        },
        {
            "title": "感情咨询测试", 
            "prompt": """塔罗解读：
咨询者：Sarah
问题：当前感情状况如何
牌阵：单张牌
牌：恋人

请提供专业解读："""
        },
        {
            "title": "个人成长测试",
            "prompt": """塔罗解读：
咨询者：KK
问题：2025年个人发展重点
牌阵：过去现在未来
牌：隐者；星币皇后；太阳

请提供专业解读："""
        }
    ]
    
    print("🎭 开始测试微调后的塔罗AI...")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】{test['title']}")
        print("-" * 40)
        print("📝 输入:")
        print(test['prompt'])
        print("\n🔮 AI解读:")
        
        try:
            response = generate_reading(model, tokenizer, test['prompt'])
            print(response)
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
        
        print("\n" + "="*60)
        
        if i < len(test_cases):
            input("按 Enter 继续下一个测试...")
    
    print("\n🎊 测试完成！")
    print("💡 如果结果不理想，可以:")
    print("   1. 增加训练数据")
    print("   2. 调整训练轮数")
    print("   3. 优化prompt格式")

if __name__ == "__main__":
    main() 