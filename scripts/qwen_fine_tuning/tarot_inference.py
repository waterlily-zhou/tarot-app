#!/usr/bin/env python3
"""
使用微调后的模型进行塔罗解读
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class TarotAI:
    def __init__(self, model_path="./qwen_tarot_lora"):
        print("🔮 加载塔罗AI模型...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载LoRA模型
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-7B-Chat",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        print("✅ 模型加载完成")
    
    def generate_reading(self, person: str, question: str, cards: list, spread: str = "自定义") -> str:
        """生成塔罗解读"""
        cards_str = "；".join(cards)
        
        prompt = f"""请为以下塔罗牌解读提供专业分析：

咨询者：{person}
问题：{question}
牌阵：{spread}
抽到的牌：{cards_str}

请提供深入的解读，包括：
1. 每张牌的含义和在此情境下的解释
2. 牌与牌之间的关系和能量流动
3. 整体的指导建议和洞察
4. 咨询者当前的能量状态和发展方向"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

# 使用示例
if __name__ == "__main__":
    ai = TarotAI()
    
    reading = ai.generate_reading(
        person="测试用户",
        question="职业发展方向",
        cards=["愚人", "魔法师(正位)", "女祭司(逆位)", "皇后"],
        spread="四元素牌阵"
    )
    
    print(reading) 