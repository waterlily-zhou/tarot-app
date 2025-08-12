#!/bin/bash
# Lambda训练环境设置脚本
set -e

echo "🚀 开始设置Lambda训练环境"
echo "================================"

echo "📋 1. 检查上传的文件..."
ls -la ~/

echo "🔓 2. 解密训练数据..."
python3 privacy_encryption.py --decrypt training_data.jsonl.encrypted

echo "📦 3. 安装依赖包..."
pip install transformers datasets accelerate bitsandbytes torch

echo "🌐 4. 下载DeepSeek R1模型（临时启用网络）..."
# 临时禁用离线模式
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE WANDB_DISABLED

echo "下载tokenizer..."
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-0528')
print('✅ Tokenizer下载完成')
"

echo "下载模型..."
python3 -c "
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained(
    'deepseek-ai/DeepSeek-R1-0528',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)
print('✅ 模型下载完成')
"

echo "🛡️ 5. 启用隐私保护环境..."
source private_env.sh

echo "🎯 6. 启动训练..."
python3 private_train.py

echo "✅ Lambda训练环境设置完成！" 