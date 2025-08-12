#!/bin/bash
# 上传主要训练文件 - training_data.jsonl
set -e

LAMBDA_HOST="209.20.157.148"
SSH_KEY="~/.ssh/tarot-training-key"

echo "🚀 上传主要训练文件到Lambda"
echo "================================"

echo "�� 1. 上传训练脚本..."
scp -i "$SSH_KEY" train_70b_qlora.py ubuntu@"$LAMBDA_HOST":~/
scp -i "$SSH_KEY" private_train.py ubuntu@"$LAMBDA_HOST":~/

echo "📤 2. 上传隐私保护脚本..."
scp -i "$SSH_KEY" privacy_encryption.py ubuntu@"$LAMBDA_HOST":~/
scp -i "$SSH_KEY" private_env.sh ubuntu@"$LAMBDA_HOST":~/

echo "📤 3. 上传主要加密训练数据..."
scp -i "$SSH_KEY" ../../data/finetune/training_data.jsonl.encrypted ubuntu@"$LAMBDA_HOST":~/

echo "📤 4. 上传加密清单..."
scp -i "$SSH_KEY" ../../data/finetune/encryption_manifest.json ubuntu@"$LAMBDA_HOST":~/

echo "✅ 主要训练文件上传完成！"
echo ""
echo "📊 上传内容："
echo "   🔐 training_data.jsonl.encrypted (55条精选数据)"
echo "   📝 所有必需的训练脚本"
echo "   🛡️ 隐私保护工具"
echo ""
echo "🎯 下一步："
echo "1. ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.157.148"
echo "2. python3 privacy_encryption.py --decrypt"
echo "3. source private_env.sh"
echo "4. python3 private_train.py"
