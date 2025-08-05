#!/bin/bash
# 安全上传加密训练数据到Lambda
# 🔐 隐私保护上传脚本

set -e

LAMBDA_HOST="209.20.158.37"
SSH_KEY="~/.ssh/tarot-training-key"
ENCRYPTED_DIR="../../data/finetune"

echo "🔐 安全上传加密训练数据"
echo "================================"

# 检查加密文件
if [ ! -f "$ENCRYPTED_DIR/encryption_manifest.json" ]; then
    echo "❌ 未找到加密清单文件，请先运行加密"
    exit 1
fi

# 上传加密文件
echo "📤 上传加密文件..."
scp -i "$SSH_KEY" "$ENCRYPTED_DIR"/*.encrypted ubuntu@"$LAMBDA_HOST":~/
scp -i "$SSH_KEY" "$ENCRYPTED_DIR"/encryption_manifest.json ubuntu@"$LAMBDA_HOST":~/

# 上传解密脚本
echo "📤 上传解密脚本..."
scp -i "$SSH_KEY" privacy_encryption.py ubuntu@"$LAMBDA_HOST":~/

echo "✅ 加密文件上传完成"
echo "💡 在Lambda服务器上运行: python3 privacy_encryption.py --decrypt"
