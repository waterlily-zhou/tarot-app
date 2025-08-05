#!/bin/bash
# 隐私保护环境变量配置 - Lambda版
echo "🔒 设置隐私保护环境..."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export TENSORBOARD_LOG_DIR=./private_logs
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

# 创建本地日志目录
mkdir -p ./private_logs

echo "✅ 隐私保护环境已激活"
echo "📊 环境变量:"
echo "   HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "   WANDB_DISABLED=$WANDB_DISABLED"
echo "   日志目录: $TENSORBOARD_LOG_DIR"
