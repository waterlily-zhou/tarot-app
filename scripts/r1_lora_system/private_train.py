#!/usr/bin/env python3
"""
隐私保护训练脚本 - Lambda服务器版 (优化版)
🔒 优化显存使用的隐私保护训练
"""
import os
import sys
import subprocess

def setup_private_environment():
    """设置隐私保护环境"""
    print("📦 激活优化的隐私保护模式...")
    
    # 🚀 显存优化环境变量
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # 禁用外部服务 (但允许模型下载)
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 本地日志
    os.environ["TENSORBOARD_LOG_DIR"] = "./private_logs"
    
    # 创建必要目录
    os.makedirs("./offload", exist_ok=True)
    os.makedirs("./private_logs", exist_ok=True)
    
    print("✅ 优化的隐私保护环境已激活")

if __name__ == "__main__":
    # 激活隐私保护
    setup_private_environment()
    
    # 🚀 GPU内存预清理
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU缓存已清理")
    except ImportError:
        pass
    
    # 导入并运行原始训练脚本
    print("🔒 开始优化的隐私保护训练...")
    
    try:
        from train_70b_qlora import train_70b_qlora
        success = train_70b_qlora()
        
        if success:
            print("🎉 隐私保护训练完成！")
        else:
            print("❌ 训练失败")
    except ImportError:
        print("❌ 无法导入训练脚本，请确保train_70b_qlora.py存在")
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
