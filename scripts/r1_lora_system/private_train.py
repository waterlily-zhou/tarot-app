#!/usr/bin/env python3
"""
隐私保护训练脚本 - Lambda服务器版
🔒 在Lambda上运行的完全离线训练
"""
import os
import sys

def setup_private_environment():
    """设置隐私保护环境"""
    print("�� 激活隐私保护模式...")
    
    # 离线模式
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # 禁用外部服务
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 本地日志
    os.environ["TENSORBOARD_LOG_DIR"] = "./private_logs"
    
    print("✅ 隐私保护环境已激活")

if __name__ == "__main__":
    # 激活隐私保护
    setup_private_environment()
    
    # 导入并运行原始训练脚本
    print("🔒 开始隐私保护训练...")
    
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
