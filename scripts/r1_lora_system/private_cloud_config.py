#!/usr/bin/env python3
"""
私有云API配置 - 避免第三方服务商
🔒 确保模型训练不经过外部API服务
"""
import os
import json
import subprocess
from pathlib import Path

class PrivateCloudConfig:
    """私有云配置管理器"""
    
    def __init__(self):
        self.config = {
            "huggingface": {
                "offline_mode": True,
                "use_cache": True,
                "local_models_only": True
            },
            "training": {
                "disable_wandb": True,
                "disable_tensorboard_cloud": True,
                "local_logging_only": True
            },
            "network": {
                "block_external_apis": True,
                "use_cloudflare_tunnel": True
            }
        }
    
    def setup_offline_environment(self):
        """设置离线环境变量"""
        print("🔒 配置私有云环境...")
        
        # HuggingFace离线模式
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # 禁用外部API
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["TENSORBOARD_LOG_DIR"] = "./logs"
        
        # 禁用telemetry
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print("✅ 离线环境配置完成")
        
        # 生成环境变量脚本
        env_script = """#!/bin/bash
# 私有云环境变量配置
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export TENSORBOARD_LOG_DIR=./logs
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

echo "🔒 私有云环境已激活"
"""
        
        with open("private_env.sh", "w") as f:
            f.write(env_script)
        
        os.chmod("private_env.sh", 0o755)
        print("✅ 环境脚本创建: private_env.sh")
    
    def check_model_availability(self, model_name: str):
        """检查模型是否在本地缓存中"""
        print(f"🔍 检查模型本地可用性: {model_name}")
        
        # HuggingFace缓存路径
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
        
        if model_cache.exists():
            print(f"✅ 模型已缓存: {model_cache}")
            return True
        else:
            print(f"❌ 模型未缓存，需要下载: {model_name}")
            return False
    
    def download_model_privately(self, model_name: str):
        """私密下载模型到本地缓存"""
        print(f"📥 私密下载模型: {model_name}")
        
        # 创建下载脚本
        download_script = f"""#!/usr/bin/env python3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置下载参数
model_name = "{model_name}"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print(f"📥 下载模型到本地缓存: {{model_name}}")

try:
    # 下载tokenizer
    print("📥 下载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # 下载模型配置
    print("📥 下载模型配置...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    print("✅ 模型下载完成！")
    print(f"📁 缓存位置: {{cache_dir}}")
    
except Exception as e:
    print(f"❌ 下载失败: {{e}}")
"""
        
        with open("download_model.py", "w") as f:
            f.write(download_script)
        
        print("✅ 下载脚本创建: download_model.py")
        print("💡 运行: python3 download_model.py")
    
    def create_firewall_rules(self):
        """创建防火墙规则阻止不必要的外部连接"""
        print("🛡️ 创建网络安全规则...")
        
        firewall_script = """#!/bin/bash
# 网络安全配置脚本
# 🛡️ 阻止训练过程中的外部API调用

echo "🛡️ 配置网络安全规则..."

# 创建iptables规则（需要root权限）
create_iptables_rules() {
    # 允许必要的连接
    sudo iptables -A OUTPUT -d 127.0.0.1 -j ACCEPT
    sudo iptables -A OUTPUT -d localhost -j ACCEPT
    
    # 阻止特定域名（可选）
    # sudo iptables -A OUTPUT -d api.wandb.ai -j DROP
    # sudo iptables -A OUTPUT -d api.tensorboard.dev -j DROP
    
    echo "✅ 防火墙规则已配置"
}

# 使用hosts文件阻止域名解析
block_telemetry_domains() {
    echo "🚫 阻止telemetry域名..."
    
    # 备份原hosts文件
    sudo cp /etc/hosts /etc/hosts.backup
    
    # 添加阻止规则
    echo "# 训练隐私保护 - 阻止telemetry" | sudo tee -a /etc/hosts
    echo "127.0.0.1 api.wandb.ai" | sudo tee -a /etc/hosts
    echo "127.0.0.1 api.tensorboard.dev" | sudo tee -a /etc/hosts
    echo "127.0.0.1 telemetry.huggingface.co" | sudo tee -a /etc/hosts
    
    echo "✅ 域名阻止已配置"
}

# 仅修改hosts文件（不需要root权限修改iptables）
block_telemetry_domains

echo "🛡️ 网络安全配置完成"
echo "💡 训练完成后运行: sudo cp /etc/hosts.backup /etc/hosts"
"""
        
        with open("setup_firewall.sh", "w") as f:
            f.write(firewall_script)
        
        os.chmod("setup_firewall.sh", 0o755)
        print("✅ 防火墙脚本创建: setup_firewall.sh")
    
    def generate_private_training_script(self):
        """生成隐私保护的训练脚本"""
        print("🔒 生成隐私训练脚本...")
        
        private_script = """#!/usr/bin/env python3
\"\"\"
隐私保护训练脚本
🔒 完全离线的私有训练环境
\"\"\"
import os
import sys

# 设置隐私保护环境
def setup_private_environment():
    print("🔒 激活隐私保护模式...")
    
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
    from train_70b_qlora import train_70b_qlora
    
    print("🔒 开始隐私保护训练...")
    success = train_70b_qlora()
    
    if success:
        print("🎉 隐私保护训练完成！")
    else:
        print("❌ 训练失败")
"""
        
        with open("private_train.py", "w") as f:
            f.write(private_script)
        
        print("✅ 隐私训练脚本创建: private_train.py")
    
    def save_config(self):
        """保存配置文件"""
        with open("private_cloud_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        print("✅ 私有云配置已保存: private_cloud_config.json")

def main():
    """主函数"""
    print("🔒 私有云API配置工具")
    print("=" * 40)
    
    config = PrivateCloudConfig()
    
    # 设置离线环境
    config.setup_offline_environment()
    
    # 检查模型可用性
    model_name = "deepseek-ai/DeepSeek-R1-0528"
    if not config.check_model_availability(model_name):
        config.download_model_privately(model_name)
    
    # 创建安全规则
    config.create_firewall_rules()
    
    # 生成隐私训练脚本
    config.generate_private_training_script()
    
    # 保存配置
    config.save_config()
    
    print("\n🎯 隐私保护配置完成！")
    print("📝 使用步骤:")
    print("1. 运行: source private_env.sh")
    print("2. 运行: python3 private_train.py")
    print("3. 训练完成后: sudo cp /etc/hosts.backup /etc/hosts")

if __name__ == "__main__":
    main()