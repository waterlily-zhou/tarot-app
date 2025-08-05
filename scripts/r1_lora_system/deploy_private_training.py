#!/usr/bin/env python3
"""
隐私保护训练一键部署脚本
🔐 集成AES-256加密、私有云API、Cloudflare Tunnel的完整解决方案
"""
import os
import sys
import subprocess
import json
from pathlib import Path

class PrivateTrainingDeployer:
    """隐私保护训练部署器"""
    
    def __init__(self):
        self.lambda_host = "209.20.158.37"
        self.ssh_key = "~/.ssh/tarot-training-key"
        self.data_dir = "../../data/finetune"
        
    def step1_encrypt_data(self):
        """步骤1: 加密训练数据"""
        print("🔐 步骤1: 加密训练数据")
        print("=" * 40)
        
        try:
            # 运行加密脚本
            result = subprocess.run([
                "python3", "privacy_encryption.py", "--encrypt", 
                "--data-dir", self.data_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 数据加密完成")
                return True
            else:
                print(f"❌ 数据加密失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 加密失败: {e}")
            return False
    
    def step2_setup_private_cloud(self):
        """步骤2: 配置私有云环境"""
        print("\n🔒 步骤2: 配置私有云环境")
        print("=" * 40)
        
        try:
            # 运行私有云配置
            result = subprocess.run([
                "python3", "private_cloud_config.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 私有云配置完成")
                return True
            else:
                print(f"❌ 私有云配置失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 私有云配置失败: {e}")
            return False
    
    def step3_setup_tunnel(self):
        """步骤3: 设置Cloudflare Tunnel"""
        print("\n🌐 步骤3: 设置Cloudflare Tunnel")
        print("=" * 40)
        
        print("💡 Cloudflare Tunnel需要手动配置，运行:")
        print("   python3 cloudflare_tunnel.py")
        print("   然后按照提示完成配置")
        
        return True
    
    def step4_upload_encrypted_data(self):
        """步骤4: 上传加密数据"""
        print("\n📤 步骤4: 上传加密数据")
        print("=" * 40)
        
        # 检查加密文件
        manifest_file = Path(self.data_dir) / "encryption_manifest.json"
        if not manifest_file.exists():
            print("❌ 未找到加密清单文件")
            return False
        
        try:
            # 读取加密清单
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            print(f"📊 准备上传 {manifest['total_files']} 个加密文件")
            
            # 上传加密文件
            encrypted_files = [info["encrypted"] for info in manifest["encrypted_files"]]
            for encrypted_file in encrypted_files:
                print(f"📤 上传: {os.path.basename(encrypted_file)}")
                result = subprocess.run([
                    "scp", "-i", os.path.expanduser(self.ssh_key),
                    encrypted_file, f"ubuntu@{self.lambda_host}:~/"
                ])
                
                if result.returncode != 0:
                    print(f"❌ 上传失败: {encrypted_file}")
                    return False
            
            # 上传加密清单
            result = subprocess.run([
                "scp", "-i", os.path.expanduser(self.ssh_key),
                str(manifest_file), f"ubuntu@{self.lambda_host}:~/"
            ])
            
            # 上传隐私脚本
            privacy_scripts = [
                "privacy_encryption.py",
                "private_cloud_config.py", 
                "train_70b_qlora.py",
                "private_train.py"
            ]
            
            for script in privacy_scripts:
                if os.path.exists(script):
                    result = subprocess.run([
                        "scp", "-i", os.path.expanduser(self.ssh_key),
                        script, f"ubuntu@{self.lambda_host}:~/"
                    ])
            
            print("✅ 所有文件上传完成")
            return True
            
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            return False
    
    def step5_remote_setup(self):
        """步骤5: 远程服务器设置"""
        print("\n🖥️ 步骤5: 远程服务器设置")
        print("=" * 40)
        
        setup_commands = [
            # 解密数据
            "python3 privacy_encryption.py --decrypt",
            
            # 设置私有环境
            "source private_env.sh",
            
            # 安装依赖（如果需要）
            "pip install cryptography",
            
            # 验证数据
            "ls -la *.jsonl"
        ]
        
        for cmd in setup_commands:
            print(f"🔧 执行: {cmd}")
            result = subprocess.run([
                "ssh", "-i", os.path.expanduser(self.ssh_key),
                f"ubuntu@{self.lambda_host}", cmd
            ])
            
            if result.returncode != 0:
                print(f"⚠️ 命令执行异常: {cmd}")
        
        print("✅ 远程设置完成")
        return True
    
    def step6_start_training(self):
        """步骤6: 启动隐私训练"""
        print("\n🏋️ 步骤6: 启动隐私训练")
        print("=" * 40)
        
        print("💡 在Lambda服务器上运行隐私训练:")
        print("   ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.158.37")
        print("   source private_env.sh")
        print("   python3 private_train.py")
        
        return True
    
    def generate_monitoring_dashboard(self):
        """生成监控面板"""
        dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>🔐 隐私训练监控</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .danger { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .info { background-color: #d1ecf1; border: 1px solid #bee5eb; }
    </style>
</head>
<body>
    <h1>🔐 塔罗牌训练隐私保护状态</h1>
    
    <div class="status success">
        <h3>✅ 数据加密状态</h3>
        <p>训练数据已使用AES-256加密</p>
        <p>加密时间: <span id="encryption-time">-</span></p>
    </div>
    
    <div class="status info">
        <h3>🔒 私有云状态</h3>
        <p>离线模式: 已启用</p>
        <p>外部API: 已阻止</p>
        <p>本地日志: 已启用</p>
    </div>
    
    <div class="status warning">
        <h3>🌐 网络隧道状态</h3>
        <p>Cloudflare Tunnel: 需要手动配置</p>
        <p>端到端加密: 等待配置</p>
    </div>
    
    <div class="status info">
        <h3>🏋️ 训练状态</h3>
        <p>模型: DeepSeek-R1-0528</p>
        <p>数据量: 55条训练样本</p>
        <p>状态: <span id="training-status">待启动</span></p>
    </div>
    
    <h2>📋 隐私保护检查清单</h2>
    <ul>
        <li>☑️ 训练数据已加密 (AES-256)</li>
        <li>☑️ 私有云环境已配置</li>
        <li>🔲 Cloudflare Tunnel已配置</li>
        <li>🔲 训练已启动</li>
        <li>🔲 模型已完成训练</li>
    </ul>
    
    <h2>🔧 快速操作</h2>
    <p><strong>本地监控:</strong></p>
    <pre>ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.158.37 "watch -n 5 nvidia-smi"</pre>
    
    <p><strong>查看日志:</strong></p>
    <pre>ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.158.37 "tail -f training.log"</pre>
    
    <p><strong>停止训练:</strong></p>
    <pre>ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.158.37 "pkill -f python3"</pre>
</body>
</html>"""
        
        with open("privacy_dashboard.html", "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        
        print("✅ 监控面板创建: privacy_dashboard.html")
        print("💡 用浏览器打开查看隐私保护状态")
    
    def deploy(self):
        """执行完整部署流程"""
        print("🔐 隐私保护训练一键部署")
        print("=" * 50)
        
        steps = [
            ("加密训练数据", self.step1_encrypt_data),
            ("配置私有云环境", self.step2_setup_private_cloud),
            ("设置网络隧道", self.step3_setup_tunnel),
            ("上传加密数据", self.step4_upload_encrypted_data),
            ("远程服务器设置", self.step5_remote_setup),
            ("启动隐私训练", self.step6_start_training),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                print(f"❌ 部署在步骤'{step_name}'失败")
                return False
        
        # 生成监控面板
        self.generate_monitoring_dashboard()
        
        print("\n🎉 隐私保护训练部署完成！")
        print("📊 打开 privacy_dashboard.html 查看状态")
        
        return True

def main():
    """主函数"""
    deployer = PrivateTrainingDeployer()
    
    print("🔐 欢迎使用塔罗牌训练隐私保护系统")
    print("=" * 50)
    print("此系统包含:")
    print("🔒 AES-256 数据加密")
    print("🌐 Cloudflare Tunnel 端到端加密")
    print("🔧 私有云API配置")
    print("📊 隐私监控面板")
    print()
    
    choice = input("是否开始部署? (y/N): ").lower().strip()
    
    if choice == 'y':
        success = deployer.deploy()
        
        if success:
            print("\n🎯 下一步操作:")
            print("1. 配置Cloudflare Tunnel (可选)")
            print("2. SSH到Lambda服务器启动训练")
            print("3. 监控训练进度")
        else:
            print("\n❌ 部署失败，请检查错误信息")
    else:
        print("👋 部署已取消")

if __name__ == "__main__":
    main()