#!/usr/bin/env python3
"""
Cloudflare Tunnel 端到端加密配置
🌐 建立安全的加密通道连接Lambda服务器
"""
import os
import json
import subprocess
import time
from pathlib import Path

class CloudflareTunnel:
    """Cloudflare Tunnel 管理器"""
    
    def __init__(self):
        self.tunnel_name = "tarot-training-tunnel"
        self.config_dir = Path.home() / ".cloudflared"
        self.config_file = self.config_dir / "config.yml"
    
    def check_cloudflared_installed(self):
        """检查cloudflared是否已安装"""
        try:
            result = subprocess.run(["cloudflared", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ cloudflared已安装: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            print("❌ cloudflared未安装")
            return False
    
    def install_cloudflared(self):
        """安装cloudflared"""
        print("📥 安装cloudflared...")
        
        # macOS安装脚本
        install_script = """#!/bin/bash
echo "📥 安装cloudflared..."

# 检测操作系统
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "使用Homebrew安装..."
        brew install cloudflared
    else
        echo "下载macOS版本..."
        curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz -o cloudflared.tgz
        tar -xzf cloudflared.tgz
        sudo mv cloudflared /usr/local/bin/
        chmod +x /usr/local/bin/cloudflared
        rm cloudflared.tgz
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "下载Linux版本..."
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
    sudo mv cloudflared /usr/local/bin/
    sudo chmod +x /usr/local/bin/cloudflared
fi

echo "✅ cloudflared安装完成"
cloudflared --version
"""
        
        with open("install_cloudflared.sh", "w") as f:
            f.write(install_script)
        
        os.chmod("install_cloudflared.sh", 0o755)
        print("✅ 安装脚本创建: install_cloudflared.sh")
        print("💡 运行: ./install_cloudflared.sh")
    
    def authenticate(self):
        """Cloudflare认证"""
        print("🔐 Cloudflare认证...")
        print("💡 这将打开浏览器进行认证，请登录你的Cloudflare账户")
        
        try:
            subprocess.run(["cloudflared", "tunnel", "login"], check=True)
            print("✅ Cloudflare认证成功")
            return True
        except subprocess.CalledProcessError:
            print("❌ Cloudflare认证失败")
            return False
    
    def create_tunnel(self):
        """创建tunnel"""
        print(f"🚇 创建tunnel: {self.tunnel_name}")
        
        try:
            # 创建tunnel
            result = subprocess.run([
                "cloudflared", "tunnel", "create", self.tunnel_name
            ], capture_output=True, text=True, check=True)
            
            print("✅ Tunnel创建成功")
            
            # 提取tunnel ID
            lines = result.stderr.split('\n')
            for line in lines:
                if 'tunnel' in line and 'with id' in line:
                    tunnel_id = line.split('with id')[1].strip()
                    print(f"🆔 Tunnel ID: {tunnel_id}")
                    return tunnel_id
                    
        except subprocess.CalledProcessError as e:
            if "already exists" in e.stderr:
                print("ℹ️ Tunnel已存在，获取现有tunnel信息...")
                return self.get_tunnel_id()
            else:
                print(f"❌ Tunnel创建失败: {e}")
                return None
    
    def get_tunnel_id(self):
        """获取现有tunnel的ID"""
        try:
            result = subprocess.run([
                "cloudflared", "tunnel", "list"
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.split('\n')
            for line in lines:
                if self.tunnel_name in line:
                    # 提取ID（通常是第一列）
                    tunnel_id = line.split()[0]
                    print(f"🆔 找到现有Tunnel ID: {tunnel_id}")
                    return tunnel_id
                    
        except subprocess.CalledProcessError:
            print("❌ 无法获取tunnel列表")
        
        return None
    
    def create_config(self, tunnel_id: str):
        """创建tunnel配置文件"""
        print("📝 创建tunnel配置...")
        
        # 确保配置目录存在
        self.config_dir.mkdir(exist_ok=True)
        
        config = {
            "tunnel": tunnel_id,
            "credentials-file": str(self.config_dir / f"{tunnel_id}.json"),
            "ingress": [
                {
                    "hostname": "tarot-training.your-domain.com",  # 用户需要替换为自己的域名
                    "service": "ssh://209.20.158.37:22"
                },
                {
                    "hostname": "tarot-logs.your-domain.com",     # 用户需要替换为自己的域名
                    "service": "http://209.20.158.37:6006"  # TensorBoard端口
                },
                {
                    "service": "http_status:404"  # 默认规则
                }
            ]
        }
        
        # 写入YAML配置
        import yaml
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"✅ 配置文件创建: {self.config_file}")
        except ImportError:
            # 如果没有yaml模块，手动写入
            config_yaml = f"""tunnel: {tunnel_id}
credentials-file: {self.config_dir}/{tunnel_id}.json

ingress:
  - hostname: tarot-training.your-domain.com
    service: ssh://209.20.158.37:22
  - hostname: tarot-logs.your-domain.com
    service: http://209.20.158.37:6006
  - service: http_status:404
"""
            with open(self.config_file, "w") as f:
                f.write(config_yaml)
            print(f"✅ 配置文件创建: {self.config_file}")
        
        return config
    
    def setup_dns(self):
        """设置DNS记录"""
        print("🌐 设置DNS记录...")
        print("💡 请在Cloudflare仪表板中添加以下CNAME记录:")
        print()
        print("记录类型: CNAME")
        print("名称: tarot-training")
        print("目标: [你的tunnel ID].cfargotunnel.com")
        print()
        print("记录类型: CNAME") 
        print("名称: tarot-logs")
        print("目标: [你的tunnel ID].cfargotunnel.com")
        print()
        print("⚠️ 请将配置文件中的域名替换为你的实际域名")
    
    def start_tunnel(self):
        """启动tunnel"""
        print("🚇 启动Cloudflare Tunnel...")
        
        try:
            # 使用配置文件启动
            subprocess.run([
                "cloudflared", "tunnel", "run", self.tunnel_name
            ], check=True)
        except subprocess.CalledProcessError:
            print("❌ Tunnel启动失败")
        except KeyboardInterrupt:
            print("\n⏹️ Tunnel已停止")
    
    def create_startup_script(self):
        """创建自动启动脚本"""
        startup_script = f"""#!/bin/bash
# Cloudflare Tunnel 自动启动脚本
echo "🚇 启动Cloudflare Tunnel..."

# 后台运行tunnel
cloudflared tunnel run {self.tunnel_name} &
TUNNEL_PID=$!

echo "✅ Tunnel已启动 (PID: $TUNNEL_PID)"
echo "🌐 可通过以下地址访问:"
echo "   SSH: tarot-training.your-domain.com"
echo "   日志: https://tarot-logs.your-domain.com"
echo ""
echo "⏹️ 停止tunnel: kill $TUNNEL_PID"

# 创建PID文件
echo $TUNNEL_PID > tunnel.pid
"""
        
        with open("start_tunnel.sh", "w") as f:
            f.write(startup_script)
        
        os.chmod("start_tunnel.sh", 0o755)
        
        # 停止脚本
        stop_script = """#!/bin/bash
# 停止Cloudflare Tunnel
if [ -f tunnel.pid ]; then
    PID=$(cat tunnel.pid)
    echo "⏹️ 停止Tunnel (PID: $PID)..."
    kill $PID
    rm tunnel.pid
    echo "✅ Tunnel已停止"
else
    echo "❌ 未找到运行中的tunnel"
fi
"""
        
        with open("stop_tunnel.sh", "w") as f:
            f.write(stop_script)
        
        os.chmod("stop_tunnel.sh", 0o755)
        
        print("✅ 启动脚本创建: start_tunnel.sh")
        print("✅ 停止脚本创建: stop_tunnel.sh")
    
    def create_secure_ssh_config(self):
        """创建安全SSH配置"""
        ssh_config = """# Cloudflare Tunnel SSH配置
# 通过加密隧道连接Lambda服务器

Host lambda-secure
    HostName tarot-training.your-domain.com
    Port 22
    User ubuntu
    IdentityFile ~/.ssh/tarot-training-key
    ServerAliveInterval 60
    ServerAliveCountMax 3
    StrictHostKeyChecking yes
    
    # 通过Cloudflare Tunnel连接
    ProxyCommand cloudflared access ssh --hostname %h
"""
        
        ssh_dir = Path.home() / ".ssh"
        config_file = ssh_dir / "config_cloudflare"
        
        with open(config_file, "w") as f:
            f.write(ssh_config)
        
        print(f"✅ SSH配置创建: {config_file}")
        print("💡 使用方法: ssh -F ~/.ssh/config_cloudflare lambda-secure")

def main():
    """主函数"""
    print("🌐 Cloudflare Tunnel 端到端加密配置")
    print("=" * 50)
    
    tunnel = CloudflareTunnel()
    
    # 检查安装
    if not tunnel.check_cloudflared_installed():
        tunnel.install_cloudflared()
        print("💡 请先安装cloudflared，然后重新运行此脚本")
        return
    
    # 认证
    if not tunnel.authenticate():
        print("❌ 请先完成Cloudflare认证")
        return
    
    # 创建tunnel
    tunnel_id = tunnel.create_tunnel()
    if not tunnel_id:
        print("❌ Tunnel创建失败")
        return
    
    # 创建配置
    tunnel.create_config(tunnel_id)
    
    # 设置DNS提示
    tunnel.setup_dns()
    
    # 创建启动脚本
    tunnel.create_startup_script()
    
    # 创建SSH配置
    tunnel.create_secure_ssh_config()
    
    print("\n🎯 Cloudflare Tunnel配置完成！")
    print("📝 使用步骤:")
    print("1. 在Cloudflare仪表板设置DNS记录")
    print("2. 更新配置文件中的域名")
    print("3. 运行: ./start_tunnel.sh")
    print("4. 使用: ssh -F ~/.ssh/config_cloudflare lambda-secure")

if __name__ == "__main__":
    main()