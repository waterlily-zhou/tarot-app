#!/usr/bin/env python3
"""
训练监控脚本
"""

import time
import subprocess
import os
from pathlib import Path

def get_training_process():
    """获取训练进程信息"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'finetune_qwen_tarot.py' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    return {
                        'pid': parts[1],
                        'cpu': parts[2],
                        'memory': parts[3],
                        'time': parts[9],
                        'command': ' '.join(parts[10:])
                    }
    except:
        pass
    return None

def check_model_output():
    """检查模型输出目录"""
    model_dir = Path("./models/qwen-tarot-24gb")
    if model_dir.exists():
        files = list(model_dir.glob("*"))
        return len(files), [f.name for f in files[:5]]
    return 0, []

def monitor_training():
    """监控训练进程"""
    print("🔍 开始监控训练进程...")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            print(f"⏰ 检查时间: {time.strftime('%H:%M:%S')}")
            
            # 检查进程
            process = get_training_process()
            if process:
                print(f"✅ 训练进程运行中:")
                print(f"   PID: {process['pid']}")
                print(f"   CPU: {process['cpu']}%")
                print(f"   内存: {process['memory']}%")
                print(f"   运行时间: {process['time']}")
            else:
                print("❌ 未找到训练进程")
            
            # 检查模型文件
            file_count, files = check_model_output()
            if file_count > 0:
                print(f"📁 模型输出: {file_count} 个文件")
                if files:
                    print(f"   最新文件: {', '.join(files)}")
            else:
                print("📁 模型输出: 无文件")
            
            # 检查系统内存
            try:
                result = subprocess.run(['top', '-l', '1'], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'PhysMem:' in line:
                        print(f"💾 系统内存: {line.strip()}")
                        break
            except:
                print("💾 系统内存: 无法获取")
            
            print("-" * 50)
            time.sleep(30)  # 每30秒检查一次
            
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")

if __name__ == "__main__":
    monitor_training() 