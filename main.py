#!/usr/bin/env python3
"""
塔罗AI系统 - 主入口
分离架构：图片识别 + AI解牌
"""

import sys
from pathlib import Path

# 添加vision模块到路径
sys.path.append(str(Path(__file__).parent / "vision"))

try:
    from simple_card_test import interactive_menu
    print("✅ 图片识别模块加载成功")
except ImportError as e:
    print(f"❌ 图片识别模块加载失败: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("🎴 塔罗AI系统 - 重组版")
    print("📁 图片识别: vision/")
    print("🤖 AI解牌: ai/")
    print("=" * 35)
    interactive_menu() 