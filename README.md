# 🎴 韦特塔罗AI识别系统

本地运行的塔罗牌识别和AI解读系统，专为韦特塔罗设计。

## ✨ 功能特色

- 🔍 **精准识别**: 识别韦特塔罗78张牌
- 🎯 **正逆位判断**: 自动检测卡牌方向
- 📍 **位置检测**: 精确定位卡牌在牌摊中的位置
- 🤖 **AI解读**: 结合专业知识库生成个性化解读
- 💾 **本地运行**: 完全在M4 MacBook Air上运行，保护隐私

## 🚀 快速开始

### 安装依赖
```bash
pip install opencv-python pillow imagehash scikit-learn sentence-transformers chromadb ollama
```

### 启动系统
```bash
python simple_card_test.py
```

## 📁 核心文件

- `waite_tarot_recognizer.py` - 核心识别引擎
- `simple_card_test.py` - 测试和演示入口
- `rag_system.py` - 知识检索系统
- `tarot_ai_system.py` - AI解读系统
- `integrated_vision_system.py` - 完整集成系统
- `real_data_processor.py` - 数据处理工具

## 🎯 使用方法

1. **简单识别测试**: 测试卡牌检测和识别功能
2. **完整系统演示**: 包含识别和AI解读的完整流程
3. **重新训练**: 从标准卡牌图片重建识别数据库

## 📊 系统架构

```
图片输入 → 卡牌检测 → 特征匹配 → 正逆位判断 → AI解读 → 结果输出
    ↓         ↓         ↓         ↓         ↓        ↓
  预处理   区域提取   参考库匹配  方向分析  RAG检索  个性化输出
```

## 🔧 配置要求

- **硬件**: M4 MacBook Air (24GB RAM)
- **系统**: macOS 24.5.0
- **Python**: 3.9+
- **LLM**: Ollama (qwen2.5:1.5b)

## 📝 许可证

MIT License