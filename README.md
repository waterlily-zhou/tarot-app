# 🎴 塔罗牌AI识别与解读系统

分离架构的塔罗牌系统，将图像识别和AI解读功能模块化设计。

## 🏗️ 重组架构

### 📁 分离式设计
```
tarot-app/
├── main.py                  # 🚀 主入口程序
├── vision/                  # 🔍 图片识别模块
│   ├── simple_card_test.py      # 简化识别功能 (Gemini + 边缘检测)
│   ├── image_preprocessor.py    # 图片预处理
│   ├── waite_tarot_recognizer.py    # 韦特塔罗识别器
│   └── integrated_vision_system.py  # 集成视觉系统
├── ai/                      # 🤖 AI解牌模块  
│   ├── professional_tarot_ai.py     # 专业解牌AI
│   ├── tarot_card_meanings.py      # 卡牌含义数据库
│   ├── tarot_spread_system.py      # 牌阵系统
│   ├── rag_system.py               # RAG检索系统
│   └── integrated_tarot_system.py  # 整合解牌系统
└── data/                    # 📁 数据文件夹
```

## 🎯 功能特色

### 🔍 图像识别 (vision/)
- **Gemini Vision**: Google AI高精度在线识别
- **边缘检测**: 专门的右侧遗漏分析
- **图片预处理**: 边距处理、坐标调整
- **简化菜单**: 只保留核心功能

### 🤖 AI解牌 (ai/)
- **专业解读**: 多维度分析系统
- **RAG检索**: 基于知识库的上下文解读
- **牌阵系统**: 支持多种经典布局
- **完整数据库**: 78张韦特塔罗牌含义

## 🚀 快速开始

### 启动系统
```bash
python main.py
```

### 识别选项
1. **🎯 单图识别** - 快速识别整张牌阵
2. **🔍 边缘遗漏分析** - 检测右侧遗漏卡牌

### 环境配置
```bash
# 设置Gemini API Key
echo "GOOGLE_API_KEY=你的API密钥" > .env.local
```

## 🎯 使用方法

### 图片识别流程
1. 运行 `python main.py`
2. 选择图片路径 
3. 选择识别策略（单图/边缘检测）
4. 查看识别结果和裁剪图片

### AI解牌流程  
1. 将识别结果导入ai模块
2. 使用专业解牌系统分析
3. 生成个性化解读报告

## 📊 技术架构

### 分离设计优势
- **模块独立**: 识别和解牌完全分离
- **灵活组合**: 可独立使用任一模块
- **便于维护**: 功能清晰，代码组织良好
- **扩展性强**: 易于添加新功能

### 核心技术栈
- **识别**: Gemini Vision API + OpenCV
- **解牌**: Ollama + ChromaDB + RAG
- **数据**: 完整韦特塔罗知识库

## 📝 许可证

MIT License