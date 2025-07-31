# 塔罗AI系统

智能塔罗牌解读系统，结合计算机视觉识别和大语言模型解读。

## 🚀 最新：R1 LoRA本地微调系统

**🆕 2024年7月31日更新**：新增基于DeepSeek R1的本地LoRA微调系统！

**主要特性：**
- 🔒 **完全本地训练**：数据隐私100%保护
- 🧠 **R1推理能力**：具备深度思考和分析能力  
- ⚡ **高效LoRA训练**：只需3-5小时即可完成微调
- 💻 **M4优化**：专为Apple Silicon优化

👉 **查看详情**：[R1 LoRA系统文档](models/r1_lora_system/README.md)

## 📁 项目结构

```
tarot-app/
├── README.md                     # 本文档
├── main.py                       # 主程序入口
├── models/                       # 模型相关文件
│   └── r1_lora_system/          # 🆕 R1 LoRA微调系统
│       ├── README.md            # R1系统详细文档
│       ├── local_lora_training.py # 本地LoRA训练脚本
│       ├── test_pure_r1.py      # R1测试和深度学习
│       ├── test_vector_rag.py   # 向量检索对比
│       └── lora_comparison.md   # LoRA技术对比
├── ai/                          # AI解读模块
│   ├── professional_tarot_ai.py # 专业塔罗AI
│   ├── tarot_spread_system.py   # 牌阵系统
│   └── ...
├── vision/                      # 计算机视觉模块
│   ├── split_gemini_recognizer.py # Gemini视觉识别
│   ├── image_preprocessor.py    # 图像预处理
│   └── ...
└── data/                        # 数据文件
    ├── deepseek_tarot_knowledge.db # 知识库
    └── card_meanings/           # 牌意笔记
```

## 🎯 系统选择指南

| 需求场景 | 推荐系统 | 特点 |
|----------|----------|------|
| **隐私至上 + 最佳质量** | [R1 LoRA本地微调](models/r1_lora_system/) | 完全本地，R1推理能力 |
| **快速原型验证** | R1 API (test_pure_r1.py) | 无需训练，即时可用 |
| **卡牌识别** | 视觉系统 (vision/) | Gemini视觉，准确识别 |

## 🔧 快速开始

### 1. R1 LoRA本地训练（推荐）
```bash
cd models/r1_lora_system/
python local_lora_training.py
```

### 2. R1 API测试
```bash
python models/r1_lora_system/test_pure_r1.py
```

### 3. 视觉识别测试
```bash
python vision/simple_card_test.py
```

## 💾 硬件要求

### R1 LoRA训练
- **内存**：8GB+ 可用内存
- **GPU**：Apple M4/M3 (MPS支持)
- **存储**：30GB 空闲空间
- **时间**：3-5小时训练

### 基础使用
- **内存**：4GB+ 
- **网络**：API调用需要
- **存储**：5GB

## 📊 性能对比

| 方案 | 质量 | 隐私 | 成本 | 速度 |
|------|------|------|------|------|
| **R1 LoRA本地** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| R1 API | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Qwen微调 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🗂️ 历史版本

### 传统AI模块 (ai/)
- 基础塔罗解读逻辑
- 牌阵系统实现
- RAG知识检索

### 视觉识别 (vision/)  
- Gemini Vision API
- 图像预处理优化
- 批量卡牌识别

### 数据管理 (data/)
- SQLite知识库
- 卡牌meaning笔记
- 历史解读数据

## 🌟 未来计划

- [ ] R1多模态支持（文本+图像）
- [ ] 网页界面开发
- [ ] 移动端适配
- [ ] 多语言支持

## 📚 文档索引

- [R1 LoRA系统](models/r1_lora_system/README.md) - 完整的本地训练指南
- [LoRA技术对比](models/r1_lora_system/lora_comparison.md) - 深度技术分析
- [视觉识别指南](vision/README.md) - 卡牌识别使用说明

---

*最后更新: 2024年7月31日*