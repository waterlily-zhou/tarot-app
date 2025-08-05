# 🔐 塔罗牌训练隐私保护系统

## 概述

这是一个完整的隐私保护方案，确保塔罗牌解读训练数据的端到端安全：

- **🔒 AES-256加密**: 训练数据本地加密后上传
- **🔧 私有云API**: 完全离线训练，不经过第三方服务
- **🌐 Cloudflare Tunnel**: 端到端加密网络通道

## 🛡️ 隐私保护特性

### 1. 数据加密保护
- **AES-256加密算法**: 军用级别加密标准
- **PBKDF2密钥派生**: 10万次迭代防暴力破解
- **本地加密**: 明文数据不离开本地环境
- **完整性验证**: SHA-256哈希验证数据完整性

### 2. 网络隐私保护
- **离线模式训练**: HuggingFace、Transformers完全离线
- **阻止外部API**: 禁用Wandb、TensorBoard云服务
- **本地日志**: 所有日志保存在本地
- **DNS阻止**: 阻止telemetry域名解析

### 3. 端到端加密通道
- **Cloudflare Tunnel**: 零信任网络架构
- **自动TLS加密**: 所有流量自动加密
- **无需公网IP**: 通过tunnel安全访问
- **访问控制**: 可配置访问策略

## 🚀 快速开始

### 方法1: 一键部署（推荐）

```bash
cd scripts/r1_lora_system
python3 deploy_private_training.py
```

### 方法2: 手动步骤

#### 步骤1: 加密训练数据
```bash
python3 privacy_encryption.py --encrypt --data-dir ../../data/finetune
```

#### 步骤2: 配置私有云环境
```bash
python3 private_cloud_config.py
source private_env.sh
```

#### 步骤3: 设置Cloudflare Tunnel（可选）
```bash
python3 cloudflare_tunnel.py
```

#### 步骤4: 上传加密数据
```bash
./secure_upload.sh
```

#### 步骤5: 启动隐私训练
```bash
ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.158.37
python3 privacy_encryption.py --decrypt
source private_env.sh
python3 private_train.py
```

## 📁 文件结构

```
scripts/r1_lora_system/
├── privacy_encryption.py      # AES-256加密工具
├── private_cloud_config.py    # 私有云配置
├── cloudflare_tunnel.py       # Cloudflare Tunnel设置
├── deploy_private_training.py # 一键部署脚本
├── train_70b_qlora.py         # 原始训练脚本
├── private_train.py           # 隐私保护训练脚本
├── secure_upload.sh           # 安全上传脚本
├── privacy_dashboard.html     # 隐私监控面板
└── README_PRIVACY.md          # 本文档
```

## 🔧 详细配置

### AES-256加密配置

```python
# 加密算法: AES-256-CBC
# 密钥派生: PBKDF2-HMAC-SHA256
# 迭代次数: 100,000
# 盐值: 固定盐确保一致性
```

### 私有云环境变量

```bash
export HF_HUB_OFFLINE=1           # HuggingFace离线模式
export TRANSFORMERS_OFFLINE=1     # Transformers离线模式
export WANDB_DISABLED=true        # 禁用Wandb
export TENSORBOARD_LOG_DIR=./logs # 本地日志目录
```

### Cloudflare Tunnel配置

```yaml
tunnel: [your-tunnel-id]
credentials-file: ~/.cloudflared/[tunnel-id].json

ingress:
  - hostname: tarot-training.your-domain.com
    service: ssh://209.20.158.37:22
  - hostname: tarot-logs.your-domain.com  
    service: http://209.20.158.37:6006
  - service: http_status:404
```

## 🔍 安全监控

### 本地监控命令

```bash
# GPU使用率监控
ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.158.37 "watch -n 5 nvidia-smi"

# 训练日志监控  
ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.158.37 "tail -f training.log"

# 网络连接监控
ssh -i ~/.ssh/tarot-training-key ubuntu@209.20.158.37 "netstat -an | grep ESTABLISHED"
```

### 隐私验证检查

```bash
# 检查外部连接（应该为空或极少）
ss -tuln | grep :443

# 检查进程（不应有wandb等外部服务）
ps aux | grep -E "(wandb|tensorboard)"

# 检查环境变量
env | grep -E "(HF_|TRANSFORMERS_|WANDB_)"
```

## ⚠️ 安全注意事项

1. **密码管理**: 
   - 使用强密码进行AES加密
   - 不要在脚本中硬编码密码
   - 考虑使用密码管理器

2. **密钥安全**:
   - SSH密钥权限设置为600
   - 定期轮换访问密钥
   - 不要提交密钥到版本控制

3. **网络安全**:
   - 使用VPN额外保护
   - 定期检查防火墙规则
   - 监控异常网络活动

4. **数据清理**:
   - 训练完成后删除服务器上的明文数据
   - 安全删除临时文件
   - 清理缓存和日志

## 🆘 故障排除

### 加密相关问题

**问题**: `cryptography module not found`
```bash
pip install cryptography
```

**问题**: 解密失败
- 检查密码是否正确
- 确认加密文件完整性
- 验证文件权限

### 网络连接问题

**问题**: SSH连接失败
- 检查SSH密钥权限
- 验证Lambda服务器状态
- 确认防火墙设置

**问题**: Cloudflare Tunnel连接失败
- 检查DNS记录配置
- 验证tunnel认证状态
- 确认域名解析

### 训练问题

**问题**: 模型下载失败
- 确认离线模式设置
- 检查本地模型缓存
- 验证网络连接

**问题**: GPU内存不足
- 减少batch_size
- 使用gradient_checkpointing
- 调整max_seq_length

## 📞 技术支持

如果遇到隐私保护相关问题：

1. 检查 `privacy_dashboard.html` 状态面板
2. 查看详细错误日志
3. 验证每个组件的配置
4. 确认网络和权限设置

## 🔄 版本更新

- v1.0: 基础AES-256加密
- v1.1: 添加私有云配置
- v1.2: 集成Cloudflare Tunnel
- v1.3: 一键部署脚本
- v1.4: 隐私监控面板

---

🔐 **记住**: 隐私保护是一个持续的过程，请定期检查和更新安全配置！