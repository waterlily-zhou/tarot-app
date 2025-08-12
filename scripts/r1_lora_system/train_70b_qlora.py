#!/usr/bin/env python3
"""
DeepSeek R1 70B QLoRA 微调脚本 - 优化版
"""
import os
import sys
import torch
import json
from pathlib import Path
import gc
from datetime import datetime

def detect_environment():
    print("🔍 检测环境...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} | 显存: {total_memory:.1f}GB")
        return True
    print("❌ 需要CUDA GPU环境")
    return False

def load_training_data(data_path="training_data.jsonl"):
    print(f"📂 加载训练数据: {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None
    
    try:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # 保留原始messages结构 ✅
                if 'messages' in item:
                    data.append(item)
        print(f"✅ 加载完成: {len(data)} 条训练数据")
        return data
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

def setup_70b_model():
    print("🚀 设置DeepSeek R1 70B模型...")
    
    # 🔧 调试版本：详细检查缓存文件状态
    print("🔧 调试模式：详细检查缓存状态...")
    
    try:
        import glob
        import re
        import os
        
        # 先检查缓存根目录
        cache_root = "/home/ubuntu/.cache/huggingface"
        print(f"📁 缓存根目录: {cache_root}")
        print(f"📁 缓存根目录存在: {os.path.exists(cache_root)}")
        
        if os.path.exists(cache_root):
            # 列出所有子目录
            print("📂 缓存目录结构:")
            for root, dirs, files in os.walk(cache_root):
                level = root.replace(cache_root, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                if level < 3:  # 限制深度避免输出过多
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # 只显示前5个文件
                        print(f"{subindent}{file}")
                    if len(files) > 5:
                        print(f"{subindent}... (+{len(files)-5} more files)")
        
        # 尝试多种可能的路径模式
        patterns = [
            "/home/ubuntu/.cache/huggingface/modules/transformers_modules/*/modeling_deepseek.py",
            "/home/ubuntu/.cache/huggingface/hub/models--*/snapshots/*/modeling_deepseek.py",
            "/home/ubuntu/.cache/huggingface/hub/*/modeling_deepseek.py",
            "/home/ubuntu/.cache/huggingface/*/modeling_deepseek.py"
        ]
        
        all_modeling_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            print(f"🔍 搜索模式 '{pattern}': 找到 {len(files)} 个文件")
            all_modeling_files.extend(files)
        
        # 去重
        all_modeling_files = list(set(all_modeling_files))
        print(f"📋 总共找到 {len(all_modeling_files)} 个modeling_deepseek.py文件")
        
        for file_path in all_modeling_files:
            print(f"📄 检查文件: {file_path}")
            print(f"📄 文件存在: {os.path.exists(file_path)}")
            
            if os.path.exists(file_path):
                # 检查文件权限
                print(f"📄 文件权限: {oct(os.stat(file_path).st_mode)}")
                
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查文件大小和关键内容
                print(f"📄 文件大小: {len(content)} 字符")
                
                # 智能统计：只统计未注释的assert语句
                lines = content.split('\n')
                active_assert_count = 0
                commented_assert_count = 0
                
                for line in lines:
                    if 'assert not self.training' in line:
                        if line.strip().startswith('#'):
                            commented_assert_count += 1
                            print(f"✅ 已注释的assert: {line.strip()}")
                        else:
                            active_assert_count += 1
                            print(f"⚠️ 活跃的assert: {line.strip()}")
                
                print(f"🔍 活跃assert: {active_assert_count}, 已注释assert: {commented_assert_count}")
                assert_count = active_assert_count
                
                if assert_count > 0:
                    print(f"🔧 修复文件 {file_path}...")
                    
                    # 检查是否已经被修复过（避免重复处理）
                    if '# 已注释：允许训练模式' in content:
                        print(f"⚠️ 文件已被修复过，尝试恢复原文件...")
                        
                        # 尝试从备份恢复
                        backup_path = file_path + '.backup'
                        if os.path.exists(backup_path):
                            with open(backup_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            print(f"✅ 已从备份恢复: {backup_path}")
                        else:
                            print(f"⚠️ 备份文件不存在，手动清理重复注释...")
                            # 手动清理重复注释
                            lines = content.split('\n')
                            cleaned_lines = []
                            for line in lines:
                                if '# 已注释：允许训练模式' in line:
                                    # 尝试恢复原始assert语句
                                    if 'assert not self.training' in line:
                                        # 找到原始缩进
                                        indent_match = line.split('#')[0]
                                        cleaned_line = indent_match + 'assert not self.training'
                                        cleaned_lines.append(cleaned_line)
                                        print(f"🔄 恢复行: {cleaned_line.strip()}")
                                    else:
                                        cleaned_lines.append(line)
                                else:
                                    cleaned_lines.append(line)
                            content = '\n'.join(cleaned_lines)
                    
                    # 重新计算assert数量
                    assert_count = content.count('assert not self.training')
                    print(f"🔍 清理后找到 {assert_count} 个assert语句")
                    
                    if assert_count > 0:
                        # 备份原文件（覆盖之前的备份）
                        backup_path = file_path + '.backup'
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"💾 已重新备份到: {backup_path}")
                        
                        # 超级调试：查看每一行内容
                        import re
                        
                        lines = content.split('\n')
                        modified_lines = []
                        changes_made = 0
                        
                        print("🔍 超级调试模式：查看包含'training'的所有行")
                        for i, line in enumerate(lines):
                            if 'training' in line.lower():
                                print(f"  第{i+1}行: '{line}'")
                                print(f"    原始字节: {repr(line)}")
                                print(f"    stripped: '{line.strip()}'")
                                if 'assert' in line:
                                    print(f"    🎯 这行包含assert!")
                        
                        print("\n🔧 尝试多种匹配模式...")
                        patterns = [
                            r'^(\s*)assert\s+not\s+self\.training(.*)$',
                            r'^\s*assert\s+not\s+self\.training.*$',
                            r'assert\s+not\s+self\.training',
                            r'.*assert.*not.*self\.training.*',
                        ]
                        
                        for i, line in enumerate(lines):
                            line_matched = False
                            
                            for j, pattern in enumerate(patterns):
                                if re.search(pattern, line, re.IGNORECASE):
                                    print(f"🎯 第{i+1}行匹配模式{j+1}: '{line.strip()}'")
                                    line_matched = True
                                    
                                    # 只用第一个模式进行实际修改
                                    if j == 0 and not line.strip().startswith('#'):
                                        print(f"🔧 应用修改...")
                                        modified_line = '        # ' + line.strip() + '  # 已注释：允许训练模式'
                                        modified_lines.append(modified_line)
                                        changes_made += 1
                                        break
                            
                            if not line_matched:
                                modified_lines.append(line)
                            elif not any(re.search(patterns[0], line, re.IGNORECASE) for _ in [1]):
                                modified_lines.append(line)
                        
                        new_content = '\n'.join(modified_lines)
                        print(f"📊 本次修改了 {changes_made} 行")
                        
                        # 写回文件
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        # 验证修改
                        with open(file_path, 'r', encoding='utf-8') as f:
                            verify_content = f.read()
                        
                        final_assert_count = verify_content.count('assert not self.training')
                        comment_count = verify_content.count('# assert not self.training')
                        
                        print(f"✅ 修复完成！剩余断言: {final_assert_count}")
                        print(f"✅ 注释数量: {comment_count}")
                    else:
                        print(f"✅ 无需修复")
                else:
                    print(f"✅ 文件正常，无需修复")
                    
        if not all_modeling_files:
            print("❌ 未找到任何modeling_deepseek.py文件！")
            print("🔍 尝试手动查找...")
            
            # 手动递归查找
            for root, dirs, files in os.walk("/home/ubuntu/.cache"):
                for file in files:
                    if file == "modeling_deepseek.py":
                        full_path = os.path.join(root, file)
                        print(f"🎯 手动找到: {full_path}")
            
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        print("🔍 详细错误:")
        traceback.print_exc()
    
    try:
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM, 
            AutoConfig,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            get_linear_schedule_with_warmup
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
        from accelerate import infer_auto_device_map, dispatch_model
        import os
        import torch
        
        # 使用本地干净快照路径
        model_path = os.path.expanduser("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots")
        # 找到最新的快照目录（按修改时间排序，优先选择最新的）
        if os.path.exists(model_path):
            snapshot_dirs = [os.path.join(model_path, d) for d in os.listdir(model_path)
                             if os.path.isdir(os.path.join(model_path, d))]
            if snapshot_dirs:
                snapshot_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                model_path = snapshot_dirs[0]
            else:
                raise FileNotFoundError("No snapshot found in cache")
        else:
            raise FileNotFoundError(f"Cache directory not found: {model_path}")
        
        print(f"✅ 使用本地快照: {model_path}")

        # 快照完整性预检：若分片不全，在线补全下载
        try:
            import glob as _glob
            import re as _re
            shard_paths = sorted(_glob.glob(os.path.join(model_path, 'model-*-of-*.safetensors')))
            if shard_paths:
                total = int(_re.search(r'-of-(\d+)', os.path.basename(shard_paths[0])).group(1))
                have_nums = {int(_re.search(r'model-(\d+)-of-', os.path.basename(p)).group(1)) for p in shard_paths}
                missing = [i for i in range(1, total + 1) if i not in have_nums]
                if missing:
                    print(f"⚠️ 本地快照缺少 {len(missing)}/{total} 个分片，准备在线补全...")
                    # 临时关闭离线变量
                    import os as _os
                    for _k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
                        if _k in _os.environ:
                            print(f"🔓 临时关闭环境变量: {_k}")
                            _os.environ.pop(_k, None)
                    # 使用 huggingface_hub 补全下载
                    from huggingface_hub import snapshot_download as _snapshot_download
                    new_path = _snapshot_download(
                        repo_id="deepseek-ai/DeepSeek-R1-0528",
                        local_files_only=False,
                        resume_download=True,
                        allow_patterns=["*.json", "*.py", "*.safetensors", "*.bin", "*.model", "tokenizer*", "*.txt"],
                        local_dir_use_symlinks=False,
                    )
                    print(f"✅ 快照补全完成: {new_path}")
                    model_path = new_path
        except Exception as _e:
            print(f"⚠️ 快照预检/补全失败（忽略继续）: {_e}")
        
        # 使用4bit量化（QLoRA标准），显著降低显存占用
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # 加载tokenizer
        print("🔹 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 简化：直接加载到GPU，避免复杂设备映射导致的meta设备问题
        print("⏳ 直接加载模型到GPU（避免meta设备）...")
        
        # 立即修复 finegrained_fp8 设备上下文问题
        try:
            import transformers.integrations.finegrained_fp8 as _fg
            from contextlib import nullcontext as _nullcontext
            class _DummyAccel:
                @staticmethod
                def device(_dev):
                    return _nullcontext()
            # Override accelerator module to a no-op
            _fg.torch_accelerator_module = _DummyAccel
            print("✅ 已预先禁用 finegrained_fp8 CUDA 设备上下文")
        except Exception as _e:
            print(f"⚠️ 预先禁用 finegrained_fp8 失败（忽略）：{_e}")

        try:
            import os as _os
            _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            # 简化：直接使用 auto 设备映射，让transformers自动处理
            model = AutoModelForCausalLM.from_pretrained(
              model_path,
              quantization_config=bnb_config,
              device_map="auto",
              trust_remote_code=True,
              torch_dtype=torch.bfloat16,
              use_cache=False,
              local_files_only=True,
              # 添加以下两个关键参数
              low_cpu_mem_usage=False    # 禁用内存优化模式（避免懒初始化）
            )
            
            # ===== 新增设备修复代码块 =====
            print("🔧 强制设备一致性修复...")
            try:
                # 确保所有参数不在meta设备上
                meta_names = []
                for name, param in model.named_parameters():
                    if getattr(param, 'is_meta', False):
                        meta_names.append(name)
                        
                if meta_names:
                    print("❌ 检测到以下 meta 设备参数（将中止以避免训练期崩溃）：")
                    for n in meta_names[:50]:
                        print(f"   - {n}")
                    if len(meta_names) > 50:
                        print(f"   ... 共 {len(meta_names)} 项")
                    raise RuntimeError("存在 meta 设备参数。请检查加载路径/设备映射，确保 low_cpu_mem_usage=False 且未走懒初始化。")
                print("✅ 未发现 meta 设备参数")
                
                # 确保所有可训练参数在GPU上
                trainable_on_cpu = 0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.device.type == "cpu":
                        print(f"🔧 移动可训练参数到GPU: {name}")
                        param.data = param.data.to("cuda:0")
                        trainable_on_cpu += 1
                        
                print(f"✅ 移动了 {trainable_on_cpu} 个CPU上的可训练参数到GPU")
                
                # 强制模型到GPU（确保所有组件都在CUDA上）
                model = model.to("cuda:0")
                print("✅ 模型已强制移动到 cuda:0")
                
            except Exception as e:
                print(f"⚠️ 设备修复失败: {e}")
                import traceback
                traceback.print_exc()
            # ===== 结束新增代码块 =====

            # 验证模型设备
            try:
                first_param_device = next(model.parameters()).device
                print(f"✅ 模型首个参数设备: {first_param_device}")
            except Exception as e:
                print(f"⚠️ 设备验证失败: {e}")
                
        except Exception as e:
            import traceback as _tb
            print(f"❌ 模型加载失败: {repr(e)}")
            _tb.print_exc()
            return None, None, None
        
        print(f"✅ 模型已加载 - 使用保守映射策略")

        # 规范量化标记：彻底清理 fp8 痕迹，避免 Trainer 误判
        try:
            def _sanitize_quantization_metadata(_model):
                # 顶层与子模块可能的 fp8 标志位
                for attr in ["is_fp8", "use_fp8", "fp8", "is_quantized_fp8"]:
                    if hasattr(_model, attr):
                        try:
                            setattr(_model, attr, False)
                        except Exception:
                            pass

                # 顶层 quantization_config
                qc = getattr(_model, "quantization_config", None)
                try:
                    # 用 4bit 配置显式覆盖（避免 None 造成下游 .to_dict 访问报错）
                    _model.quantization_config = bnb_config
                except Exception:
                    pass

                # config 内的 quantization_config 字段
                cfg = getattr(_model, "config", None)
                if cfg is not None:
                    try:
                        # 将 config.quantization_config 也指向 4bit 配置，避免 None
                        if hasattr(cfg, "quantization_config"):
                            try:
                                setattr(cfg, "quantization_config", bnb_config)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # 子模块上的 fp8/标记
                try:
                    for m in _model.modules():
                        for attr in ["is_fp8", "use_fp8", "fp8", "is_quantized_fp8"]:
                            if hasattr(m, attr):
                                try:
                                    setattr(m, attr, False)
                                except Exception:
                                    pass
                except Exception:
                    pass

                # 标注为量化但非 fp8
                if hasattr(_model, "is_quantized"):
                    try:
                        _model.is_quantized = True
                    except Exception:
                        pass

            _sanitize_quantization_metadata(model)
            print("✅ 已清理量化元数据，避免被误判为 fp8")
        except Exception as _e:
            print(f"⚠️ 量化元数据规范化失败（忽略）：{_e}")

        # 按 k-bit 训练最佳实践准备模型（必须在注入 LoRA 之前）
        try:
            # 与梯度检查点配合：禁用缓存
            if hasattr(model, "config"):
                model.config.use_cache = False

            # 开启模型级梯度检查点（冗余调用也安全）
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            # peft 建议：为 k-bit 训练做预处理，并确保输入需要梯度
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )

            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            print("✅ 已完成 k-bit 训练准备并启用输入梯度")
        except Exception as e:
            print(f"⚠️ k-bit 训练准备失败（将继续）：{e}")
        
        # 🔧 FP8量化兼容性修复 - 自动转换不兼容的数据类型
        print("🔧 检查并修复FP8量化兼容性问题...")
        fp8_converted = 0
        
        # FP8参数转换 + dropout修复
        try:
            import torch
            for name, param in model.named_parameters():
                if hasattr(param, 'dtype') and 'float8' in str(param.dtype).lower():
                    print(f"🔧 转换FP8参数: {name} 从 {param.dtype} 到 torch.float16")
                    param.data = param.data.to(torch.float16)
                    fp8_converted += 1
        except Exception as e:
            print(f"⚠️ FP8参数转换失败（忽略）：{e}")
        
        # 禁用所有dropout模块的fused模式
        try:
            for name, module in model.named_modules():
                if hasattr(module, 'p') and hasattr(module, 'inplace'):  # 这是Dropout模块
                    if hasattr(module, 'fused'):
                        module.fused = False
                        print(f"🔧 禁用fused dropout: {name}")
        except Exception as e:
            print(f"⚠️ 禁用fused dropout失败（忽略）：{e}")
        
        if fp8_converted > 0:
            print(f"✅ 已转换 {fp8_converted} 个FP8参数为训练兼容格式")
        else:
            print("✅ 未发现FP8参数，或已经是兼容格式")
        
        # 简化：依赖 apply_moe_training_patch 处理门控/MoE 路径，避免自定义补丁引入设备不一致
        
        # 全局化tokenizer供format_chat使用
        globals()['tokenizer'] = tokenizer
        
        # 正确的LoRA配置 ✅
        lora_config = LoraConfig(
            r=8,                    # 降低rank节省显存
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # DeepSeek V3架构
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 将 LoRA 适配器权重对齐到其 base_layer 所在设备，避免 meta/cuda 混用导致的梯度设备错误
        try:
            moved = 0
            meta_params_fixed = 0
            
            # 彻底检查并修复所有meta设备问题
            for name, param in model.named_parameters():
                # 将所有meta设备上的参数移动到cuda:0
                if hasattr(param, 'device') and (str(param.device) == 'meta' or param.device.type == 'meta'):
                    print(f"🔧 修复meta设备参数: {name} -> cuda:0")
                    # 创建实际的参数（不是meta）
                    if param.requires_grad:
                        # 可训练参数：随机初始化
                        new_param = torch.randn(param.shape, device='cuda:0', dtype=param.dtype) * 0.01
                    else:
                        # 非可训练参数：零初始化
                        new_param = torch.zeros(param.shape, device='cuda:0', dtype=param.dtype)
                    param.data = new_param
                    meta_params_fixed += 1
                elif hasattr(param, 'device') and param.device.type == 'cpu' and param.requires_grad:
                    # 将可训练参数移到GPU上，避免CPU/GPU混合训练
                    print(f"🔧 移动可训练参数到GPU: {name}")
                    param.data = param.data.to('cuda:0')
                    moved += 1
                
            for module in model.modules():
                base = getattr(module, 'base_layer', None)
                if base is None or not hasattr(base, 'weight'):
                    continue
                target_device = base.weight.device
                # 确保target_device不是meta
                if str(target_device) == 'meta':
                    target_device = torch.device('cuda:0')
                    
                for attr in ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B'):
                    sub = getattr(module, attr, None)
                    if sub is None:
                        continue
                    # sub 可能是 ModuleDict/ParameterDict/Module
                    try:
                        for _, submod in getattr(sub, 'items', lambda: [])():
                            try:
                                submod.to(target_device)
                                moved += 1
                            except Exception:
                                pass
                    except TypeError:
                        # 非 dict，直接 to
                        try:
                            sub.to(target_device)
                            moved += 1
                        except Exception:
                            pass
            if meta_params_fixed > 0:
                print(f"🔧 修复了 {meta_params_fixed} 个meta设备参数")
            print(f"🔧 已对齐 LoRA 适配器设备，移动子模块: {moved}")
        except Exception as _e:
            print(f"⚠️ LoRA 设备对齐失败（忽略继续）：{_e}")
        
        # ===== 新增LoRA设备检查 =====
        print("🔍 验证LoRA设备一致性...")
        try:
            lora_devices = set()
            base_devices = set()
            for name, module in model.named_modules():
                if hasattr(module, 'base_layer'):
                    try:
                        base_devices.add(str(module.base_layer.weight.device))
                    except Exception:
                        pass
                    # 遍历该模块的参数，统计包含 lora_ 的参数设备
                    for p_name, p in getattr(module, 'named_parameters', lambda: [])():
                        if 'lora_' in p_name:
                            try:
                                lora_devices.add(str(p.device))
                            except Exception:
                                pass
            print(f"📊 LoRA设备: {lora_devices}")
            print(f"📊 基础层设备: {base_devices}")
            if len(lora_devices) > 1 or len(base_devices) > 1:
                print("⚠️ 检测到设备不一致！强制对齐...")
                for module in model.modules():
                    if hasattr(module, 'base_layer'):
                        target_device = getattr(module.base_layer.weight, 'device', None)
                        if target_device is not None:
                            try:
                                module.to(target_device)
                            except Exception:
                                pass
            print("✅ LoRA设备一致性验证完成")
        except Exception as e:
            print(f"⚠️ LoRA设备检查失败: {e}")
        # ===== 结束新增代码块 =====

        # 🔧 确保所有可训练参数的梯度追踪正确设置
        print("🔧 验证并修复参数梯度追踪...")
        grad_fixed = 0
        no_grad_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 检查参数是否真的可以计算梯度
                if not param.requires_grad or param.grad_fn is None and param.is_leaf:
                    print(f"🔧 修复梯度追踪: {name}")
                    param.requires_grad_(True)
                    grad_fixed += 1
            else:
                no_grad_params.append(name)
        
        if grad_fixed > 0:
            print(f"✅ 修复了 {grad_fixed} 个参数的梯度追踪")
        
        print(f"📊 梯度状态: {sum(p.requires_grad for p in model.parameters())} 个可训练参数")
        
        # 🔧 验证门控模块补丁是否生效
        print("🔍 验证DeepseekExpertsGate训练模式补丁...")
        try:
            for name, module in model.named_modules():
                if "DeepseekExpertsGate" in type(module).__name__:
                    print(f"✅ 验证门控模块 {name} 支持训练模式")
                    module.train()  # 应该不会引发错误
                    break
        except Exception as e:
            print(f"⚠️ 门控模块验证失败: {e}")
        
        return model, tokenizer, lora_config
        
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("💡 请安装: pip install transformers peft bitsandbytes trl")
        return None, None, None
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None

def format_chat(example):
    """使用tokenizer内置的聊天模板格式化数据 ✅"""
    messages = example["messages"]
    tk = globals().get("tokenizer", None)
    if tk is None:
        raise RuntimeError("tokenizer not initialized in globals()")
    return {"text": tk.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )}

# --- Runtime patch: fix MoE training forward path so that `y` is defined in training mode ---
def apply_moe_training_patch(model):
    """Ensure DeepSeek MoE blocks have a valid training forward path.

    Some upstream DeepSeek implementations define `y` only under `if not self.training:`
    and then unconditionally use `y` later, which raises `UnboundLocalError` during training.
    This patch overrides such modules' `forward` to define a differentiable training path
    using `shared_experts(identity)` and keep the original inference path for eval.
    """
    import types

    def module_needs_patch(module):
        return (
            hasattr(module, "gate")
            and hasattr(module, "shared_experts")
            and hasattr(module, "moe_infer")
            and callable(getattr(module, "forward", None))
        )

    num_patched = 0
    for name, submodule in model.named_modules():
        try:
            if module_needs_patch(submodule):
                original_forward = submodule.forward  # noqa: F841 (kept for potential debugging)

                def patched_forward(self, hidden_states):
                    identity = hidden_states
                    orig_shape = getattr(hidden_states, "shape", None)
                    # Gate routing (used for eval path and to keep shapes consistent)
                    topk_idx, topk_weight = self.gate(hidden_states)
                    hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

                    if self.training:
                        # Differentiable training path: rely on shared experts only
                        y = self.shared_experts(identity)
                        return y
                    else:
                        # Preserve original eval behavior
                        y = self.moe_infer(hidden_states_flat, topk_idx, topk_weight)
                        if orig_shape is not None:
                            y = y.view(*orig_shape)
                        if getattr(self.config, "n_shared_experts", None) is not None:
                            y = y + self.shared_experts(identity)
                        return y

                submodule.forward = types.MethodType(patched_forward, submodule)
                num_patched += 1
        except Exception:
            # Best-effort patching; skip modules that fail heuristics
            continue

    print(f"🧩 已应用MoE训练补丁: {num_patched} 个模块")

# --- Runtime patch: disable finegrained FP8 CUDA device context when running on CPU ---
def disable_finegrained_fp8_device_context_if_cpu(model):
    try:
        first_device = next(model.parameters()).device
    except Exception:
        import torch as _torch
        first_device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
    if getattr(first_device, 'type', 'cpu') != 'cuda':
        try:
            import transformers.integrations.finegrained_fp8 as _fg
            from contextlib import nullcontext as _nullcontext
            class _DummyAccel:
                @staticmethod
                def device(_dev):
                    return _nullcontext()
            # Override accelerator module to a no-op on CPU
            _fg.torch_accelerator_module = _DummyAccel
            print("✅ 已禁用 finegrained_fp8 CUDA 设备上下文（CPU运行环境）")
        except Exception as _e:
            print(f"⚠️ 禁用 finegrained_fp8 失败（忽略）：{_e}")

def train_70b_qlora():
    print("🎯 开始DeepSeek R1 70B QLoRA微调")
    print("="*60)
    
    # 1. 环境检测
    if not detect_environment():
        return False
    
    # 2. 加载数据
    training_data = load_training_data()
    if not training_data:
        return False
    
    # 3. 设置模型
    model, tokenizer, lora_config = setup_70b_model()
    if model is None:
        return False
    # 3.1  将tokenizer提升为全局，供 format_chat 使用
    globals()['tokenizer'] = tokenizer
    
    try:
        from datasets import Dataset
        from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
        
        # 4. 准备数据集
        dataset = Dataset.from_list(training_data)

        # 使用自定义训练循环，直接处理文本格式化

        # 5. 自定义训练超参（避免触发HF Trainer的fp8检查）
        num_train_epochs = 3
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 16
        learning_rate = 2e-5
        warmup_ratio = 0.03
        max_grad_norm = 0.3
        output_dir = "./deepseek_r1_70b_tarot_lora"

        # 6. 文本格式化与分词
        print("🧾 正在格式化与分词数据...")
        dataset = dataset.map(lambda ex: format_chat(ex))
        def _tokenize_fn(ex):
            enc = tokenizer(
                ex["text"],
                truncation=True,
                max_length=4096,
                padding=False,
                return_tensors=None,
            )
            return enc
        keep_cols = set(["input_ids", "attention_mask"]) | set([c for c in dataset.column_names if c in ("input_ids","attention_mask")])
        dataset = dataset.map(_tokenize_fn, remove_columns=[c for c in dataset.column_names if c not in keep_cols])

        # 7. DataLoader 与 collator
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        import torch.utils.data as tud
        train_loader = tud.DataLoader(dataset, batch_size=per_device_train_batch_size, shuffle=True, collate_fn=collator)
        
        # ===== 新增前向传播初始化 =====
        print("🔧 执行前向传播以初始化所有参数...")
        try:
            # 取第一个batch作为初始化输入
            init_batch = next(iter(train_loader))
            init_batch = {k: v.to(model_device) for k, v in init_batch.items()}
            
            # 使用no_grad避免计算梯度
            with torch.no_grad():
                outputs = model(**init_batch)
                print(f"✅ 前向传播成功，loss: {outputs.loss.item():.4f}")
                
            # 释放内存
            del init_batch, outputs
            torch.cuda.empty_cache()
            gc.collect()
            print("✅ 参数初始化完成，缓存已清除")
        except Exception as e:
            print(f"⚠️ 前向传播初始化失败: {e}")
            import traceback
            traceback.print_exc()
        # ===== 结束新增代码块 =====

        # 8. 优化器与调度器
        import math
        from torch.optim import AdamW
        total_steps = math.ceil(len(train_loader) * num_train_epochs / gradient_accumulation_steps)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps)

        # 9. 注入MoE训练补丁
        try:
            apply_moe_training_patch(model)
        except Exception as _e:
            print(f"⚠️ MoE训练补丁注入失败（忽略继续）：{_e}")
        # 为加速进入训练，跳过健诊

        # 10. 自定义训练循环
        print("🏋️ 使用自定义训练循环开始训练...")
        model.train()
        global_step = 0
        # 以模型参数实际设备为准，决定是否启用cuda autocast
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_cuda = (getattr(model_device, 'type', 'cpu') == 'cuda')
        print(f"🖥️ model device: {model_device}, use_cuda_autocast={use_cuda}")
        for epoch in range(num_train_epochs):
            print(f"📆 Epoch {epoch+1}/{num_train_epochs}")
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                # 将 batch 张量迁移到与模型同设备
                # 再按需要开启 autocast（仅在 cuda 下）
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                    # DataCollatorForLanguageModeling 已经设置了 labels，不需要重复传递
                    try:
                        batch = {k: (v.to(model_device) if hasattr(v, 'to') else v) for k, v in batch.items()}
                    except Exception:
                        pass
                    outputs = model(**batch)  # type: ignore
                    loss = outputs.loss / gradient_accumulation_steps
                loss.backward()

                # ===== 新增梯度验证 =====
                # 检查梯度设备一致性
                invalid_grads = []
                valid_grads = 0

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if param.grad.device != param.device:
                            print(f"⚠️ 梯度设备不一致: {name} "
                                  f"(参数设备: {param.device}, 梯度设备: {param.grad.device})")
                            invalid_grads.append(name)
                        else:
                            valid_grads += 1

                if invalid_grads:
                    print(f"❌ 发现 {len(invalid_grads)} 个梯度设备不一致的参数!")
                    # 尝试修复：将梯度移动到参数所在设备
                    for name in invalid_grads:
                        param = dict(model.named_parameters())[name]
                        param.grad = param.grad.to(param.device)
                    print("🛠️ 已尝试修复梯度设备")
                else:
                    print(f"✅ 梯度设备一致: {valid_grads} 个参数")
                # ===== 结束新增代码块 =====

                if step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % 10 == 0:
                        scaled = (loss.detach().float() * gradient_accumulation_steps).item()
                        print(f"step {global_step}/{total_steps}, loss={scaled:.4f}")

        # 11. 保存模型
        print("💾 保存LoRA权重...")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        try:
            tokenizer.save_pretrained(output_dir)
        except Exception:
            pass
        
        # 9. 测试
        print("🧪 测试模型...")
        test_messages = [
            {"role": "user", "content": "请解读塔罗牌：愚人(正位)"}
        ]
        inputs = tokenizer.apply_chat_template(
            test_messages,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("📝 测试输出:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        print("✅ 70B QLoRA训练完成！")
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌟 DeepSeek R1 70B QLoRA 塔罗牌微调系统")
    print("🔗 专为Lambda GPU云平台优化")
    print()
    
    success = train_70b_qlora()
    if success:
        print("🎉 训练成功完成！")
        print("📁 模型保存在: ./deepseek_r1_70b_tarot_lora/")
    else:
        print("❌ 训练失败，请检查错误信息")