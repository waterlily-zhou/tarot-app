# 同一个项目中
import torch
import tvm  # MLC-AI

# 图像处理用 PyTorch MPS
vision_device = torch.device("mps")
vision_model = load_vision_model().to(vision_device)

# 文本生成用 MLC-AI
llm_model = load_mlc_llm_model()  # 自动用 ANE + GPU