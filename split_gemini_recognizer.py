#!/usr/bin/env python3
"""
分块调用 Gemini：把横向切成 N 块，分别识别，再合并
"""
import cv2, os, google.generativeai as genai, uuid, shutil
from PIL import Image
from pathlib import Path

# 加载环境变量
def load_env_file():
    env_file = Path('.env.local')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()

IMG = "data/card_images/spread_0_4821735726296_.pic.jpg"
N_BLOCKS = 3
TMP_DIR = Path("tmp_blocks")
TMP_DIR.mkdir(exist_ok=True)
API = os.getenv("GEMINIAPI")
genai.configure(api_key=API)
model = genai.GenerativeModel("gemini-1.5-flash")

def call_gemini(p):
    prompt = """这张图片中有哪些塔罗牌？请按以下格式输出：
    卡牌名称,正位/逆位
    
    例如：
    权杖三,正位
    圣杯国王,逆位
    
    如果没有塔罗牌，请回答"无"。只输出识别结果，不要解释。"""
    rsp = model.generate_content([prompt, Image.open(p)])
    return rsp.text.strip()

def main():
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    
    # 增加重叠的分块策略
    overlap = 0.2  # 20%重叠
    step = int(w * (1 - overlap) / (N_BLOCKS - 1)) if N_BLOCKS > 1 else w
    block_width = int(w / N_BLOCKS * (1 + overlap))
    
    results = {}
    
    print(f"🔪 将图片切成 {N_BLOCKS} 块，每块宽度 {block_width}px，重叠 {int(overlap*100)}%")
    
    for i in range(N_BLOCKS):
        print(f"\n📦 处理第 {i+1} 块...")
        
        # 计算起始和结束位置（带重叠）
        start = max(0, i * step)
        end = min(w, start + block_width)
        
        sub = img[:, start:end]
        fn = TMP_DIR / f"blk_{i}.jpg"
        cv2.imwrite(str(fn), sub)
        print(f"   区域: {start}-{end}px，已保存: {fn}")
        
        txt = call_gemini(fn)
        print(f"   Gemini识别结果:\n{txt}")
        
        # 解析结果
        for line in txt.splitlines():
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    orient = parts[1].strip()
                    results[(name, orient)] = i  # 记录在哪个块中发现
    
    # 清理
    shutil.rmtree(TMP_DIR)
    
    print(f"\n=== 合并去重后共识别到 {len(results)} 张卡牌 ===")
    for (name, orient), block_id in results.items():
        print(f" • {name} ({orient}) - 发现于第{block_id+1}块")

if __name__ == "__main__":
    main() 