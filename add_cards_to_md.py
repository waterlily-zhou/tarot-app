#!/usr/bin/env python3
"""
批量识别牌阵图片，把卡牌信息写入同名 MD 顶部
用法:
    python add_cards_to_md.py "data/Readings 6c05b8c41bcf40f9be4f7dd503141fd2"
依赖:
    • vision/simple_card_test.py 中的识别函数
    • 环境变量 GOOGLE_API_KEY（或 .env.local）
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Set

# 添加vision目录到Python路径，以便正确导入模块
vision_dir = Path(__file__).parent / "vision"
sys.path.insert(0, str(vision_dir))

# 导入识别函数
try:
    from simple_card_test import gemini_card_recognition, gemini_edge_detection
    print("✅ 成功导入识别模块")
except ImportError as e:
    print(f"❌ 无法导入识别模块: {e}")
    sys.exit(1)

# 匹配已有的cards注释行
CARDS_COMMENT = re.compile(r"^<!--\s*cards\s*:", re.IGNORECASE)

def collect_images(folder: Path) -> List[Path]:
    """收集文件夹中的图片文件"""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".pic"}
    images = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            images.append(file)
    return images

def recognize_cards_from_image(img_path: Path) -> Set[str]:
    """识别单张图片中的卡牌，返回卡牌集合"""
    print(f"🔍 识别图片: {img_path.name}")
    
    try:
        # 使用边缘检测版本，准确率更高
        result = gemini_edge_detection(str(img_path))
        
        if isinstance(result, dict) and "final_cards" in result:
            cards = result["final_cards"]
        else:
            cards = result
            
    except Exception as e:
        print(f"⚠️  边缘检测失败，尝试普通识别: {e}")
        try:
            cards = gemini_card_recognition(str(img_path), silent=True)
        except Exception as e2:
            print(f"❌ 普通识别也失败: {e2}")
            return set()
    
    card_set = set()
    if cards:
        for card in cards:
            name = card.get("card_name", "")
            orientation = card.get("orientation", "")
            if name and orientation:
                card_set.add(f"{name}({orientation})")
    
    return card_set

def update_md_with_cards(md_file: Path, cards_line: str):
    """在MD文件顶部添加或更新卡牌信息"""
    try:
        # 读取文件内容
        content = md_file.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        
        # 检查是否已有cards注释
        if lines and CARDS_COMMENT.match(lines[0]):
            # 替换第一行
            lines[0] = cards_line + "\n"
        else:
            # 在开头插入新行
            lines.insert(0, cards_line + "\n\n")
        
        # 写回文件
        md_file.write_text("".join(lines), encoding="utf-8")
        print(f"✅ 更新 {md_file.name}")
        
    except Exception as e:
        print(f"❌ 更新文件失败 {md_file.name}: {e}")

def main():
    if len(sys.argv) != 2:
        print("用法: python add_cards_to_md.py <Readings目录路径>")
        print('示例: python add_cards_to_md.py "data/Readings 6c05b8c41bcf40f9be4f7dd503141fd2"')
        sys.exit(1)
    
    readings_dir = Path(sys.argv[1])
    if not readings_dir.exists():
        print(f"❌ 目录不存在: {readings_dir}")
        sys.exit(1)
    
    print(f"🚀 开始处理目录: {readings_dir}")
    
    # 遍历所有MD文件
    processed_count = 0
    skipped_count = 0
    
    for md_file in readings_dir.glob("*.md"):
        print(f"\n📄 处理: {md_file.name}")
        
        # 查找同名文件夹
        folder_name = md_file.stem  # 不包含扩展名的文件名
        potential_folders = [
            folder for folder in readings_dir.iterdir() 
            if folder.is_dir() and folder.name.startswith(folder_name)
        ]
        
        if not potential_folders:
            print(f"⏭️  跳过 {md_file.name} (找不到同名文件夹)")
            skipped_count += 1
            continue
        
        # 使用第一个匹配的文件夹
        image_folder = potential_folders[0]
        print(f"📁 使用文件夹: {image_folder.name}")
        
        # 收集图片
        images = collect_images(image_folder)
        if not images:
            print(f"⏭️  跳过 {md_file.name} (文件夹中无图片)")
            skipped_count += 1
            continue
        
        print(f"🖼️  发现 {len(images)} 张图片")
        
        # 识别所有图片中的卡牌
        all_cards = set()
        for img in images:
            cards = recognize_cards_from_image(img)
            all_cards.update(cards)
        
        if not all_cards:
            print(f"⚠️  {md_file.name} 中未识别到任何卡牌")
            skipped_count += 1
            continue
        
        # 生成卡牌注释行
        cards_line = "<!--cards: " + "; ".join(sorted(all_cards)) + " -->"
        
        # 更新MD文件
        update_md_with_cards(md_file, cards_line)
        processed_count += 1
    
    print(f"\n🎉 处理完成！")
    print(f"✅ 成功处理: {processed_count} 个文件")
    print(f"⏭️  跳过: {skipped_count} 个文件")

if __name__ == "__main__":
    main() 