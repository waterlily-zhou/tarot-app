#!/usr/bin/env python3
"""
使用单张卡牌图片训练韦特塔罗识别系统
"""

from waite_tarot_recognizer import WaiteTarotRecognizer
from pathlib import Path
import time

def main():
    print("🎴 韦特塔罗单张卡牌训练系统")
    print("="*50)
    
    # 初始化识别器
    recognizer = WaiteTarotRecognizer()
    
    # 检查单张卡牌图片目录
    cards_dir = Path("data/card_dataset/images/rider-waite-tarot")
    if not cards_dir.exists():
        print(f"❌ 单张卡牌图片目录不存在: {cards_dir}")
        print("请确保您已将所有单张卡牌图片放在该目录中")
        return
    
    # 获取图片文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(cards_dir.glob(ext))
        image_files.extend(cards_dir.glob(ext.upper()))
    
    print(f"📁 找到 {len(image_files)} 张图片")
    
    # 显示样本文件名
    if image_files:
        print("\n📸 样本文件:")
        for i, img in enumerate(image_files[:5], 1):
            print(f"   {i}. {img.name}")
        if len(image_files) > 5:
            print(f"   ... 还有 {len(image_files) - 5} 张")
    
    # 统计卡牌类型
    major_count = sum(1 for img in image_files if any(img.stem.startswith(str(i)) for i in range(22)))
    minor_count = len(image_files) - major_count
    
    print(f"\n📊 卡牌统计:")
    print(f"   大阿卡纳: ~{major_count} 张")
    print(f"   小阿卡纳: ~{minor_count} 张")
    print(f"   总计: {len(image_files)} 张")
    
    # 询问是否开始训练
    print(f"\n🎯 训练目标:")
    print(f"   - 提取每张卡牌的多维特征")
    print(f"   - 构建高精度参考数据库") 
    print(f"   - 支持颜色、边缘、纹理、哈希等特征")
    
    confirm = input(f"\n❓ 确定开始训练吗? (y/N): ").strip().lower()
    
    if confirm not in ['y', 'yes', '是']:
        print("❌ 已取消训练")
        return
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    start_time = time.time()
    
    success = recognizer.retrain_from_single_cards()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    if success:
        print(f"\n✅ 训练完成!")
        print(f"⏱️  训练耗时: {training_time:.1f} 秒")
        print(f"📈 成功构建参考数据库: {len(recognizer.reference_db)} 张卡牌")
        
        # 显示训练结果统计
        enhanced_count = sum(1 for ref in recognizer.reference_db.values() if ref.get('is_enhanced', False))
        print(f"🔧 增强特征数据库: {enhanced_count} 张")
        
        # 测试基本功能
        print(f"\n🧪 快速测试...")
        test_image = "data/card_images/spread_0_4821735726296_.pic.jpg"
        if Path(test_image).exists():
            try:
                result = recognizer.analyze_spread_image(test_image)
                if 'error' not in result:
                    print(f"✅ 测试成功: 检测到 {result['total_regions']} 个区域, 识别 {result['recognition_count']} 张卡牌")
                else:
                    print(f"⚠️  测试警告: {result['error']}")
            except Exception as e:
                print(f"⚠️  测试异常: {e}")
        
        print(f"\n🎉 系统已准备就绪!")
        print(f"💡 使用 'python simple_card_test.py' 进行完整测试")
        
    else:
        print(f"\n❌ 训练失败!")
        print(f"💡 请检查:")
        print(f"   - 图片文件是否完整")
        print(f"   - 文件名是否为中文塔罗牌名称")
        print(f"   - 是否有足够的卡牌图片 (至少60张)")

if __name__ == "__main__":
    main() 