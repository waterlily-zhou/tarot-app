#!/usr/bin/env python3
"""
韦特塔罗识别测试和演示系统
"""

from waite_tarot_recognizer import WaiteTarotRecognizer, retrain_database
from integrated_vision_system import IntegratedTarotVisionSystem
import cv2
from pathlib import Path

def simple_card_recognition_test():
    """简单的卡牌识别测试"""
    print("🎴 韦特塔罗识别测试")
    print("="*40)
    
    recognizer = WaiteTarotRecognizer()
    image_path = "data/card_images/spread_0_4821735726296_.pic.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return
    
    image = cv2.imread(image_path)
    
    # 检测卡牌区域
    card_regions = recognizer.detect_card_regions(image)
    print(f"检测到 {len(card_regions)} 个卡牌区域")
    print()
    
    # 识别每张卡牌
    recognized_cards = []
    
    for i, region in enumerate(card_regions, 1):
        # 提取卡牌ROI
        card_roi, is_upside_down = recognizer.extract_card_roi(image, region)
        
        # 匹配卡牌
        match_result = recognizer.match_card_to_reference(card_roi)
        
        if match_result.get('all_matches'):
            best_match = match_result['all_matches'][0]
            orientation = "逆位" if is_upside_down else "正位"
            
            # 显示更详细的信息
            print(f"{i:2d}. {best_match['card_name']} ({orientation}) - 置信度: {best_match['similarity']:.3f}")
            print(f"     面积: {region['area']:.0f}, 位置: {region['center']}")
            
            recognized_cards.append({
                'card_name': best_match['card_name'],
                'orientation': orientation,
                'confidence': best_match['similarity'],
                'area': region['area'],
                'position': region['center']
            })
        else:
            print(f"{i:2d}. 未识别 - 面积: {region['area']:.0f}, 位置: {region['center']}")
    
    print()
    print(f"总结: 检测 {len(card_regions)} 个区域, 识别 {len(recognized_cards)} 张卡牌")
    
    # 按置信度排序显示
    if recognized_cards:
        print("\n按置信度排序:")
        sorted_cards = sorted(recognized_cards, key=lambda x: x['confidence'], reverse=True)
        for i, card in enumerate(sorted_cards, 1):
            print(f"{i:2d}. {card['card_name']} ({card['orientation']}) - {card['confidence']:.3f}")

def full_system_demo():
    """完整系统演示"""
    print("🌟 集成塔罗AI系统演示")
    print("="*40)
    
    try:
        system = IntegratedTarotVisionSystem()
        
        # 分析图片并生成AI解读
        image_path = "data/card_images/spread_0_4821735726296_.pic.jpg"
        result = system.analyze_tarot_spread_image(
            image_path, 
            user_id="demo_user",
            question="请为这个塔罗牌摊进行详细解读"
        )
        
        if result['success']:
            print("\n✅ 完整分析成功！")
            print(f"识别卡牌: {result['summary']['cards_identified']} 张")
            print(f"平均置信度: {result['summary']['average_confidence']:.1%}")
            
            # 显示AI解读
            if result.get('ai_reading', {}).get('interpretation'):
                print(f"\n🔮 AI解读:")
                print(result['ai_reading']['interpretation'][:300] + "...")
        else:
            print(f"❌ 分析失败: {result['error']}")
            
    except Exception as e:
        print(f"💥 演示失败: {e}")

def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n🎯 韦特塔罗AI系统")
        print("="*30)
        print("1. 简单卡牌识别测试")
        print("2. 完整系统演示(识别+AI解读)")
        print("3. 重新训练识别数据库")
        print("4. 查看系统状态")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == "1":
            simple_card_recognition_test()
        elif choice == "2":
            full_system_demo()
        elif choice == "3":
            print("🔄 重新训练识别数据库...")
            if retrain_database():
                print("✅ 训练完成")
            else:
                print("❌ 训练失败")
        elif choice == "4":
            recognizer = WaiteTarotRecognizer()
            print(f"\n📊 系统状态:")
            print(f"   参考数据库: {len(recognizer.reference_db)} 张卡牌")
            print(f"   支持功能: 卡牌识别、正逆位判断、位置检测、AI解读")
        elif choice == "5":
            print("👋 退出系统")
            break
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    interactive_menu() 