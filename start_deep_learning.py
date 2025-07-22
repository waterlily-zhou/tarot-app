#!/usr/bin/env python3
"""
韦特塔罗识别测试和演示系统 - Gemini Vision版
在线识别 + 本地AI解读
"""

from waite_tarot_recognizer import WaiteTarotRecognizer, retrain_database
from integrated_vision_system import IntegratedTarotVisionSystem
import cv2
from pathlib import Path

# Gemini Vision 识别功能
def gemini_card_recognition(image_path: str, api_key: str = None):
    """使用Google Gemini Vision进行塔罗牌识别"""
    try:
        import google.generativeai as genai
        from PIL import Image
        
        if not api_key:
            print("❌ 需要Google API Key")
            print("💡 获取方法: https://makersuite.google.com/app/apikey")
            return None
        
        # 配置API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 加载图片
        img = Image.open(image_path)
        
        # 优化的提示词
        prompt = """
        请识别这张韦特塔罗牌阵图片中的所有卡牌。

        韦特塔罗包含78+2张牌：
        - 大阿卡纳22张：愚人、魔法师、女祭司、皇后、皇帝、教皇、恋人、战车、力量、隐士、命运之轮、正义、倒吊人、死神、节制、恶魔、高塔、星星、月亮、太阳、审判、世界
        - 小阿卡纳56张：权杖/圣杯/宝剑/星币 各14张(一到十、侍从、骑士、皇后、国王)
        - 依恋附属牌2张：扭曲的爱、母子依恋

        要求：
        1. 从图片中心开始、顺时针向外扩张，识别所有的卡牌
        2. 扫描最右、最左、最上、最下边缘，看是否有遗漏的卡牌
        3. 使用准确的中文名称(如"3皇后"、"权杖五"、"圣杯国王")
        4. 判断正位或逆位
        5. 识别牌的坐标位置，以图片中心为原点，向右为x轴正方向，向下为y轴正方向
        6. 只输出识别结果，不要解读

        输出格式(每行一张牌)：
        卡牌名称,正位/逆位,坐标位置(x, y)
        
        例如：
        愚人,正位,(1, 3)
        权杖三,逆位,(2, 3)
        星币皇后,正位, (-3, 3)
        """
        
        print("🌐 使用Google Gemini Vision识别...")
        print("⏳ 分析中...")
        
        # 调用API
        response = model.generate_content([prompt, img])
        
        if response.text:
            print("✅ Gemini识别完成！")
            
            # 解析结果
            lines = response.text.strip().split('\n')
            cards = []
            
            for i, line in enumerate(lines, 1):
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        card_name = parts[0].strip()
                        orientation = parts[1].strip()
                        cards.append({
                            'card_name': card_name,
                            'orientation': orientation,
                            'position': i
                        })
            
            return cards
        else:
            print("❌ Gemini无法识别此图片")
            return None
            
    except ImportError:
        print("❌ 请先安装: pip install google-generativeai pillow")
        return None
    except Exception as e:
        print(f"❌ Gemini识别失败: {e}")
        return None

def gemini_recognition_test():
    """Gemini在线识别测试"""
    print("🔮 Gemini Vision 塔罗牌识别")
    print("="*40)
    
    # 获取API Key
    api_key = input("请输入Google API Key (回车跳过): ").strip()
    if not api_key:
        print("⏭️ 跳过在线识别，使用模拟结果...")
        # 模拟结果用于演示
        mock_cards = [
            {'card_name': '7战车', 'orientation': '正位', 'position': 1},
            {'card_name': '宝剑八', 'orientation': '正位', 'position': 2},
            {'card_name': '星币十', 'orientation': '逆位', 'position': 3},
        ]
        return mock_cards
    
    image_path = "data/card_images/spread_0_4821735726296_.pic.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return None
    
    # 使用Gemini识别
    recognized_cards = gemini_card_recognition(image_path, api_key)
    
    if recognized_cards:
        print(f"\n🎴 识别到 {len(recognized_cards)} 张卡牌:")
        for card in recognized_cards:
            print(f"  {card['position']}. {card['card_name']} ({card['orientation']})")
    
    return recognized_cards

def hybrid_reading_demo():
    """完整演示：Gemini识别 + 本地AI解读"""
    print("🌟 完整塔罗AI系统演示")
    print("🌐 在线识别 + 🤖 本地解读")
    print("="*45)
    
    # 1. Gemini识别
    cards = gemini_recognition_test()
    
    if not cards:
        print("❌ 识别失败，演示结束")
        return
    
    # 2. 本地AI解读
    print(f"\n🤖 开始本地AI解读...")
    try:
        from tarot_ai_system import TarotAISystem
        
        ai_system = TarotAISystem()
        
        # 转换格式
        card_names = [card['card_name'] for card in cards]
        
        # 生成解读
        result = ai_system.generate_reading(
            cards=card_names,
            question="请结合我的个人课程笔记和星盘信息，为这个塔罗牌摊进行专业解读",
            user_id="mel"
        )
        
        if result.get('interpretation'):
            print(f"\n🔮 专业AI解读:")
            print("="*50)
            print(result['interpretation'])
            print("="*50)
            print(f"✅ 解读完成，已保存到本地数据库")
        else:
            print("❌ AI解读失败")
            
    except Exception as e:
        print(f"❌ 本地AI解读失败: {e}")
        print("💡 请确保Ollama和本地LLM正常运行")

def simple_card_recognition_test():
    """本地卡牌识别测试"""
    print("🎴 本地韦特塔罗识别测试")
    print("="*35)
    
    recognizer = WaiteTarotRecognizer()
    image_path = "data/card_images/spread_0_4821735726296_.pic.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return
    
    image = cv2.imread(image_path)
    card_regions = recognizer.detect_card_regions(image)
    print(f"检测到 {len(card_regions)} 个卡牌区域")
    
    recognized_cards = []
    for i, region in enumerate(card_regions, 1):
        card_roi, is_upside_down = recognizer.extract_card_roi(image, region)
        match_result = recognizer.match_card_to_reference(card_roi)
        
        if match_result.get('all_matches'):
            best_match = match_result['all_matches'][0]
            orientation = "逆位" if is_upside_down else "正位"
            print(f"{i:2d}. {best_match['card_name']} ({orientation}) - 置信度: {best_match['similarity']:.3f}")
            recognized_cards.append({
                'card_name': best_match['card_name'],
                'orientation': orientation,
                'confidence': best_match['similarity']
            })
    
    print(f"\n总结: 本地识别 {len(recognized_cards)} 张卡牌")

def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n🎯 韦特塔罗AI系统 v2.0 (Gemini版)")
        print("="*45)
        print("1. 🌐 Gemini识别测试")
        print("2. 🌟 完整演示 (Gemini识别+本地解读)")
        print("3. 🔧 本地识别测试")
        print("4. 🔄 重新训练本地模型")
        print("5. 📊 查看系统状态")
        print("6. ❓ 获取API Key帮助")
        print("7. 🚪 退出")
        
        choice = input("\n请选择 (1-7): ").strip()
        
        if choice == "1":
            gemini_recognition_test()
        elif choice == "2":
            hybrid_reading_demo()
        elif choice == "3":
            simple_card_recognition_test()
        elif choice == "4":
            print("🔄 重新训练本地识别模型...")
            if retrain_database():
                print("✅ 训练完成")
            else:
                print("❌ 训练失败")
        elif choice == "5":
            recognizer = WaiteTarotRecognizer()
            print(f"\n📊 系统状态:")
            print(f"   🔧 本地识别: {len(recognizer.reference_db)} 张卡牌")
            print(f"   🌐 在线识别: Google Gemini Vision")
            print(f"   🤖 本地AI: Ollama + Qwen2.5")
            print(f"   📚 知识库: 课程笔记 + 星盘数据")
            print(f"   🔒 隐私保护: 本地解读，在线仅识别")
        elif choice == "6":
            print(f"\n📖 获取Google API Key:")
            print(f"   1. 访问: https://makersuite.google.com/app/apikey")
            print(f"   2. 登录Google账号")
            print(f"   3. 点击'Create API Key'")
            print(f"   4. 复制API Key即可使用")
            print(f"   💰 免费额度: 每天1500次调用")
        elif choice == "7":
            print("👋 感谢使用韦特塔罗AI系统")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    interactive_menu() 