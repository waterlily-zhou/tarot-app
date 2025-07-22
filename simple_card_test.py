#!/usr/bin/env python3
"""
韦特塔罗识别测试和演示系统 - Gemini Vision版
在线识别 + 本地AI解读
"""

import cv2
from pathlib import Path
import os

# 尝试导入其他模块，如果失败则跳过
try:
    from waite_tarot_recognizer import WaiteTarotRecognizer, retrain_database
    LOCAL_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 本地识别模块导入失败: {e}")
    LOCAL_RECOGNITION_AVAILABLE = False

try:
    from integrated_vision_system import IntegratedTarotVisionSystem
    INTEGRATED_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 集成系统模块导入失败: {e}")
    INTEGRATED_SYSTEM_AVAILABLE = False



def load_env_file():
    """加载.env.local文件"""
    env_file = Path('.env.local')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def gemini_card_recognition(image_path: str, api_key: str = None):
    """Gemini Vision塔罗牌识别函数"""
    try:
        import google.generativeai as genai
        from PIL import Image
        
        if not api_key:
            load_env_file()
            api_key = os.getenv('GEMINIAPI')
        
        if not api_key:
            print("❌ 需要Google API Key")
            print("💡 请在.env.local文件中设置: GEMINIAPI=你的API密钥")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 使用原图
        img = Image.open(image_path)
        
        prompt = """
        请仔细扫描这张塔罗牌阵图片，识别所有可见的塔罗牌。

        🔍 扫描策略：
        1. 系统性扫描整个图片，包括所有边缘区域
        2. 从图片中心开始、顺时针向外扩张
        3. 特别注意最右、最左、最上、最下边缘
        4. 识别所有可见的卡牌

        ⚠️ 重要提醒：不要遗漏右侧角落和边缘的卡牌

        韦特塔罗包含78+2张牌：
        - 大阿卡纳22张：愚人、魔法师、女祭司、皇后、皇帝、教皇、恋人、战车、力量、隐士、命运之轮、正义、倒吊人、死神、节制、恶魔、高塔、星星、月亮、太阳、审判、世界
        - 小阿卡纳56张：权杖/圣杯/宝剑/星币 各14张(一到十、侍从、骑士、皇后、国王)
        - 依恋附属牌2张：22依恋、23母子

        识别要求：
        1. 使用准确的中文名称(如"3皇后"、"权杖五"、"圣杯国王"、"1魔法师")
        2. 判断正位或逆位
        3. 标注坐标位置，以图片中心为原点，向右为x轴正方向，向下为y轴正方向
        4. 只输出识别结果，不要解读

        输出格式(每行一张牌)：
        卡牌名称,正位/逆位,坐标位置(x, y)

        例如：
        愚人,正位,(1, 3)
        权杖三,逆位,(2, 3)
        星币皇后,正位,(3, 1)
        1魔法师,逆位,(-1, 2)
        22依恋,正位,(0, 1)
        23母子,逆位,(-1, 2)

        请开始识别所有可见的塔罗牌：
        """
        
        print("🌐 使用Gemini Vision识别...")
        print("⏳ 分析中...")
        
        response = model.generate_content([prompt, img])
        
        if response.text:
            print("✅ Gemini识别完成！")
            print("\n📋 Gemini识别结果:")
            print("-" * 50)
            print(response.text)
            print("-" * 50)
            
            lines = response.text.strip().split('\n')
            cards = []
            
            for i, line in enumerate(lines, 1):
                if ',' in line and not line.strip().startswith('卡牌名称'):
                    # 智能分割：先找到坐标部分（包含括号的部分）
                    if '(' in line and ')' in line:
                        # 找到坐标的开始和结束位置
                        start_coord = line.find('(')
                        end_coord = line.find(')', start_coord) + 1
                        
                        # 分离坐标前的部分和坐标
                        before_coord = line[:start_coord].rstrip(',').strip()
                        coord_part = line[start_coord:end_coord].strip()
                        
                        # 分割卡牌名称和方位
                        before_parts = before_coord.split(',')
                        if len(before_parts) >= 2:
                            card_name = before_parts[0].strip()
                            orientation = before_parts[1].strip()
                            position = coord_part
                            
                            cards.append({
                                'card_name': card_name,
                                'orientation': orientation,
                                'position': position,
                                'order': len(cards) + 1
                            })
                    else:
                        # 没有坐标的情况，按原逻辑处理
                        parts = line.split(',')
                        if len(parts) >= 2:
                            card_name = parts[0].strip()
                            orientation = parts[1].strip()
                            cards.append({
                                'card_name': card_name,
                                'orientation': orientation,
                                'position': "未知位置",
                                'order': len(cards) + 1
                            })
            
            # 简单的结果统计，不预设期望
            if len(cards) > 0:
                print(f"\n✅ 成功识别到 {len(cards)} 张卡牌")
            else:
                print(f"\n⚠️ 未识别到任何卡牌")
            
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
    
    image_path = "data/card_images/spread_0_4821735726296_.pic.jpg"
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return None
    
    # 使用Gemini单图识别
    recognized_cards = gemini_card_recognition(image_path)
    
    if recognized_cards:
        print(f"\n🎴 解析后的卡牌列表 ({len(recognized_cards)} 张):")
        for card in recognized_cards:
            print(f"  {card['order']}. {card['card_name']} ({card['orientation']}) - 位置: {card['position']}")
    
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
    if INTEGRATED_SYSTEM_AVAILABLE:
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
    else:
        print("❌ 本地AI系统不可用，请安装缺少的依赖")

def simple_card_recognition_test():
    """本地卡牌识别测试（旧版本）"""
    if not LOCAL_RECOGNITION_AVAILABLE:
        print("❌ 本地识别系统不可用")
        return
        
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

def retrain_with_attachments():
    """重新训练，包含依恋附属牌"""
    if not LOCAL_RECOGNITION_AVAILABLE:
        print("❌ 本地识别系统不可用")
        return False
        
    print("🔄 重新训练本地识别模型（包含依恋附属牌）...")
    
    # 修改依恋牌名称映射
    attachment_cards = ["22依恋", "23母子"]
    print(f"📌 将包含以下依恋附属牌: {', '.join(attachment_cards)}")
    
    if retrain_database():
        print("✅ 训练完成（包含80张卡牌）")
        return True
    else:
        print("❌ 训练失败")
        return False

def check_api_key_status():
    """检查API Key状态"""
    load_env_file()
    api_key = os.getenv('GEMINIAPI')
    
    if api_key:
        masked_key = api_key[:8] + "*" * (len(api_key) - 16) + api_key[-8:] if len(api_key) > 16 else "****"
        print(f"✅ API Key已配置: {masked_key}")
        return True
    else:
        print("❌ 未找到API Key")
        print("💡 请在.env.local文件中设置: GEMINIAPI=你的API密钥")
        return False

def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n🎯 韦特塔罗AI系统 v2.0 (Gemini版)")
        print("="*45)
        print("1. 🌐 Gemini在线识别测试 (推荐)")
        print("2. 🌟 完整演示 (Gemini识别+本地解读)")
        print("3. 🔧 本地识别测试 (准确率低)")
        print("4. 🔄 重新训练本地模型(含依恋牌)")
        print("5. 🔑 检查API Key状态")
        print("6. 📊 查看系统状态")
        print("7. ❓ 获取API Key帮助")
        print("8. 🚪 退出")
        
        choice = input("\n请选择 (1-8): ").strip()
        
        if choice == "1":
            gemini_recognition_test()
        elif choice == "2":
            hybrid_reading_demo()
        elif choice == "3":
            simple_card_recognition_test()
        elif choice == "4":
            retrain_with_attachments()
        elif choice == "5":
            check_api_key_status()
        elif choice == "6":
            print(f"\n📊 系统状态:")
            print(f"   🌐 在线识别: Google Gemini Vision")
            print(f"   🔧 本地识别: {'✅ 可用' if LOCAL_RECOGNITION_AVAILABLE else '❌ 不可用'}")
            print(f"   🤖 集成系统: {'✅ 可用' if INTEGRATED_SYSTEM_AVAILABLE else '❌ 不可用'}")
            if LOCAL_RECOGNITION_AVAILABLE:
                recognizer = WaiteTarotRecognizer()
                print(f"   📚 本地数据库: {len(recognizer.reference_db)} 张卡牌")
            print(f"   🔒 隐私保护: 本地解读，在线仅识别")
        elif choice == "7":
            print(f"\n📖 获取Google API Key:")
            print(f"   1. 访问: https://makersuite.google.com/app/apikey")
            print(f"   2. 登录Google账号")
            print(f"   3. 点击'Create API Key'")
            print(f"   4. 复制API Key")
            print(f"   5. 在项目根目录创建.env.local文件")
            print(f"   6. 在文件中添加: GEMINIAPI=你的API密钥")
            print(f"   💰 免费额度: 每天1500次调用")
        elif choice == "8":
            print("👋 感谢使用韦特塔罗AI系统")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    interactive_menu() 