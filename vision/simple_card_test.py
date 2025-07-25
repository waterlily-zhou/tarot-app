#!/usr/bin/env python3
"""
韦特塔罗识别测试和演示系统 - Gemini Vision版
在线识别 + 本地AI解读
"""

import cv2
from pathlib import Path
import os

# 导入图片预处理模块
try:
    from image_preprocessor import ImagePreprocessor
    PREPROCESSOR_AVAILABLE = True
    print("✅ 图片预处理模块已加载")
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    print("⚠️ 图片预处理模块不可用")





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

def gemini_card_recognition(image_path: str, api_key: str = None, silent: bool = False):
    """Gemini Vision塔罗牌识别函数"""
    try:
        import google.generativeai as genai
        from PIL import Image
        
        if not api_key:
            load_env_file()
            api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            print("❌ 需要Google API Key")
            print("💡 请在.env.local文件中设置: GOOGLE_API_KEY=你的API密钥")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 图片预处理：添加安全边距
        processed_image_path = image_path
        preprocessor = None
        
        if PREPROCESSOR_AVAILABLE:
            preprocessor = ImagePreprocessor()
            # 增大边距，尝试捕获更多边缘卡牌
            processed_image_path = preprocessor.add_safe_margin(image_path, margin_size=30)
            if not silent:
                print("🖼️ 使用预处理后的图片进行识别（30px边距）")
        else:
            if not silent:
                print("⚠️ 跳过图片预处理")
        
        # 使用预处理后的图片
        img = Image.open(processed_image_path)
        
        # 获取原图尺寸用于坐标转换（统一使用原图中心坐标系）
        original_img = Image.open(image_path)
        original_width, original_height = original_img.size
        
        prompt = """
        请仔细扫描这张塔罗牌阵图片，识别所有可见的塔罗牌。

        🔍 完整扫描策略（必须按顺序执行）：
        1. **角落优先**：左上角→右上角→右下角→左下角（即使只露出一角也要识别）
        2. **边缘完整**：上边缘→右边缘→下边缘→左边缘（包括半张卡牌）  
        3. **中心区域**：从中心向外螺旋扫描
        4. **二次确认**：重新检查是否有遗漏的边缘卡牌
        5. **最终验证**：确保图片每个区域都被检查过

        🚨 关键要求：
        - 图片边缘的卡牌绝对不能遗漏！
        - 即使卡牌被裁切、只露出一部分也必须识别
        - 特别注意图片最边缘和角落区域
        - 宁可多识别也不要遗漏
        - 扫描范围必须覆盖整个图片的100%区域
        

        韦特塔罗标准名称（必须严格使用）：
        
        📋 大阿卡纳(22张)：
        愚人、魔法师、女祭司、皇后、皇帝、教皇、恋人、战车、力量、隐士、命运之轮、正义、倒吊人、死神、节制、恶魔、高塔、星星、月亮、太阳、审判、世界
        
        📋 小阿卡纳数字牌(40张)：
        权杖一、权杖二、权杖三、权杖四、权杖五、权杖六、权杖七、权杖八、权杖九、权杖十
        圣杯一、圣杯二、圣杯三、圣杯四、圣杯五、圣杯六、圣杯七、圣杯八、圣杯九、圣杯十
        宝剑一、宝剑二、宝剑三、宝剑四、宝剑五、宝剑六、宝剑七、宝剑八、宝剑九、宝剑十
        星币一、星币二、星币三、星币四、星币五、星币六、星币七、星币八、星币九、星币十
        
        📋 小阿卡纳宫廷牌(16张)：
        权杖侍从、权杖骑士、权杖皇后、权杖国王
        圣杯侍从、圣杯骑士、圣杯皇后、圣杯国王
        宝剑侍从、宝剑骑士、宝剑皇后、宝剑国王
        星币侍从、星币骑士、星币皇后、星币国王
        
        📋 附属牌(2张)：
        22依恋、23母子
        
        ⚠️ 重要要求：
        1. 请多留意牌面的罗马数字作为参考，通常在正位牌面上方、逆位牌面下方
        2. 必须严格使用上述标准名称，不得使用变体名称
        3. 错误示例：星币女王❌ → 正确：星币皇后✅
        4. 错误示例：十号星币❌ → 正确：星币十✅
        5. 错误示例：圣杯国王❌ → 正确：圣杯国王✅（这个是正确的）
        6. 判断正位或逆位
        7. 标注坐标位置
        8. 只输出识别结果，不要解读

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
        
        if not silent:
            print("🌐 使用Gemini Vision识别...")
            print("⏳ 分析中...")
        
        response = model.generate_content([prompt, img])
        
        if response.text:
            if not silent:
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
                            
                            # 转换坐标为原图中心坐标系
                            if PREPROCESSOR_AVAILABLE:
                                x, y = preprocessor.parse_coordinate_string(coord_part)
                                if x is not None and y is not None:
                                    # 预处理图片坐标 → 原图坐标 → 原图中心坐标
                                    original_x = x - 30  # 减去左边距
                                    original_y = y - 30  # 减去上边距
                                    center_x, center_y = preprocessor.convert_to_center_coordinates(
                                        original_x, original_y, original_width, original_height
                                    )
                                    position = f"({center_x}, {center_y})"
                                else:
                                    position = coord_part
                            else:
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
            if not silent:
                if len(cards) > 0:
                    print(f"\n✅ 成功识别到 {len(cards)} 张卡牌")
                else:
                    print(f"\n⚠️ 未识别到任何卡牌")
            
                    # 清理预处理临时文件
        if PREPROCESSOR_AVAILABLE and preprocessor:
            preprocessor.cleanup_temp_files()
            
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




def gemini_precise_recognition(image_path: str):
    """精确识别单张或少量卡牌"""
    try:
        import google.generativeai as genai
        from PIL import Image
        
        load_env_file()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ 需要Google API Key")
            print("💡 请在.env.local文件中设置: GOOGLE_API_KEY=你的API密钥")
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        img = Image.open(image_path)
        
        # 精确识别提示词，强调数字和细节
        prompt = """
        请精确识别这张图片中的塔罗牌。这可能是裁剪后的图片，请特别注意：

        🔢 **数字牌识别重点**：
        - 仔细数星币、圣杯、权杖、宝剑的具体数量
        - 不要猜测，要根据实际看到的符号数量
        - 星币7: 7个星币符号，通常一个人看着星币树
        - 星币10: 10个星币符号，通常有家庭场景
        - 其他数字牌也请准确计数

        🎯 **识别标准**：
        1. 首先数符号数量（最重要！）
        2. 观察人物和场景
        3. 确认正逆位
        4. 如果看不清楚，请说"无法确定"

        📝 **输出格式**：
        卡牌名称,正位/逆位,(x坐标,y坐标)

        请开始识别：
        """
        
        response = model.generate_content([prompt, img])
        response_text = response.text.strip()
        
        # 解析结果
        cards = []
        if response_text and "无法" not in response_text:
            lines = [line.strip() for line in response_text.split('\n') if line.strip()]
            
            for i, line in enumerate(lines):
                if ',' in line:
                    # 检查是否包含坐标
                    if '(' in line and ')' in line:
                        # 有坐标的情况
                        start_coord = line.find('(')
                        end_coord = line.find(')', start_coord) + 1
                        
                        before_coord = line[:start_coord].rstrip(',').strip()
                        coord_part = line[start_coord:end_coord].strip()
                        
                        before_parts = before_coord.split(',')
                        if len(before_parts) >= 2:
                            card_name = before_parts[0].strip()
                            orientation = before_parts[1].strip()
                            
                            # 保持裁剪图片的左上角坐标（在gemini_edge_detection中统一转换）
                            position = coord_part
                        else:
                            # 格式不正确，跳过这行
                            continue
                    
                    else:
                        # 没有坐标的情况
                        parts = line.split(',')
                        if len(parts) >= 2:
                            card_name = parts[0].strip()
                            orientation = parts[1].strip()
                            position = "(0, 0)"  # 无坐标时使用原点
                    
                    cards.append({
                        'card_name': card_name,
                        'orientation': orientation,
                        'position': position,
                        'order': i + 1
                    })
        
        return cards if cards else None
        
    except Exception as e:
        print(f"❌ Gemini精确识别失败: {e}")
        return None

def gemini_edge_detection(image_path: str):
    """完整边缘检测分析"""
    if not PREPROCESSOR_AVAILABLE:
        print("❌ 图片预处理模块不可用")
        return None
        
    preprocessor = ImagePreprocessor()
    
    print("🔍 分析中...")
    
    # 1. 完整图片识别（静默）
    full_cards = gemini_card_recognition(image_path, silent=True)
    
    if not full_cards:
        print("❌ 完整图片识别失败")
        return None
    
    full_card_names = [card['card_name'] for card in full_cards]
    
    # 2. 右侧边缘检测（直接使用原图）
    # 获取原图尺寸用于坐标转换
    from PIL import Image
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    
    # 直接对原图进行右侧裁剪
    right_crop = preprocessor.crop_right_edge(image_path, crop_percentage=0.2, silent=True)
    right_cards = gemini_precise_recognition(right_crop)
    
    # 3. 整合结果
    final_cards = []
    new_cards_found = []
    
    # 添加完整识别的卡牌
    for card in full_cards:
        final_cards.append({
            'card_name': card['card_name'],
            'orientation': card['orientation'], 
            'position': card['position'],
            'source': '完整识别'
        })
    
    # 添加新发现的卡牌
    if right_cards:
        right_card_names = [card['card_name'] for card in right_cards]
        new_cards = [name for name in right_card_names if name not in full_card_names]
        
        # 计算右侧裁剪区域的坐标转换
        # 裁剪起始位置：原图宽度的80%位置开始
        crop_start_x = int(original_width * 0.8)
        
        for card in right_cards:
            if card['card_name'] in new_cards:
                # 获取裁剪图片中的坐标（这是基于裁剪图片左上角的坐标）
                crop_position = card.get('position', '(0, 0)')
                crop_x, crop_y = preprocessor.parse_coordinate_string(crop_position)
                
                if crop_x is not None and crop_y is not None:
                    # 转换为原图左上角坐标系
                    original_x = crop_start_x + crop_x
                    original_y = crop_y  # y坐标不变
                    
                    # 转换为原图中心坐标系
                    center_x, center_y = preprocessor.convert_to_center_coordinates(
                        original_x, original_y, original_width, original_height
                    )
                    converted_position = f"({center_x}, {center_y})"
                else:
                    converted_position = "(右侧区域)"
                
                final_cards.append({
                    'card_name': card['card_name'],
                    'orientation': card['orientation'],
                    'position': converted_position,
                    'source': '边缘补充'
                })
                new_cards_found.append(card['card_name'])
    
    # 4. 输出最终结果
    print(f"\n🎴 完整识别结果 ({len(final_cards)} 张卡牌)")
    print("📐 坐标系统: 中心坐标系，原点(0,0)在图片中心，单位像素")
    print("=" * 50)
    
    for i, card in enumerate(final_cards, 1):
        source_icon = "✅" if card['source'] == '完整识别' else "🆕"
        print(f"{i:2d}. {card['card_name']} ({card['orientation']}) - {card['position']} {source_icon}")
    
    print("=" * 50)
    
    if new_cards_found:
        print(f"🆕 边缘检测补充发现: {len(new_cards_found)} 张")
        for card_name in new_cards_found:
            print(f"   • {card_name}")
    else:
        print("✅ 完整识别已覆盖所有卡牌")
    
    print(f"\n📁 右侧裁剪图片: {right_crop}")
    input("按回车键清理临时文件...")
    
    # 清理临时文件
    preprocessor.cleanup_temp_files()
    
    return {
        'final_cards': final_cards,
        'new_cards_found': new_cards_found,
        'total_count': len(final_cards)
    }

def gemini_recognition_test():
    """Gemini在线识别测试"""
    print("🔮 Gemini Vision 塔罗牌识别")
    print("="*40)
    
    # 让用户选择图片
    print("请选择要识别的图片：")
    print("1. 原始测试图片 (spread_0_4821735726296_.pic.jpg)")
    print("2. 自定义图片路径")
    
    while True:
        img_choice = input("请选择图片 (1-2): ").strip()
        if img_choice in ['1', '2']:
            break
        print("❌ 请输入1或2")
    
    if img_choice == '1':
        image_path = "data/card_images/spread_0_4821735726296_.pic.jpg"
    else:
        image_path = input("请输入图片路径: ").strip()
    
    # 让用户选择识别策略
    print("\n请选择识别策略：")
    print("1. 单图识别")
    print("2. 单图识别+边缘遗漏分析 (右侧20%裁剪)")
    
    while True:
        choice = input("请选择 (1-2): ").strip()
        if choice in ['1', '2']:
            break
        print("❌ 请输入1或2")
    
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return None
    
    # 根据选择使用不同策略
    if choice == '1':
        recognized_cards = gemini_card_recognition(image_path)
    elif choice == '2':
        recognized_cards = gemini_edge_detection(image_path)
        return recognized_cards  # 边缘检测直接返回
    
    if recognized_cards:
        print(f"\n🎴 识别结果 ({len(recognized_cards)} 张卡牌)")
        print("📐 坐标系统: 中心坐标系，原点(0,0)在图片中心，单位像素")
        print("=" * 50)
        for card in recognized_cards:
            print(f"{card['order']:2d}. {card['card_name']} ({card['orientation']}) - {card['position']}")
        print("=" * 50)
    
    return recognized_cards



def interactive_menu():
    """交互式菜单"""
    while True:
        print("\n🎯 塔罗牌识别系统")
        print("="*35)
        print("1. 🔮 开始识别")
        print("2. 🚪 退出")
        
        choice = input("\n请选择 (1-2): ").strip()
        
        if choice == "1":
            gemini_recognition_test()
        elif choice == "2":
            print("👋 感谢使用塔罗牌识别系统")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    interactive_menu() 