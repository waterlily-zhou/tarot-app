#!/usr/bin/env python3
"""
集成视觉识别的塔罗AI系统
结合韦特塔罗识别和现有的AI解读系统
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
from waite_tarot_recognizer import WaiteTarotRecognizer
from tarot_ai_system import TarotAISystem

class IntegratedTarotVisionSystem:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # 初始化视觉识别系统
        print("🔮 初始化韦特塔罗识别系统...")
        self.vision_recognizer = WaiteTarotRecognizer(data_dir)
        
        # 初始化AI解读系统
        print("🤖 初始化AI解读系统...")
        self.ai_system = TarotAISystem()
        
        print("✅ 集成系统初始化完成！")
    
    def analyze_tarot_spread_image(self, image_path: str, user_id: str = "user", 
                                 question: str = None, spread_type: str = "综合解读") -> Dict:
        """完整的塔罗牌摊图片分析和AI解读"""
        
        print(f"\n🎴 开始分析塔罗牌摊: {Path(image_path).name}")
        print("=" * 60)
        
        # 第一步：视觉识别
        print("👁️  第一步：视觉识别")
        vision_results = self.vision_recognizer.analyze_spread_image(image_path)
        
        if 'error' in vision_results:
            return {
                'success': False,
                'error': vision_results['error'],
                'stage': 'vision_recognition'
            }
        
        # 提取识别到的卡牌
        recognized_cards = vision_results.get('recognized_cards', [])
        
        if not recognized_cards:
            return {
                'success': False,
                'error': '未识别到任何卡牌',
                'vision_results': vision_results,
                'stage': 'vision_recognition'
            }
        
        print(f"\n📊 视觉识别结果:")
        print(f"   检测区域: {vision_results['total_regions']}")
        print(f"   成功识别: {vision_results['recognition_count']}")
        print(f"   成功率: {vision_results['success_rate']:.1%}")
        
        # 第二步：构建卡牌列表和布局信息
        print(f"\n🗂️  第二步：整理卡牌信息")
        
        # 按位置排序卡牌（从左到右，从上到下）
        sorted_cards = sorted(recognized_cards, key=lambda x: (x['position'][1], x['position'][0]))
        
        # 提取卡牌名称列表
        card_names = []
        layout_info = []
        
        for i, card in enumerate(sorted_cards, 1):
            card_name = card['card_name']
            if card['is_reversed']:
                card_name += " (逆位)"
                
            card_names.append(card['card_name'])  # AI系统用原始名称
            layout_info.append({
                'position': i,
                'card': card_name,
                'confidence': card['confidence'],
                'location': card['position'],
                'is_reversed': card['is_reversed']
            })
            
            orientation = "逆位" if card['is_reversed'] else "正位"
            print(f"   位置{i}: {card['card_name']} ({orientation}) - 置信度: {card['confidence']:.3f}")
        
        # 第三步：AI解读
        print(f"\n🧠 第三步：AI解读")
        
        # 构建解读问题
        if not question:
            question = f"请为这个{len(card_names)}张卡牌的塔罗牌摊进行详细解读"
        
        # 添加正逆位信息到问题中
        detailed_question = question + "\n\n卡牌布局详情：\n"
        for info in layout_info:
            detailed_question += f"位置{info['position']}: {info['card']}\n"
        
        print(f"   🔍 解读问题: {question}")
        print(f"   🎯 牌阵类型: {spread_type}")
        
        try:
            # 调用AI系统进行解读
            ai_reading = self.ai_system.generate_reading(
                cards=card_names,
                question=detailed_question,
                user_id=user_id,
                spread_type=spread_type
            )
            
            print(f"   ✅ AI解读完成")
            
        except Exception as e:
            print(f"   ❌ AI解读失败: {e}")
            return {
                'success': False,
                'error': f'AI解读失败: {str(e)}',
                'vision_results': vision_results,
                'layout_info': layout_info,
                'stage': 'ai_reading'
            }
        
        # 第四步：整合结果
        print(f"\n📋 第四步：整合结果")
        
        integrated_result = {
            'success': True,
            'image_path': image_path,
            'analysis_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'user_id': user_id,
            'question': question,
            'spread_type': spread_type,
            
            # 视觉识别结果
            'vision_analysis': {
                'total_regions_detected': vision_results['total_regions'],
                'cards_recognized': vision_results['recognition_count'],
                'success_rate': vision_results['success_rate'],
                'recognition_details': vision_results['recognized_cards']
            },
            
            # 卡牌布局信息
            'card_layout': {
                'total_cards': len(card_names),
                'card_sequence': card_names,
                'layout_details': layout_info,
                'has_reversed_cards': any(info['is_reversed'] for info in layout_info)
            },
            
            # AI解读结果
            'ai_reading': ai_reading,
            
            # 综合信息
            'summary': {
                'cards_identified': len(card_names),
                'reading_generated': 'reading_id' in ai_reading,
                'confidence_scores': [info['confidence'] for info in layout_info],
                'average_confidence': np.mean([info['confidence'] for info in layout_info])
            }
        }
        
        print(f"   ✅ 完整分析成功")
        
        return integrated_result
    
    def save_analysis_result(self, result: Dict, save_dir: str = None):
        """保存分析结果"""
        if not save_dir:
            save_dir = self.data_dir / "vision_analysis_results"
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        timestamp = int(time.time())
        result_file = save_dir / f"tarot_analysis_{timestamp}.json"
        
        # 保存JSON结果
        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 分析结果已保存: {result_file}")
        
        return result_file
    
    def generate_analysis_report(self, result: Dict) -> str:
        """生成人性化的分析报告"""
        if not result['success']:
            return f"❌ 分析失败: {result['error']}"
        
        report = f"""
🎴 塔罗牌摊AI分析报告
{'='*50}

📷 图片信息:
   文件: {Path(result['image_path']).name}
   分析时间: {result['analysis_time']}
   用户: {result['user_id']}

❓ 解读问题:
   {result['question']}

👁️ 视觉识别结果:
   检测区域: {result['vision_analysis']['total_regions_detected']}
   成功识别: {result['vision_analysis']['cards_recognized']} 张卡牌
   识别成功率: {result['vision_analysis']['success_rate']:.1%}
   平均置信度: {result['summary']['average_confidence']:.1%}

🎯 卡牌布局:
"""
        
        for info in result['card_layout']['layout_details']:
            orientation = " (逆位)" if info['is_reversed'] else " (正位)"
            report += f"   位置 {info['position']}: {info['card'][:info['card'].find(' (') if ' (' in info['card'] else len(info['card'])]}{orientation}\n"
        
        if result['ai_reading'] and 'interpretation' in result['ai_reading']:
            report += f"""
🔮 AI解读内容:
{result['ai_reading']['interpretation']}

📊 解读信息:
   解读ID: {result['ai_reading'].get('reading_id', 'Unknown')}
   牌阵类型: {result['spread_type']}
   生成时间: {result['ai_reading'].get('timestamp', 'Unknown')}
"""
        
        report += f"""
📈 分析统计:
   识别卡牌数: {result['summary']['cards_identified']}
   包含逆位: {'是' if result['card_layout']['has_reversed_cards'] else '否'}
   AI解读成功: {'是' if result['summary']['reading_generated'] else '否'}

---
🤖 由集成塔罗AI系统自动生成
"""
        
        return report
    
    def batch_analyze_images(self, image_directory: str, user_id: str = "batch_user") -> List[Dict]:
        """批量分析图片目录中的所有塔罗牌摊"""
        image_dir = Path(image_directory)
        
        if not image_dir.exists():
            print(f"❌ 目录不存在: {image_directory}")
            return []
        
        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_dir.glob(ext))
        
        if not image_files:
            print(f"📷 在 {image_directory} 中未找到图片文件")
            return []
        
        print(f"🗂️  批量分析 {len(image_files)} 张图片...")
        
        results = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n📷 [{i}/{len(image_files)}] 处理: {image_file.name}")
            
            try:
                result = self.analyze_tarot_spread_image(
                    str(image_file), 
                    user_id=f"{user_id}_{i}",
                    question=f"第{i}张图片的塔罗解读"
                )
                
                results.append(result)
                
                if result['success']:
                    print(f"   ✅ 成功 - 识别{result['summary']['cards_identified']}张卡牌")
                else:
                    print(f"   ❌ 失败 - {result['error']}")
                    
            except Exception as e:
                print(f"   💥 异常 - {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_path': str(image_file)
                })
        
        # 生成批量分析摘要
        successful = len([r for r in results if r['success']])
        print(f"\n📊 批量分析完成:")
        print(f"   总计: {len(results)} 张图片")
        print(f"   成功: {successful} 张")
        print(f"   失败: {len(results) - successful} 张")
        print(f"   成功率: {successful/len(results)*100:.1f}%")
        
        return results
    
    def interactive_demo(self):
        """交互式演示"""
        print("\n🎯 集成塔罗AI系统 - 交互演示")
        print("=" * 50)
        
        while True:
            print("\n选择功能:")
            print("1. 分析单张塔罗牌摊图片")
            print("2. 批量分析图片目录")
            print("3. 查看可用图片")
            print("4. 退出")
            
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == "1":
                self._demo_single_image()
            elif choice == "2":
                self._demo_batch_analysis()
            elif choice == "3":
                self._show_available_images()
            elif choice == "4":
                print("👋 退出演示")
                break
            else:
                print("❌ 无效选择")
    
    def _demo_single_image(self):
        """单张图片演示"""
        card_images_dir = Path("data/card_images")
        
        if not card_images_dir.exists():
            print("❌ 图片目录不存在")
            return
        
        image_files = list(card_images_dir.glob("*.jpg")) + \
                     list(card_images_dir.glob("*.png"))
        
        if not image_files:
            print("📷 未找到图片文件")
            return
        
        print("\n可用图片:")
        for i, img_file in enumerate(image_files, 1):
            print(f"{i}. {img_file.name}")
        
        try:
            choice = int(input(f"\n选择图片 (1-{len(image_files)}): ")) - 1
            if 0 <= choice < len(image_files):
                image_path = image_files[choice]
                
                question = input("解读问题 (回车使用默认): ").strip()
                if not question:
                    question = "请为这个塔罗牌摊进行详细解读"
                
                user_id = input("用户ID (回车使用默认): ").strip()
                if not user_id:
                    user_id = "demo_user"
                
                # 执行分析
                result = self.analyze_tarot_spread_image(str(image_path), user_id, question)
                
                # 生成报告
                report = self.generate_analysis_report(result)
                print(report)
                
                # 保存结果
                self.save_analysis_result(result)
                
            else:
                print("❌ 无效选择")
        except ValueError:
            print("❌ 请输入数字")
    
    def _demo_batch_analysis(self):
        """批量分析演示"""
        directory = input("输入图片目录路径 (回车使用 data/card_images): ").strip()
        if not directory:
            directory = "data/card_images"
        
        user_id = input("批量用户ID前缀 (回车使用 batch): ").strip()
        if not user_id:
            user_id = "batch"
        
        results = self.batch_analyze_images(directory, user_id)
        
        if results:
            # 保存批量结果
            batch_summary = {
                'batch_analysis': True,
                'directory': directory,
                'total_images': len(results),
                'successful_analyses': len([r for r in results if r['success']]),
                'results': results,
                'analysis_time': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.save_analysis_result(batch_summary)
    
    def _show_available_images(self):
        """显示可用图片"""
        card_images_dir = Path("data/card_images")
        
        if not card_images_dir.exists():
            print("❌ 图片目录不存在")
            return
        
        image_files = list(card_images_dir.glob("*.jpg")) + \
                     list(card_images_dir.glob("*.png"))
        
        print(f"\n📷 在 {card_images_dir} 中找到 {len(image_files)} 张图片:")
        
        for img_file in image_files:
            size_kb = round(img_file.stat().st_size / 1024, 1)
            print(f"  - {img_file.name} ({size_kb} KB)")

def main():
    """主函数"""
    print("🌟 集成塔罗AI视觉系统")
    print("=" * 60)
    print("结合韦特塔罗识别和AI解读的完整系统")
    
    try:
        system = IntegratedTarotVisionSystem()
        system.interactive_demo()
        
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
    except Exception as e:
        print(f"\n💥 系统错误: {e}")

if __name__ == "__main__":
    main() 