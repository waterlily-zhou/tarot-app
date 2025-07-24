#!/usr/bin/env python3
"""
完整塔罗AI系统
整合图片识别、专业解牌、知识检索的完整解决方案
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# 导入各个子系统
from professional_tarot_ai import ProfessionalTarotAI
try:
    from simple_card_test import gemini_card_recognition, gemini_overlap_recognition
    RECOGNITION_AVAILABLE = True
except ImportError:
    print("⚠️ 图片识别模块不可用")
    RECOGNITION_AVAILABLE = False

class IntegratedTarotSystem:
    """完整塔罗AI系统"""
    
    def __init__(self):
        print("🌟 初始化完整塔罗AI系统...")
        
        # 初始化专业解牌AI
        self.tarot_ai = ProfessionalTarotAI()
        
        print("✅ 完整塔罗AI系统初始化完成！")
    
    def analyze_image_and_read(self, 
                              image_path: str, 
                              question: str = None,
                              user_id: str = None,
                              use_overlap_recognition: bool = False) -> Dict:
        """从图片识别到完整解牌的一站式服务"""
        
        print(f"🔮 开始完整塔罗分析流程...")
        print(f"图片: {image_path}")
        print(f"问题: {question}")
        
        start_time = time.time()
        
        # 1. 图片识别
        print("\n📸 步骤1: 图片识别...")
        if not RECOGNITION_AVAILABLE:
            print("❌ 图片识别功能不可用，使用模拟数据")
            recognized_cards = self._get_demo_cards()
        else:
            if use_overlap_recognition:
                recognized_cards = gemini_overlap_recognition(image_path)
            else:
                recognized_cards = gemini_card_recognition(image_path)
        
        if not recognized_cards:
            return {
                "success": False,
                "error": "图片识别失败",
                "total_time": time.time() - start_time
            }
        
        print(f"✅ 识别到 {len(recognized_cards)} 张卡牌")
        
        # 2. 转换识别结果
        print("\n🎴 步骤2: 分析卡牌布局...")
        reading_cards = self.tarot_ai.analyze_cards_from_recognition(recognized_cards)
        
        # 3. 自动识别牌阵类型
        spread_type, layout = self.tarot_ai.spread_system.analyze_card_layout(recognized_cards)
        print(f"🎯 识别到牌阵类型: {spread_type}")
        
        # 4. 生成专业解读
        print("\n🧠 步骤3: 生成专业解读...")
        reading = self.tarot_ai.generate_professional_reading(
            cards=reading_cards,
            spread_type=spread_type,
            question=question,
            user_id=user_id
        )
        
        total_time = time.time() - start_time
        
        # 5. 整理结果
        result = {
            "success": True,
            "recognition_result": recognized_cards,
            "reading": reading,
            "formatted_output": self.tarot_ai.format_reading_result(reading),
            "total_time": total_time,
            "process_breakdown": {
                "recognition_time": "约2-5秒",
                "analysis_time": f"{reading.generation_time:.2f}秒",
                "total_time": f"{total_time:.2f}秒"
            }
        }
        
        # 6. 保存结果
        self._save_complete_reading(result, user_id)
        
        print(f"\n✅ 完整分析流程完成！总用时: {total_time:.2f}秒")
        return result
    
    def _get_demo_cards(self) -> List[Dict]:
        """获取演示用的卡牌数据"""
        return [
            {"card_name": "皇后", "orientation": "正位", "position": "(1, 3)", "order": 1},
            {"card_name": "力量", "orientation": "正位", "position": "(2, 3)", "order": 2},
            {"card_name": "星币七", "orientation": "逆位", "position": "(3, 3)", "order": 3}
        ]
    
    def _save_complete_reading(self, result: Dict, user_id: str = None):
        """保存完整解读结果"""
        if not result.get("success"):
            return
            
        # 保存到专门的目录
        complete_readings_dir = Path("data/complete_readings")
        complete_readings_dir.mkdir(exist_ok=True)
        
        timestamp = int(result["reading"].timestamp)
        filename = f"complete_reading_{timestamp}.json"
        filepath = complete_readings_dir / filename
        
        # 准备保存的数据（移除不能序列化的对象）
        save_data = {
            "timestamp": result["reading"].timestamp,
            "user_id": result["reading"].user_id,
            "question": result["reading"].question,
            "recognition_result": result["recognition_result"],
            "spread_type": result["reading"].spread_type,
            "spread_name": result["reading"].spread_name,
            "cards": [
                {
                    "card_name": card.card_name,
                    "orientation": card.orientation,
                    "position": card.position,
                    "position_meaning": card.position_meaning,
                    "order": card.order
                }
                for card in result["reading"].cards
            ],
            "analyses": {
                "spread_analysis": result["reading"].spread_analysis,
                "relationship_analysis": result["reading"].relationship_analysis,
                "elemental_analysis": result["reading"].elemental_analysis,
                "overall_interpretation": result["reading"].overall_interpretation,
                "advice": result["reading"].advice
            },
            "metadata": {
                "confidence_score": result["reading"].confidence_score,
                "generation_time": result["reading"].generation_time,
                "total_time": result["total_time"],
                "model_used": result["reading"].model_used
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 完整解读已保存: {filepath}")
    
    def interactive_session(self):
        """交互式解牌会话"""
        print("\n🔮 欢迎使用完整塔罗AI系统")
        print("=" * 60)
        
        while True:
            print("\n请选择功能：")
            print("1. 🖼️  图片识别 + 专业解牌 (完整流程)")
            print("2. 📝 手动输入卡牌 + 专业解牌")
            print("3. 🎴 牌阵系统测试")
            print("4. 🃏 卡牌含义查询")
            print("5. 📊 系统状态查看")
            print("6. 🚪 退出")
            
            choice = input("\n请选择 (1-6): ").strip()
            
            if choice == "1":
                self._handle_image_reading()
            elif choice == "2":
                self._handle_manual_reading()
            elif choice == "3":
                self._handle_spread_test()
            elif choice == "4":
                self._handle_card_query()
            elif choice == "5":
                self._handle_system_status()
            elif choice == "6":
                print("👋 感谢使用完整塔罗AI系统！")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def _handle_image_reading(self):
        """处理图片解牌"""
        print("\n🖼️ 图片识别 + 专业解牌")
        print("-" * 40)
        
        if not RECOGNITION_AVAILABLE:
            print("❌ 图片识别功能不可用")
            print("💡 请确保已配置Gemini API Key")
            return
        
        # 获取输入
        image_path = input("请输入图片路径 (或按Enter使用默认测试图片): ").strip()
        if not image_path:
            image_path = "data/card_images/spread_0_4821735726296_.pic.jpg"
        
        if not Path(image_path).exists():
            print(f"❌ 图片不存在: {image_path}")
            return
        
        question = input("请输入你的问题 (可选): ").strip()
        user_id = input("请输入用户ID (可选): ").strip() or "anonymous"
        
        # 选择识别策略
        print("\n选择识别策略：")
        print("1. 🎯 标准识别 (快速)")
        print("2. 🔄 重叠分块识别 (更全面)")
        
        strategy = input("请选择 (1-2): ").strip()
        use_overlap = strategy == "2"
        
        # 执行分析
        result = self.analyze_image_and_read(
            image_path=image_path,
            question=question,
            user_id=user_id,
            use_overlap_recognition=use_overlap
        )
        
        if result["success"]:
            print("\n" + result["formatted_output"])
            
            # 询问是否查看详细分析
            detail = input("\n🔍 是否查看详细卡牌分析？(y/n): ").strip().lower()
            if detail == 'y':
                self._show_detailed_analysis(result["reading"])
        else:
            print(f"❌ 分析失败: {result.get('error', '未知错误')}")
    
    def _handle_manual_reading(self):
        """处理手动输入解牌"""
        print("\n📝 手动输入卡牌解牌")
        print("-" * 40)
        
        # 获取卡牌信息
        print("请输入卡牌信息（格式：卡牌名称,正位/逆位）")
        print("例如：皇后,正位 或 星币七,逆位")
        print("输入完成后按Enter继续，输入'done'结束")
        
        cards_input = []
        order = 1
        
        while True:
            card_input = input(f"第{order}张卡牌: ").strip()
            
            if card_input.lower() == 'done':
                break
            
            if ',' in card_input:
                parts = card_input.split(',')
                if len(parts) >= 2:
                    card_name = parts[0].strip()
                    orientation = parts[1].strip()
                    
                    cards_input.append({
                        "card_name": card_name,
                        "orientation": orientation,
                        "position": f"({order}, 1)",
                        "order": order
                    })
                    order += 1
                else:
                    print("❌ 格式错误，请重新输入")
            else:
                print("❌ 请使用格式：卡牌名称,正位/逆位")
        
        if not cards_input:
            print("❌ 未输入任何卡牌")
            return
        
        # 获取问题和用户信息
        question = input("\n请输入你的问题 (可选): ").strip()
        user_id = input("请输入用户ID (可选): ").strip() or "anonymous"
        
        # 转换为解牌格式
        reading_cards = self.tarot_ai.analyze_cards_from_recognition(cards_input)
        
        # 识别牌阵类型
        spread_type, _ = self.tarot_ai.spread_system.analyze_card_layout(cards_input)
        
        # 生成解读
        reading = self.tarot_ai.generate_professional_reading(
            cards=reading_cards,
            spread_type=spread_type,
            question=question,
            user_id=user_id
        )
        
        # 显示结果
        formatted_result = self.tarot_ai.format_reading_result(reading)
        print("\n" + formatted_result)
        
        # 保存结果
        result = {
            "success": True,
            "recognition_result": cards_input,
            "reading": reading,
            "formatted_output": formatted_result,
            "total_time": reading.generation_time
        }
        self._save_complete_reading(result, user_id)
    
    def _handle_spread_test(self):
        """处理牌阵测试"""
        print("\n🎴 牌阵系统测试")
        print("-" * 40)
        
        spreads = self.tarot_ai.spread_system.list_spreads()
        print("可用牌阵:")
        for i, spread_name in enumerate(spreads, 1):
            spread = self.tarot_ai.spread_system.get_spread(spread_name)
            print(f"{i}. {spread.name} - {spread.description}")
        
        try:
            choice = int(input(f"\n请选择牌阵 (1-{len(spreads)}): ")) - 1
            if 0 <= choice < len(spreads):
                spread_name = spreads[choice]
                spread = self.tarot_ai.spread_system.get_spread(spread_name)
                
                print(f"\n📖 {spread.name} 详细信息:")
                print(f"描述: {spread.description}")
                print(f"位置数: {len(spread.positions)}")
                
                print("\n位置说明:")
                for pos_id, position in spread.positions.items():
                    print(f"- {position.name}: {position.description}")
                
                if spread.interpretation_guide:
                    print(f"\n解读指南:\n{spread.interpretation_guide}")
                    
        except ValueError:
            print("❌ 请输入有效数字")
    
    def _handle_card_query(self):
        """处理卡牌查询"""
        print("\n🃏 卡牌含义查询")
        print("-" * 40)
        
        card_name = input("请输入卡牌名称: ").strip()
        
        card = self.tarot_ai.card_db.get_card(card_name)
        if card:
            print(f"\n📜 {card.card_name} 详细信息:")
            print(f"花色: {card.suit.value}")
            print(f"核心能量: {card.core_energy}")
            print(f"元素: {card.element}")
            print(f"占星: {card.astrology}")
            
            print(f"\n正位关键词: {', '.join(card.upright_keywords)}")
            print(f"逆位关键词: {', '.join(card.reversed_keywords)}")
            
            print(f"\n正位含义: {card.upright_meaning}")
            print(f"逆位含义: {card.reversed_meaning}")
            
            print(f"\n在不同位置的含义:")
            print(f"过去位置: {card.past_position}")
            print(f"现在位置: {card.present_position}")
            print(f"未来位置: {card.future_position}")
            print(f"建议位置: {card.advice_position}")
        else:
            print(f"❌ 未找到卡牌: {card_name}")
            print("💡 请确保使用正确的中文名称，如：愚人、魔法师、皇后等")
    
    def _handle_system_status(self):
        """处理系统状态查看"""
        print("\n📊 系统状态")
        print("-" * 40)
        
        # RAG系统状态
        rag_stats = self.tarot_ai.rag.get_database_stats()
        print("📚 知识库状态:")
        for name, count in rag_stats.items():
            print(f"  {name}: {count} 条记录")
        
        # 牌阵系统状态
        spreads = self.tarot_ai.spread_system.list_spreads()
        print(f"\n🎴 牌阵系统: {len(spreads)} 种牌阵")
        
        # 卡牌数据库状态
        card_count = len(self.tarot_ai.card_db.cards)
        print(f"🃏 卡牌数据库: {card_count} 张卡牌")
        
        # 图片识别状态
        print(f"📸 图片识别: {'✅ 可用' if RECOGNITION_AVAILABLE else '❌ 不可用'}")
        
        # 解读记录统计
        complete_readings_dir = Path("data/complete_readings")
        if complete_readings_dir.exists():
            reading_count = len(list(complete_readings_dir.glob("*.json")))
            print(f"📖 完整解读记录: {reading_count} 条")
        else:
            print("📖 完整解读记录: 0 条")
    
    def _show_detailed_analysis(self, reading):
        """显示详细卡牌分析"""
        print("\n🔍 详细卡牌分析")
        print("=" * 60)
        
        for card in reading.cards:
            if card.position_id in reading.card_analyses:
                print(f"\n{reading.card_analyses[card.position_id]}")
                print("-" * 40)

def main():
    """主函数"""
    system = IntegratedTarotSystem()
    
    # 显示欢迎信息
    print("\n🌟 完整塔罗AI系统")
    print("整合图片识别、专业解牌、知识检索的完整解决方案")
    print("=" * 60)
    
    # 开始交互会话
    system.interactive_session()

if __name__ == "__main__":
    main() 