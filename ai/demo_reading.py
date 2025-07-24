#!/usr/bin/env python3
"""
塔罗AI系统演示脚本
展示完整的AI解牌流程
"""

from tarot_ai_system import TarotAISystem

def demo_reading():
    print("🌟 塔罗AI系统演示")
    print("=" * 60)
    
    # 初始化系统
    print("1. 初始化AI系统...")
    tarot_ai = TarotAISystem()
    
    # 演示用的测试数据
    test_cases = [
        {
            "cards": ["皇后", "力量", "星币七"],
            "question": "关于个人成长和心轮能量的指导",
            "user_id": "mel"
        },
        {
            "cards": ["魔法师", "塔", "审判"],
            "question": "未来一年的事业发展",
            "user_id": "mel"
        }
    ]
    
    print(f"\n📊 系统状态检查:")
    stats = tarot_ai.get_system_stats()
    print(f"- 知识库记录: {stats['knowledge_base']}")
    print(f"- LLM模型: {stats['llm_model']}")
    
    # 执行演示解牌
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🔮 演示解牌 #{i}")
        print(f"{'='*60}")
        
        result = tarot_ai.generate_reading(
            cards=test_case["cards"],
            question=test_case["question"],
            user_id=test_case["user_id"],
            spread_type="三卡展开"
        )
        
        # 显示结果
        print(f"\n🎴 卡牌组合: {', '.join(result['cards'])}")
        print(f"❓ 解读问题: {result['question']}")
        print(f"🕐 生成时间: {result['generation_time']:.2f}秒")
        print(f"🤖 AI模型: {result['model_used']}")
        
        print(f"\n📖 AI解读:")
        print("-" * 50)
        print(result['reading'])
        print("-" * 50)
        
        print(f"\n🔍 使用的背景知识片段:")
        print(result['context_used'])
        
        # 保存结果
        timestamp = int(result['timestamp'])
        filename = f"demo_reading_{i}_{timestamp}.json"
        
        import json
        from pathlib import Path
        
        demo_dir = Path("data/demo_results")
        demo_dir.mkdir(exist_ok=True)
        
        with open(demo_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 演示结果已保存: {demo_dir / filename}")
        
        # 间隔
        if i < len(test_cases):
            input("\n⏸️  按Enter继续下一个演示...")
    
    print(f"\n✅ 演示完成！")
    
    # 显示最终统计
    final_stats = tarot_ai.get_system_stats()
    print(f"\n📊 最终统计:")
    print(f"- 知识库: {final_stats['knowledge_base']}")
    print(f"- 演示解读: {len(test_cases)} 次")
    print(f"- 结果保存在: data/demo_results/")

if __name__ == "__main__":
    demo_reading() 