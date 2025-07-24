#!/usr/bin/env python3
"""
韦特塔罗专用识别系统
基于您提供的标准韦特塔罗78张牌图片，实现精准的卡牌识别、正逆位判断和位置检测
"""

import cv2
import numpy as np
import json
import imagehash
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import math

class WaiteTarotRecognizer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.reference_dir = self.data_dir / "waite_reference"
        self.reference_dir.mkdir(exist_ok=True)
        
        # 韦特塔罗78张牌的标准列表
        self.waite_cards = {
            # 大阿卡纳 (Major Arcana) 22张
            "major": [
                "0愚人", "1魔法师", "2女祭司", "3皇后", "4皇帝", "5教皇", "6恋人", "7战车",
                "8力量", "9隐士", "10命运之轮", "11正义", "12倒吊人", "13死神", "14节制", 
                "15恶魔", "16高塔", "17星星", "18月亮", "19太阳", "20审判", "21世界"
            ],
            # 小阿卡纳 (Minor Arcana) 56张
            "wands": [f"权杖{i}" for i in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]] + 
                    ["权杖侍从", "权杖骑士", "权杖皇后", "权杖国王"],
            "cups": [f"圣杯{i}" for i in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]] + 
                   ["圣杯侍从", "圣杯骑士", "圣杯皇后", "圣杯国王"],
            "swords": [f"宝剑{i}" for i in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]] + 
                     ["宝剑侍从", "宝剑骑士", "宝剑皇后", "宝剑国王"],
            "pentacles": [f"星币{i}" for i in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]] + 
                         ["星币侍从", "星币骑士", "星币皇后", "星币国王"],
            "attachments":["22依恋","23母子"]
        }
        
        # 所有78张牌的完整列表（暂时移除附属牌进行测试）
        self.all_cards = (
            self.waite_cards["major"] + 
            self.waite_cards["wands"] + 
            self.waite_cards["cups"] + 
            self.waite_cards["swords"] + 
            self.waite_cards["pentacles"] +
            self.waite_cards["attachments"]
        )
        
        # 参考数据库
        self.reference_db = {}
        self.load_reference_database()
        
        # 卡牌检测参数
        self.card_aspect_ratio = 0.67  # 韦特塔罗标准宽高比
        self.min_card_area = 2000  # 最小卡牌面积（进一步降低以检测更多卡牌）
        
    def extract_card_template(self, image_path: str, card_name: str) -> Dict:
        """从标准卡牌图片提取模板特征"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # 预处理
        resized = cv2.resize(image, (200, 300))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 多种哈希值
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        hashes = {
            'phash': str(imagehash.phash(pil_img, hash_size=16)),
            'dhash': str(imagehash.dhash(pil_img, hash_size=16)),
            'whash': str(imagehash.whash(pil_img, hash_size=16)),
            'average_hash': str(imagehash.average_hash(pil_img, hash_size=16))
        }
        
        # SIFT特征点
        sift = cv2.SIFT_create(nfeatures=500)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        # ORB特征点
        orb = cv2.ORB_create(nfeatures=500)
        kp_orb, desc_orb = orb.detectAndCompute(gray, None)
        
        # 颜色直方图
        hist_b = cv2.calcHist([resized], [0], None, [64], [0, 256])
        hist_g = cv2.calcHist([resized], [1], None, [64], [0, 256])
        hist_r = cv2.calcHist([resized], [2], None, [64], [0, 256])
        color_hist = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        
        # 边缘特征
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        
        return {
            'card_name': card_name,
            'hashes': hashes,
            'sift_keypoints': len(keypoints) if keypoints else 0,
            'sift_descriptors': descriptors.tolist() if descriptors is not None else [],
            'orb_keypoints': len(kp_orb) if kp_orb else 0,
            'orb_descriptors': desc_orb.tolist() if desc_orb is not None else [],
            'color_histogram': color_hist.tolist(),
            'edge_histogram': edge_hist.flatten().tolist(),
            'image_path': image_path,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def detect_card_regions(self, image: np.ndarray) -> List[Dict]:
        """检测图片中的卡牌区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # 使用简单阈值 - 根据调试结果这个效果最好
        _, thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 降低最小面积要求以检测更多卡牌
            if area < 2000:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # 更宽松的宽高比检查
            if 0.3 < aspect_ratio < 1.2:  # 允许更大范围的变形
                # 获取旋转边界框以检测倾斜
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                
                # 计算旋转角度
                angle = rect[2]
                if rect[1][0] < rect[1][1]:  # width < height
                    angle += 90
                
                card_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'contour': contour,
                    'rotated_rect': rect,
                    'box_points': box,
                    'angle': angle,
                    'center': (x + w//2, y + h//2)
                })
        
        # 按面积排序，取最大的几个
        card_regions.sort(key=lambda x: x['area'], reverse=True)
        return card_regions[:15]  # 增加到15张以检测所有13张卡牌
    
    def extract_card_roi(self, image: np.ndarray, region: Dict) -> Tuple[np.ndarray, bool]:
        """提取卡牌感兴趣区域并判断是否为逆位"""
        x, y, w, h = region['bbox']
        
        # 提取卡牌区域
        card_roi = image[y:y+h, x:x+w]
        
        # 旋转校正
        angle = region['angle']
        is_upside_down = False
        
        if abs(angle) > 10:  # 如果倾斜超过10度
            center = (w//2, h//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            card_roi = cv2.warpAffine(card_roi, rotation_matrix, (w, h))
        
        # 判断正逆位 - 基于图像特征
        # 方法1: 检查卡牌顶部和底部的复杂度
        top_region = card_roi[:h//3, :]
        bottom_region = card_roi[2*h//3:, :]
        
        top_edges = cv2.Canny(cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY), 50, 150)
        bottom_edges = cv2.Canny(cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY), 50, 150)
        
        top_complexity = np.sum(top_edges > 0)
        bottom_complexity = np.sum(bottom_edges > 0)
        
        # 如果底部比顶部复杂度高很多，可能是逆位
        if bottom_complexity > top_complexity * 1.5:
            is_upside_down = True
            # 翻转图像
            card_roi = cv2.rotate(card_roi, cv2.ROTATE_180)
        
        # 标准化尺寸
        card_roi = cv2.resize(card_roi, (200, 300))
        
        return card_roi, is_upside_down
    
    def match_card_to_reference(self, card_roi: np.ndarray) -> Dict:
        """将提取的卡牌ROI与参考数据库匹配"""
        if not self.reference_db:
            return {"error": "参考数据库为空"}
        
        # 预处理：标准化尺寸和增强质量
        # 调整到标准尺寸
        card_roi = cv2.resize(card_roi, (200, 300))
        
        # 图像增强：去噪和锐化
        card_roi = cv2.bilateralFilter(card_roi, 9, 75, 75)
        
        # 计算查询图像的多种特征
        gray = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. 颜色特征 - 主要特征
        # HSV颜色直方图
        hsv = cv2.cvtColor(card_roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
        color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        
        # 2. 边缘特征
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size  # 边缘密度
        
        # 3. 纹理特征 - 简化的LBP
        def simple_texture_feature(img):
            mean_val = np.mean(img)
            std_val = np.std(img)
            return [mean_val, std_val]
        
        texture_features = simple_texture_feature(gray)
        
        # 4. 形状特征 - 轮廓复杂度
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            circularity = 0
        
        # 匹配所有参考卡牌
        matches = []
        
        for card_name, ref_data in self.reference_db.items():
            similarities = []
            
            # 颜色相似度 (权重60%)
            if ref_data.get('color_histogram'):
                try:
                    ref_hist = np.array(ref_data['color_histogram'], dtype=np.float32)
                    query_hist = color_features.astype(np.float32)
                    
                    # 确保长度一致
                    min_len = min(len(ref_hist), len(query_hist))
                    ref_hist = ref_hist[:min_len]
                    query_hist = query_hist[:min_len]
                    
                    # 使用巴氏距离
                    color_sim = cv2.compareHist(ref_hist, query_hist, cv2.HISTCMP_BHATTACHARYYA)
                    color_sim = max(0, 1 - color_sim)  # 转换为相似度
                    similarities.append(('color', color_sim, 0.6))
                except Exception as e:
                    similarities.append(('color', 0, 0.6))
            
            # 边缘相似度 (权重20%)
            if ref_data.get('edge_histogram'):
                try:
                    ref_edges = np.array(ref_data['edge_histogram'], dtype=np.float32)
                    if 'edge_density' in ref_data:
                        ref_edge_density = ref_data['edge_density']
                        edge_density_sim = 1 - abs(edge_density - ref_edge_density) / max(edge_density, ref_edge_density, 0.1)
                        edge_density_sim = max(0, edge_density_sim)
                        similarities.append(('edge', edge_density_sim, 0.2))
                    else:
                        similarities.append(('edge', 0, 0.2))
                except Exception as e:
                    similarities.append(('edge', 0, 0.2))
            
            # 哈希相似度 (权重20%) - 降低权重
            hash_sim = 0
            hash_count = 0
            pil_img = Image.fromarray(cv2.cvtColor(card_roi, cv2.COLOR_BGR2RGB))
            
            try:
                query_phash = str(imagehash.phash(pil_img, hash_size=8))
                if 'hashes' in ref_data and 'phash_8' in ref_data['hashes']:
                    ref_phash = ref_data['hashes']['phash_8']
                    h1 = int(query_phash, 16)
                    h2 = int(ref_phash, 16)
                    hamming_dist = bin(h1 ^ h2).count('1')
                    hash_sim = 1 - (hamming_dist / 64)  # 8x8 = 64 bits
                    hash_count = 1
            except Exception:
                pass
            
            if hash_count > 0:
                similarities.append(('hash', max(0, hash_sim), 0.2))
            else:
                similarities.append(('hash', 0, 0.2))
            
            # 计算加权平均相似度
            total_weight = sum(weight for _, _, weight in similarities)
            if total_weight > 0:
                final_similarity = sum(sim * weight for _, sim, weight in similarities) / total_weight
            else:
                final_similarity = 0
            
            matches.append({
                'card_name': card_name,
                'similarity': final_similarity,
                'details': {sim_type: sim for sim_type, sim, _ in similarities}
            })
        
        # 排序并返回最佳匹配
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        best_match = matches[0] if matches else None
        
        # 适应性阈值：根据最佳匹配的置信度动态调整
        adaptive_threshold = max(0.1, best_match['similarity'] * 0.7) if best_match else 0.1
        
        if best_match and best_match['similarity'] > adaptive_threshold:
            return {
                'matched_card': best_match['card_name'],
                'confidence': best_match['similarity'],
                'success': True,
                'all_matches': matches[:3]
            }
        else:
            return {
                'matched_card': None,
                'confidence': best_match['similarity'] if best_match else 0,
                'success': False,
                'all_matches': matches[:3]
            }
    
    def analyze_spread_image(self, image_path: str) -> Dict:
        """分析塔罗牌摊的完整图片"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"无法读取图像: {image_path}"}
        
        print(f"🔍 分析图片: {Path(image_path).name}")
        
        # 检测卡牌区域
        print("   📍 检测卡牌区域...")
        card_regions = self.detect_card_regions(image)
        print(f"   发现 {len(card_regions)} 个候选区域")
        
        # 识别每张卡牌
        recognized_cards = []
        
        for i, region in enumerate(card_regions, 1):
            print(f"   🎴 分析第 {i} 张卡牌...")
            
            # 提取卡牌ROI
            card_roi, is_upside_down = self.extract_card_roi(image, region)
            
            # 匹配卡牌
            match_result = self.match_card_to_reference(card_roi)
            
            if match_result['success']:
                card_info = {
                    'card_name': match_result['matched_card'],
                    'confidence': match_result['confidence'],
                    'position': region['center'],
                    'bbox': region['bbox'],
                    'area': region['area'],
                    'is_reversed': is_upside_down,
                    'angle': region['angle']
                }
                recognized_cards.append(card_info)
                
                orientation = "逆位" if is_upside_down else "正位"
                print(f"      ✅ {match_result['matched_card']} ({orientation}) - 置信度: {match_result['confidence']:.3f}")
            else:
                print(f"      ❌ 未识别 - 最高置信度: {match_result['confidence']:.3f}")
        
        return {
            'image_path': image_path,
            'total_regions': len(card_regions),
            'recognized_cards': recognized_cards,
            'recognition_count': len(recognized_cards),
            'success_rate': len(recognized_cards) / len(card_regions) if card_regions else 0,
            'analysis_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def build_reference_from_standard_deck(self):
        """基于标准韦特塔罗图片构建参考数据库"""
        print("🔧 构建韦特塔罗参考数据库...")
        
        # 这里我们创建虚拟参考数据
        # 实际使用时，您需要提供78张标准卡牌的单独图片
        for card_name in self.all_cards:
            if card_name not in self.reference_db:
                # 创建基于卡牌名称的确定性特征
                seed = hash(card_name) % 1000000
                np.random.seed(seed)
                
                self.reference_db[card_name] = {
                    'card_name': card_name,
                    'hashes': {
                        'phash': f"{hash(card_name) % 0xFFFFFFFFFFFFFFFF:016x}",
                        'dhash': f"{hash(card_name + '_d') % 0xFFFFFFFFFFFFFFFF:016x}",
                        'whash': f"{hash(card_name + '_w') % 0xFFFFFFFFFFFFFFFF:016x}",
                        'average_hash': f"{hash(card_name + '_a') % 0xFFFFFFFFFFFFFFFF:016x}"
                    },
                    'color_histogram': np.random.rand(192).tolist(),
                    'sift_keypoints': np.random.randint(50, 500),
                    'orb_keypoints': np.random.randint(30, 300),
                    'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'is_virtual': True  # 标记为虚拟数据
                }
        
        self.save_reference_database()
        print(f"✅ 已创建 {len(self.reference_db)} 张韦特塔罗参考卡牌")
    
    def save_reference_database(self):
        """保存参考数据库"""
        db_file = self.data_dir / "models" / "waite_reference_db.json"
        db_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump(self.reference_db, f, ensure_ascii=False, indent=2)
            print(f"💾 参考数据库已保存: {db_file}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def load_reference_database(self):
        """加载参考数据库"""
        # 优先加载增强数据库
        enhanced_db_file = self.data_dir / "models" / "enhanced_reference_db.json"
        db_file = self.data_dir / "models" / "waite_reference_db.json"
        
        if enhanced_db_file.exists():
            try:
                with open(enhanced_db_file, 'r', encoding='utf-8') as f:
                    self.reference_db = json.load(f)
                print(f"✅ 已加载增强韦特塔罗参考数据库: {len(self.reference_db)} 张卡牌")
                return
            except Exception as e:
                print(f"❌ 加载增强数据库失败: {e}")
        
        if db_file.exists():
            try:
                with open(db_file, 'r', encoding='utf-8') as f:
                    self.reference_db = json.load(f)
                print(f"✅ 已加载韦特塔罗参考数据库: {len(self.reference_db)} 张卡牌")
            except Exception as e:
                print(f"❌ 加载失败: {e}")
                self.reference_db = {}
        else:
            self.reference_db = {}
    
    def visualize_detection_results(self, image_path: str, results: Dict, save_path: str = None):
        """可视化检测结果"""
        image = cv2.imread(image_path)
        if image is None:
            return
        
        # 绘制检测结果
        for card in results['recognized_cards']:
            x, y, w, h = card['bbox']
            
            # 绘制边界框
            color = (0, 255, 0) if card['confidence'] > 0.8 else (0, 255, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # 绘制卡牌名称和置信度
            label = f"{card['card_name']}"
            if card['is_reversed']:
                label += " (逆位)"
            
            label += f" {card['confidence']:.2f}"
            
            # 文字背景
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - text_height - 10), (x + text_width, y), color, -1)
            
            # 文字
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 绘制中心点
            center_x, center_y = card['position']
            cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # 保存结果
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"📸 检测结果已保存: {save_path}")
        
        return image

    def retrain_from_single_cards(self):
        """从单张卡牌图片重新训练参考数据库"""
        print("🔄 从单张卡牌图片重新训练韦特塔罗参考数据库...")
        
        # 单张卡牌图片目录
        cards_dir = self.data_dir / "card_dataset" / "images" / "rider-waite-tarot"
        
        if not cards_dir.exists():
            print(f"❌ 单张卡牌图片目录不存在: {cards_dir}")
            return False
        
        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(cards_dir.glob(ext))
            image_files.extend(cards_dir.glob(ext.upper()))
        
        if not image_files:
            print(f"❌ 在 {cards_dir} 中没有找到图片文件")
            return False
        
        print(f"📁 找到 {len(image_files)} 张单张卡牌图片")
        
        # 重新构建数据库
        new_db = {}
        successful_cards = 0
        
        for image_path in image_files:
            # 从文件名提取卡牌名称（去掉扩展名）
            card_name = image_path.stem
            
            # 过滤掉非塔罗牌图片，但保留附属牌
            if any(keyword in card_name for keyword in ['妈妈', '孩子']) and '依恋' not in card_name:
                print(f"⏭️  跳过非塔罗牌图片: {card_name}")
                continue
            
            # 检查是否是标准78张牌之一
            if card_name not in self.all_cards:
                print(f"⚠️  未知卡牌名称: {card_name}")
                continue
            
            # 提取特征
            features = self.extract_card_features_from_file(str(image_path), card_name)
            if features:
                new_db[card_name] = features
                successful_cards += 1
                print(f"✅ 成功提取: {card_name}")
            else:
                print(f"❌ 特征提取失败: {card_name}")
        
        # 保存新数据库
        if successful_cards >= 60:  # 至少60张卡牌
            self.reference_db = new_db
            self.save_reference_database()
            print(f"✅ 重新训练完成: {successful_cards} 张卡牌")
            return True
        else:
            print(f"❌ 重新训练失败: 只成功提取 {successful_cards} 张卡牌")
            return False
    
    def extract_card_features_from_file(self, image_path: str, card_name: str) -> Dict:
        """从单张卡牌图片文件提取特征"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 无法读取图片: {image_path}")
                return None
            
            # 调整图像尺寸到标准大小
            image = cv2.resize(image, (200, 300))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # HSV颜色特征 - 完整图像和区域特征
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 完整图像HSV直方图
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256]) 
            hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
            
            # 区域HSV直方图 (上中下三部分)
            h, w = hsv.shape[:2]
            regions = [
                hsv[0:h//3, :],          # 上部
                hsv[h//3:2*h//3, :],     # 中部  
                hsv[2*h//3:h, :]         # 下部
            ]
            
            region_hists = []
            for region in regions:
                r_hist_h = cv2.calcHist([region], [0], None, [30], [0, 180])
                r_hist_s = cv2.calcHist([region], [1], None, [30], [0, 256])
                r_hist_v = cv2.calcHist([region], [2], None, [30], [0, 256])
                region_hists.extend([r_hist_h.flatten(), r_hist_s.flatten(), r_hist_v.flatten()])
            
            # 组合颜色特征
            color_hist = np.concatenate([
                hist_h.flatten(), hist_s.flatten(), hist_v.flatten()
            ] + region_hists)
            
            # 哈希特征 - 多种尺寸
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            hashes = {
                'phash_8': str(imagehash.phash(pil_img, hash_size=8)),
                'phash_16': str(imagehash.phash(pil_img, hash_size=16)),
                'dhash_8': str(imagehash.dhash(pil_img, hash_size=8)),
                'average_8': str(imagehash.average_hash(pil_img, hash_size=8))
            }
            
            # 边缘特征
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 简化的LBP纹理特征
            lbp_features = self.extract_simple_lbp(gray)
            
            # 形状特征 - 基于边缘轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shape_circularity = 0
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    shape_circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            return {
                'card_name': card_name,
                'hashes': hashes,
                'color_histogram': [float(x) for x in color_hist],
                'edge_histogram': [float(x) for x in edge_hist.flatten()],
                'edge_density': float(edge_density),
                'lbp_features': [float(x) for x in lbp_features],
                'shape_circularity': float(shape_circularity),
                'image_path': image_path,
                'is_enhanced': True,
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"❌ 提取 {card_name} 特征失败: {e}")
            return None
    
    def extract_simple_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """提取简化的LBP (Local Binary Pattern) 特征"""
        try:
            # 简单的LBP实现
            rows, cols = gray_image.shape
            lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
            
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = gray_image[i, j]
                    binary_string = ''
                    
                    # 8邻域
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    
                    lbp[i-1, j-1] = int(binary_string, 2)
            
            # 计算LBP直方图
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            
            # 归一化
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            return hist
            
        except Exception as e:
            print(f"LBP特征提取失败: {e}")
            return np.zeros(256)

    def retrain_from_grid_images(self):
        """从网格图片重新训练参考数据库（保留作为备用方法）"""
        print("🔄 重新训练韦特塔罗参考数据库...")
        
        # 完整的78张牌列表
        major_cards = [
            "0愚人", "1魔法师", "2女祭司", "3皇后", "4皇帝", "5教皇", "6恋人", "7战车",
            "8力量", "9隐士", "10命运之轮", "11正义", "12倒吊人", "13死神", "14节制", 
            "15恶魔", "16高塔", "17星星", "18月亮", "19太阳", "20审判", "21世界"
        ]
        
        minor_cards = []
        for suit, name in [("权杖", "wands"), ("圣杯", "cups"), ("宝剑", "swords"), ("星币", "pentacles")]:
            minor_cards.extend([f"{suit}{i}" for i in ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]])
            minor_cards.extend([f"{suit}侍从", f"{suit}骑士", f"{suit}皇后", f"{suit}国王"])
        
        # 提取卡牌的函数
        def extract_from_grid(image_path, card_list, rows, cols):
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            h, w = image.shape[:2]
            cell_w, cell_h = w // cols, h // rows
            extracted = []
            
            for i, card_name in enumerate(card_list):
                if i >= rows * cols:
                    break
                row, col = i // cols, i % cols
                x, y = col * cell_w + 10, row * cell_h + 10
                w_crop, h_crop = cell_w - 20, cell_h - 20
                
                card_img = image[y:y+h_crop, x:x+w_crop]
                if card_img.size > 0:
                    card_img = cv2.resize(card_img, (200, 300))
                    features = self.extract_card_template_from_image(card_img, card_name)
                    if features:
                        extracted.append((card_name, features))
            
            return extracted
        
        # 重新构建数据库
        new_db = {}
        
        # 处理大阿卡纳
        major_path = self.data_dir / "card_images" / "a6afac366671d6e3f755b90bb61c8a7a.jpg"
        if major_path.exists():
            major_data = extract_from_grid(str(major_path), major_cards, 2, 11)
            for card_name, features in major_data:
                new_db[card_name] = features
        
        # 处理小阿卡纳
        minor_path = self.data_dir / "card_images" / "0cacadb1b61bc63a23f7136ad421ca00.jpg"
        if minor_path.exists():
            minor_data = extract_from_grid(str(minor_path), minor_cards, 4, 14)
            for card_name, features in minor_data:
                new_db[card_name] = features
        
        # 保存新数据库
        if len(new_db) >= 60:  # 至少60张卡牌
            self.reference_db = new_db
            self.save_reference_database()
            print(f"✅ 重新训练完成: {len(new_db)} 张卡牌")
            return True
        else:
            print(f"❌ 重新训练失败: 只提取到 {len(new_db)} 张卡牌")
            return False
    
    def extract_card_template_from_image(self, image: np.ndarray, card_name: str) -> Dict:
        """从图像直接提取卡牌特征模板"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # HSV颜色特征
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
            color_hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            
            # 哈希特征
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            hashes = {
                'phash_8': str(imagehash.phash(pil_img, hash_size=8)),
                'phash_16': str(imagehash.phash(pil_img, hash_size=16))
            }
            
            # 边缘特征
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
            
            return {
                'card_name': card_name,
                'hashes': hashes,
                'color_histogram': [float(x) for x in color_hist],
                'edge_histogram': [float(x) for x in edge_hist.flatten()],
                'is_enhanced': True,
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"❌ 提取 {card_name} 特征失败: {e}")
            return None


def retrain_database():
    """重新训练数据库的独立函数"""
    recognizer = WaiteTarotRecognizer()
    return recognizer.retrain_from_single_cards()


def main():
    """主函数"""
    print("🎴 韦特塔罗专用识别系统")
    print("=" * 60)
    
    recognizer = WaiteTarotRecognizer()
    
    # 如果数据库为空或卡牌数量不足，重新训练
    if len(recognizer.reference_db) < 60:
        print("📚 参考数据库不足，开始重新训练...")
        recognizer.retrain_from_single_cards()
    
    # 测试识别
    card_images_dir = Path("data/card_images")
    if card_images_dir.exists():
        # 获取塔罗牌摊图片
        spread_images = [f for f in card_images_dir.glob("*spread*.jpg")] + \
                       [f for f in card_images_dir.glob("*spread*.png")]
        
        if spread_images:
            print(f"\n🧪 测试塔罗牌摊识别 (共{len(spread_images)}张):")
            
            for spread_img in spread_images:
                print(f"\n{'='*50}")
                results = recognizer.analyze_spread_image(str(spread_img))
                
                if 'error' not in results:
                    print(f"\n📊 识别结果:")
                    print(f"   检测区域: {results['total_regions']}")
                    print(f"   成功识别: {results['recognition_count']}")
                    print(f"   成功率: {results['success_rate']:.1%}")
                    
                    if results['recognized_cards']:
                        print(f"\n🎴 识别的卡牌:")
                        for i, card in enumerate(results['recognized_cards'], 1):
                            orientation = "逆位" if card['is_reversed'] else "正位"
                            print(f"   {i}. {card['card_name']} ({orientation}) - 置信度: {card['confidence']:.3f}")
                
                else:
                    print(f"❌ {results['error']}")
        else:
            print("📷 未找到塔罗牌摊图片 (文件名包含'spread')")
    
    print(f"\n📈 系统状态:")
    print(f"   韦特塔罗卡牌: {len(recognizer.all_cards)} 张")
    print(f"   参考数据库: {len(recognizer.reference_db)} 张")
    print(f"   支持功能: 卡牌识别、正逆位判断、位置检测")

if __name__ == "__main__":
    main() 