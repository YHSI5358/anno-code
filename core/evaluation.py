"""
Evaluation metrics module.
Implements various metrics for evaluating bounding box aggregation quality.
"""

from typing import List, Dict, Tuple
import numpy as np
from .bounding_box import BoundingBox

class EvaluationMetrics:
    """Calculates various evaluation metrics for bounding box aggregation."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize with IOU threshold for matching.
        
        Args:
            iou_threshold: Minimum IOU to consider boxes matched
        """
        self.iou_threshold = iou_threshold
        
    def calculate_one_to_one_metrics(
        self,
        predicted_boxes: List[BoundingBox],
        ground_truth_boxes: List[BoundingBox],
        predicted_categories: List[str],
        ground_truth_categories: List[str]
    ) -> Dict:
        """
        Calculate one-to-one matching metrics.
        
        Args:
            predicted_boxes: List of predicted bounding boxes
            ground_truth_boxes: List of ground truth boxes
            predicted_categories: List of predicted categories
            ground_truth_categories: List of ground truth categories
            
        Returns:
            Dictionary containing various metrics
        """
        # Initialize metrics
        metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'correct_categories': 0,
            'wrong_categories': 0,
            'iou_sum': 0.0,
            'matched_pairs': []
        }
        
        # Create IOU matrix
        iou_matrix = np.zeros((len(ground_truth_boxes), len(predicted_boxes)))
        for i, gt_box in enumerate(ground_truth_boxes):
            for j, pred_box in enumerate(predicted_boxes):
                iou_matrix[i][j] = gt_box.calculate_iou(pred_box)
        
        # Greedy matching
        matched_pred = set()
        matched_gt = set()
        
        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
                
            gt_idx, pred_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            if gt_idx not in matched_gt and pred_idx not in matched_pred:
                metrics['true_positives'] += 1
                metrics['iou_sum'] += max_iou
                
                # Check category match
                if predicted_categories[pred_idx] == ground_truth_categories[gt_idx]:
                    metrics['correct_categories'] += 1
                else:
                    metrics['wrong_categories'] += 1
                    
                metrics['matched_pairs'].append((gt_idx, pred_idx))
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                
            # Zero out the matched row and column
            iou_matrix[gt_idx, :] = 0
            iou_matrix[:, pred_idx] = 0
            
        # Calculate remaining unmatched
        metrics['false_positives'] = len(predicted_boxes) - len(matched_pred)
        metrics['false_negatives'] = len(ground_truth_boxes) - len(matched_gt)
        
        # Calculate derived metrics
        precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
        recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = metrics['iou_sum'] / metrics['true_positives'] if metrics['true_positives'] > 0 else 0
        category_accuracy = metrics['correct_categories'] / (metrics['correct_categories'] + metrics['wrong_categories']) if (metrics['correct_categories'] + metrics['wrong_categories']) > 0 else 0
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_iou': mean_iou,
            'category_accuracy': category_accuracy
        })
        
        return metrics
        
    def calculate_braylan_metrics(
        self,
        predicted_boxes: List[BoundingBox],
        ground_truth_boxes: List[BoundingBox],
        predicted_categories: List[str],
        ground_truth_categories: List[str]
    ) -> Dict:
        """
        Calculate Braylan-style metrics (precision-like and recall-like).
        
        Args:
            predicted_boxes: List of predicted bounding boxes
            ground_truth_boxes: List of ground truth boxes
            predicted_categories: List of predicted categories
            ground_truth_categories: List of ground truth categories
            
        Returns:
            Dictionary containing PLG and RLG metrics
        """
        plg_scores = []
        rlg_scores = []
        correct_categories = 0
        wrong_categories = 0
        
        # Calculate PLG (precision-like)
        for gt_box, gt_cat in zip(ground_truth_boxes, ground_truth_categories):
            max_iou = 0
            matched_cat = None
            for pred_box, pred_cat in zip(predicted_boxes, predicted_categories):
                iou = gt_box.calculate_iou(pred_box)
                if iou > max_iou:
                    max_iou = iou
                    matched_cat = pred_cat
            plg_scores.append(max_iou)
            if matched_cat == gt_cat:
                correct_categories += 1
            elif matched_cat is not None:
                wrong_categories += 1
                
        # Calculate RLG (recall-like)
        for pred_box, pred_cat in zip(predicted_boxes, predicted_categories):
            max_iou = 0
            matched_cat = None
            for gt_box, gt_cat in zip(ground_truth_boxes, ground_truth_categories):
                iou = pred_box.calculate_iou(gt_box)
                if iou > max_iou:
                    max_iou = iou
                    matched_cat = gt_cat
            rlg_scores.append(max_iou)
            
        mean_plg = np.mean(plg_scores) if plg_scores else 0
        mean_rlg = np.mean(rlg_scores) if rlg_scores else 0
        braylan_f1 = 2 * mean_plg * mean_rlg / (mean_plg + mean_rlg) if (mean_plg + mean_rlg) > 0 else 0
        category_accuracy = correct_categories / (correct_categories + wrong_categories) if (correct_categories + wrong_categories) > 0 else 0
        
        return {
            'mean_plg': mean_plg,
            'mean_rlg': mean_rlg,
            'braylan_f1': braylan_f1,
            'category_accuracy': category_accuracy,
            'plg_scores': plg_scores,
            'rlg_scores': rlg_scores
        }
