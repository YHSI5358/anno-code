"""
BoundingBox class and related operations.
Encapsulates all bounding box related functionality including IOU calculations.
"""

from typing import Tuple, List
import numpy as np

class BoundingBox:
    """Represents a bounding box with coordinates and provides related operations."""
    
    def __init__(self, coords: List[float]):
        """
        Initialize with coordinates [x1, y1, x2, y2].
        
        Args:
            coords: List of 4 coordinates representing the bounding box
        """
        self.coords = np.array(coords, dtype=np.float32)
        self.x1, self.y1, self.x2, self.y2 = self.coords
        
    @property
    def area(self) -> float:
        """Calculate area of the bounding box."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
        
    def calculate_iou(self, other: 'BoundingBox') -> float:
        """
        Calculate Intersection over Union (IOU) with another bounding box.
        
        Args:
            other: Another BoundingBox instance
            
        Returns:
            IOU value between 0 and 1
        """
        inter_area = self._calculate_intersection_area(other)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0
        
    def calculate_giou(self, other: 'BoundingBox') -> float:
        """
        Calculate Generalized IOU (GIOU) with another bounding box.
        
        Args:
            other: Another BoundingBox instance
            
        Returns:
            GIOU value between -1 and 1
        """
        inter_area = self._calculate_intersection_area(other)
        union_area = self.area + other.area - inter_area
        
        # Calculate enclosing box (C)
        x_min = min(self.x1, other.x1)
        y_min = min(self.y1, other.y1)
        x_max = max(self.x2, other.x2)
        y_max = max(self.y2, other.y2)
        c_area = (x_max - x_min) * (y_max - y_min)
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou - (c_area - union_area) / c_area if c_area > 0 else 0
        
    def _calculate_intersection_area(self, other: 'BoundingBox') -> float:
        """Calculate intersection area between two bounding boxes."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        return max(0, x2 - x1) * max(0, y2 - y1)
        
    def __repr__(self) -> str:
        return f"BoundingBox([{self.x1}, {self.y1}, {self.x2}, {self.y2}])"
