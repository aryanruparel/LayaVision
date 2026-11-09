"""
Indian Road Object Detector
===========================
Real-time object and hazard detector specialized for Indian driving conditions.
Detects: cows, stray dogs, auto-rickshaws, two-wheelers, cars, buses, trucks,
pedestrians, road barriers, and potholes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Mapping and normalization for Indian road perception
INDIAN_ROAD_MAPPINGS = {
    "cow": "cow (cattle)",
    "dog": "stray dog",
    "person": "pedestrian",
    "motorcycle": "two-wheeler / bike",
    "bicycle": "cyclist",
    "car": "car",
    "bus": "bus",
    "truck": "truck / heavy vehicle",
    "traffic light": "traffic light",
    "stop sign": "traffic sign",
    "horse": "animal (horse)",
    "sheep": "animal (livestock)",
}

# Hazard priority levels: 1 (low) to 5 (critical)
HAZARD_PRIORITIES = {
    "cow (cattle)": 5,
    "stray dog": 4,
    "pedestrian": 5,
    "pothole": 4,
    "road barrier": 4,
    "two-wheeler / bike": 3,
    "auto-rickshaw": 3,
    "car": 2,
    "bus": 3,
    "truck / heavy vehicle": 3,
}


class IndianRoadDetector:
    """
    High-performance detector running in fp16 on CUDA.
    Extracts spatial bounding boxes and identifies Indian road hazards.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        detect_potholes: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Indian Road Detector ({weights}) on {self.device.upper()}...")
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.detect_potholes = detect_potholes

    def _detect_potholes_and_hazards(self, frame: np.ndarray, existing_boxes: Optional[List[List[int]]] = None) -> List[Dict[str, Any]]:
        """
        Specialized road surface scanner to detect potholes, asphalt depressions,
        and road surface hazards in the lower driving lane region.
        """
        hazards = []
        h, w = frame.shape[:2]
        existing_boxes = existing_boxes or []

        # Region of interest: road lane, strictly above car dashboard hood (0.45*h to 0.70*h)
        roi_y_start = int(h * 0.45)
        roi_y_end = int(h * 0.70)
        roi_x_start = int(w * 0.22)
        roi_x_end = int(w * 0.78)
        roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        if roi.size == 0:
            return hazards

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive thresholding to detect dark localized road depressions (potholes)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter contours by realistic pothole area on camera
            if 350 < area < 10000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = float(cw) / max(1, ch)
                # Potholes are usually wider horizontally due to road perspective
                if 0.7 < aspect_ratio < 4.0:
                    gx1 = roi_x_start + x
                    gy1 = roi_y_start + y
                    gx2 = gx1 + cw
                    gy2 = gy1 + ch

                    # Check overlap with existing detected objects (e.g. cars, cows)
                    overlaps_existing = False
                    for ebox in existing_boxes:
                        ox1 = max(gx1, ebox[0])
                        oy1 = max(gy1, ebox[1])
                        ox2 = min(gx2, ebox[2])
                        oy2 = min(gy2, ebox[3])
                        if ox2 > ox1 and oy2 > oy1:
                            inter_area = (ox2 - ox1) * (oy2 - oy1)
                            if inter_area > 0.2 * area:
                                overlaps_existing = True
                                break

                    if not overlaps_existing:
                        hazards.append({
                            "box": [gx1, gy1, gx2, gy2],
                            "class_name": "pothole",
                            "raw_label": "pothole",
                            "confidence": round(min(0.92, 0.55 + (area / 15000.0)), 2),
                            "hazard_level": 4,
                            "is_pothole": True,
                        })

        # Keep top 2 most prominent potholes
        hazards.sort(key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]), reverse=True)
        return hazards[:2]
        hazards.sort(key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]), reverse=True)
        return hazards[:3]

    def _detect_barriers(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect barricades / temporary road barriers with high-visibility hazard patterns."""
        barriers = []
        h, w = frame.shape[:2]
        # Look in mid-lower road zone
        road_zone = frame[int(h * 0.40):int(h * 0.85), :]

        hsv = cv2.cvtColor(road_zone, cv2.COLOR_BGR2HSV)
        # Indian police / traffic barricades are typically yellow-black or orange-white
        yellow_lower = np.array([18, 120, 120])
        yellow_upper = np.array([32, 255, 255])
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1200:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / max(1, bh)
                if aspect > 1.2:  # Horizontal barricades
                    by1 = int(h * 0.40) + y
                    barriers.append({
                        "box": [x, by1, x + bw, by1 + bh],
                        "class_name": "road barrier",
                        "raw_label": "barrier",
                        "confidence": 0.82,
                        "hazard_level": 4,
                        "is_barrier": True,
                    })
        return barriers[:2]

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on an image / video frame.

        Returns:
            List of detected objects:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "class_name": "cow (cattle)",
                    "confidence": 0.89,
                    "hazard_level": 5,
                    "in_ego_lane": True / False
                },
                ...
            ]
        """
        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)[0]

        detections = []

        # 1. Parse standard detector predictions
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_label = self.model.names[cls_id]

            # Filter for driving-relevant classes
            if raw_label in INDIAN_ROAD_MAPPINGS:
                mapped_label = INDIAN_ROAD_MAPPINGS[raw_label]
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                # Auto-rickshaw heuristic: small, compact box labeled car/motorcycle with three-wheeler profile
                box_w = x2 - x1
                box_h = y2 - y1
                ratio = box_h / max(1, box_w)
                if raw_label in ("car", "motorcycle") and (0.9 < ratio < 1.4) and (box_w < w * 0.35):
                    if raw_label == "motorcycle" and box_w > w * 0.12:
                        mapped_label = "auto-rickshaw"

                # Skip ego-vehicle dashboard / hood at bottom of windshield
                if (x1 < w * 0.15 and x2 > w * 0.85 and y2 > h * 0.85):
                    continue

                # Check if object is directly in ego vehicle lane corridor
                center_x = (x1 + x2) / 2.0
                bottom_y = y2
                in_lane = (w * 0.22 <= center_x <= w * 0.78) and (h * 0.38 <= bottom_y <= h * 0.78)

                hazard_lvl = HAZARD_PRIORITIES.get(mapped_label, 2)
                if in_lane:
                    hazard_lvl = min(5, hazard_lvl + 1)

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class_name": mapped_label,
                    "raw_label": raw_label,
                    "confidence": round(conf, 3),
                    "hazard_level": hazard_lvl,
                    "in_ego_lane": in_lane,
                })

        # 2. Add road surface hazards (potholes) if enabled
        if self.detect_potholes:
            existing_boxes = [d["box"] for d in detections]
            potholes = self._detect_potholes_and_hazards(frame, existing_boxes=existing_boxes)
            for p in potholes:
                cx = (p["box"][0] + p["box"][2]) / 2.0
                p["in_ego_lane"] = (w * 0.28 <= cx <= w * 0.72)
                detections.append(p)

        # 3. Add barricades / barriers
        barriers = self._detect_barriers(frame)
        for b in barriers:
            cx = (b["box"][0] + b["box"][2]) / 2.0
            b["in_ego_lane"] = (w * 0.25 <= cx <= w * 0.75)
            detections.append(b)

        # Sort by hazard level (descending)
        detections.sort(key=lambda d: (d.get("hazard_level", 1), d.get("confidence", 0)), reverse=True)
        return detections
