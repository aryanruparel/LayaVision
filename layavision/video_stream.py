"""
LayaVision Video Stream Processor & HUD Renderer
================================================
Processes video feeds (webcam, video files, or synthesized driving streams),
detects Indian road objects and hazards, and renders real-time bounding boxes
and autonomous driving HUD telemetry.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from .detector import IndianRoadDetector
from .driving_agent import LayaDrivingAgent


# Color palette (BGR for OpenCV)
COLOR_CRITICAL_HAZARD = (40, 40, 240)    # Crimson Red (Cows, Pedestrians in path)
COLOR_MODERATE_HAZARD = (20, 140, 255)   # Amber / Orange (Dogs, Potholes, Barriers)
COLOR_CAUTION = (30, 220, 240)           # Golden Yellow (Auto-rickshaws, Bikes)
COLOR_TRAFFIC = (60, 220, 60)            # Vivid Green (Normal cars, clear road)
COLOR_HUD_BG = (20, 24, 28)              # Dark Slate for HUD


class VideoStreamProcessor:
    """
    Real-time video pipeline combining IndianRoadDetector + LayaDrivingAgent.
    """

    def __init__(
        self,
        detector: Optional[IndianRoadDetector] = None,
        driving_agent: Optional[LayaDrivingAgent] = None,
    ):
        self.detector = detector or IndianRoadDetector()
        self.driving_agent = driving_agent or LayaDrivingAgent()

    def _get_box_color(self, detection: Dict[str, Any]) -> Tuple[int, int, int]:
        """Assign color based on hazard level and lane positioning."""
        hazard_lvl = detection.get("hazard_level", 2)
        in_lane = detection.get("in_ego_lane", False)

        if hazard_lvl >= 5 or (hazard_lvl >= 4 and in_lane):
            return COLOR_CRITICAL_HAZARD
        elif hazard_lvl >= 4:
            return COLOR_MODERATE_HAZARD
        elif hazard_lvl >= 3:
            return COLOR_CAUTION
        return COLOR_TRAFFIC

    def _draw_hud(
        self,
        frame: np.ndarray,
        decision: Dict[str, Any],
        fps: float,
        detection_time_ms: float,
        detections: List[Dict[str, Any]],
    ):
        """Render autonomous driving telemetry and status HUD overlay."""
        h, w = frame.shape[:2]

        # 1. Top HUD Banner (semi-transparent glassmorphism bar)
        overlay = frame.copy()
        hud_h = 75
        cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Bottom border for HUD
        cv2.line(frame, (0, hud_h), (w, hud_h), (80, 95, 110), 1)

        # 2. System Title & FPS (Left)
        cv2.putText(frame, "LAYAVISION AI DRIVE", (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:4.1f} | Latency: {detection_time_ms:4.1f}ms", (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 200, 210), 1)

        # 3. Driving Decision & Warning Banner (Center)
        action = decision.get("action", "maintain_speed")
        banner_text = decision.get("warning_banner", "PATH CLEAR")
        risk_score = decision.get("collision_risk_score", 0.0)

        if risk_score >= 2.5:
            banner_bg = COLOR_CRITICAL_HAZARD
            text_color = (255, 255, 255)
        elif risk_score >= 1.5:
            banner_bg = COLOR_MODERATE_HAZARD
            text_color = (20, 20, 20)
        else:
            banner_bg = (30, 80, 40)
            text_color = (240, 255, 240)

        # Center banner badge
        banner_w = int(w * 0.44)
        bx1 = int((w - banner_w) / 2)
        bx2 = bx1 + banner_w
        cv2.rectangle(frame, (bx1, 14), (bx2, 58), banner_bg, -1)
        cv2.rectangle(frame, (bx1, 14), (bx2, 58), (255, 255, 255), 1)

        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
        tx = int(bx1 + (banner_w - tw) / 2)
        cv2.putText(frame, banner_text, (tx, 40), cv2.FONT_HERSHEY_DUPLEX, 0.44, text_color, 1)

        # 4. Detected Objects Breakdown (Right side)
        obj_summary = {}
        for d in detections:
            cls_name = d.get("class_name", "object")
            obj_summary[cls_name] = obj_summary.get(cls_name, 0) + 1

        summary_parts = [f"{k[:10]}:{v}" for k, v in list(obj_summary.items())[:3]]
        summary_str = " | ".join(summary_parts) if summary_parts else "None"
        cv2.putText(frame, f"Hazards: {len(detections)}", (w - 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)
        cv2.putText(frame, summary_str, (w - 200, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 190, 200), 1)

        # 5. Ego-Lane Corridor Overlay
        corridor_overlay = frame.copy()
        lane_pts = np.array([
            [int(w * 0.38), int(h * 0.44)],
            [int(w * 0.62), int(h * 0.44)],
            [int(w * 0.78), int(h * 0.75)],
            [int(w * 0.22), int(h * 0.75)],
        ], np.int32)

        corridor_color = (0, 0, 180) if not decision.get("lane_clear", True) else (0, 180, 80)
        cv2.fillPoly(corridor_overlay, [lane_pts], corridor_color)
        cv2.addWeighted(corridor_overlay, 0.15, frame, 0.85, 0, frame)

        # Lane boundary lines
        cv2.polylines(frame, [lane_pts], isClosed=False, color=corridor_color, thickness=2)

    def _draw_bounding_boxes(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """Draw bounding boxes and labels for detected objects."""
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            label = d["class_name"]
            conf = d.get("confidence", 0.0)
            color = self._get_box_color(d)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add corner styling accents
            corner_len = min(16, (x2 - x1) // 4, (y2 - y1) // 4)
            if corner_len > 4:
                # Top-left
                cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
                # Top-right
                cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
                # Bottom-left
                cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
                # Bottom-right
                cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Label text
            caption = f"{label} {int(conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # Label badge
            by1 = max(0, y1 - th - 8)
            by2 = y1
            cv2.rectangle(frame, (x1, by1), (x1 + tw + 8, by2), color, -1)
            cv2.putText(
                frame,
                caption,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255) if color != COLOR_CAUTION else (0, 0, 0),
                1,
            )

    def process_frame(self, frame: np.ndarray, fps: float = 30.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single video frame: detect objects, evaluate Laya decisions,
        and render bounding boxes and HUD.
        """
        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        detection_ms = (time.perf_counter() - t0) * 1000.0

        decision = self.driving_agent.evaluate_scene(detections, frame.shape)

        annotated_frame = frame.copy()
        self._draw_bounding_boxes(annotated_frame, detections)
        self._draw_hud(annotated_frame, decision, fps, detection_ms, detections)

        result_telemetry = {
            "detections": detections,
            "decision": decision,
            "detection_ms": detection_ms,
            "fps": fps,
        }
        return annotated_frame, result_telemetry

    def process_video(
        self,
        video_source: Union[str, int],
        output_path: Optional[str] = "output_driving_annotated.mp4",
        max_frames: Optional[int] = None,
        display_window: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a video file or webcam stream.

        Args:
            video_source: Path to video file or integer camera index (e.g. 0).
            output_path: Optional destination path to save annotated video.
            max_frames: Max frames to process (None for entire video).
            display_window: Whether to show live cv2.imshow window.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(output_path, fourcc, source_fps, (w, h))

        frame_count = 0
        total_time = 0.0
        decisions_summary = []

        print(f"Streaming video: {w}x{h} @ {source_fps:.1f} FPS (Target Output: {output_path})...")

        try:
            while True:
                if max_frames and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                t_start = time.perf_counter()
                annotated_frame, telemetry = self.process_frame(frame, fps=source_fps)
                dt = time.perf_counter() - t_start
                total_time += dt

                # Compute rolling FPS
                rolling_fps = 1.0 / max(1e-4, dt)
                telemetry["fps"] = rolling_fps
                decisions_summary.append(telemetry["decision"]["action"])

                if out_writer:
                    out_writer.write(annotated_frame)

                if display_window:
                    cv2.imshow("LayaVision Autonomous Drive", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames | Current FPS: {rolling_fps:.1f} | Decision: {telemetry['decision']['action']}")

        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if display_window:
                cv2.destroyAllWindows()

        avg_fps = frame_count / max(1e-4, total_time)
        print(f"\nProcessing Complete!")
        print(f"Total Frames : {frame_count}")
        print(f"Average FPS  : {avg_fps:.1f}")
        if output_path:
            print(f"Output Video : {output_path}")

        return {
            "total_frames": frame_count,
            "average_fps": round(avg_fps, 1),
            "output_path": output_path,
        }
