"""
Synthetic Indian Road Driving Video Generator
=============================================
Generates a realistic road driving video sequence for testing
autonomous driving perception: cows, stray dogs, auto-rickshaws,
potholes, pedestrians, and barricades.
"""

import math
import os
from typing import Tuple
import cv2
import numpy as np


def create_indian_road_test_video(
    output_path: str = "sample_indian_road_drive.mp4",
    num_frames: int = 150,
    fps: int = 30,
    size: Tuple[int, int] = (854, 480),
) -> str:
    """
    Generate a dynamic road driving clip simulating Indian road conditions.
    """
    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Synthesizing Indian Road Driving test clip: {num_frames} frames ({output_path})...")

    horizon_y = int(h * 0.45)

    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Sky & Environment
        frame[0:horizon_y, :] = (210, 180, 140)  # Dusty daytime Indian sky
        cv2.rectangle(frame, (0, horizon_y - 30), (w, horizon_y), (80, 130, 90), -1)  # Distant foliage / trees

        # 2. Road surface (asphalt with texture)
        road_pts = np.array([
            [int(w * 0.38), horizon_y],
            [int(w * 0.62), horizon_y],
            [int(w * 0.95), h],
            [int(w * 0.05), h],
        ], np.int32)
        cv2.fillPoly(frame, [road_pts], (70, 72, 75))

        # Road shoulders (dusty dirt shoulders)
        left_shoulder = np.array([[0, horizon_y], [int(w * 0.38), horizon_y], [int(w * 0.05), h], [0, h]], np.int32)
        right_shoulder = np.array([[int(w * 0.62), horizon_y], [w, horizon_y], [w, h], [int(w * 0.95), h]], np.int32)
        cv2.fillPoly(frame, [left_shoulder], (90, 120, 145))
        cv2.fillPoly(frame, [right_shoulder], (90, 120, 145))

        # 3. Dynamic Center Lane Markings (moving forward)
        lane_offset = (i * 12) % 60
        for y_dash in range(horizon_y + lane_offset, h, 60):
            t = (y_dash - horizon_y) / float(h - horizon_y)
            dash_w = int(2 + t * 8)
            dash_h = int(12 + t * 24)
            dash_x = int(w * 0.5 - dash_w / 2)
            cv2.rectangle(frame, (dash_x, y_dash), (dash_x + dash_w, min(h, y_dash + dash_h)), (220, 220, 220), -1)

        # 4. Indian Road Hazard 1: Pothole approaching in ego lane
        # Moves closer to ego vehicle as frame progresses
        pothole_t = min(1.0, (i % 80) / 70.0)
        py = int(horizon_y + 40 + pothole_t * (h - horizon_y - 90))
        px = int(w * 0.44 + (pothole_t * 0.04 * w))
        pw = int(16 + pothole_t * 54)
        ph = int(8 + pothole_t * 22)
        # Draw dark jagged pothole with cracked rim
        cv2.ellipse(frame, (px + pw // 2, py + ph // 2), (pw // 2, ph // 2), 0, 0, 360, (25, 25, 28), -1)
        cv2.ellipse(frame, (px + pw // 2, py + ph // 2), (pw // 2 + 2, ph // 2 + 2), 0, 0, 360, (50, 52, 55), 2)

        # 5. Indian Road Hazard 2: Cow (cattle) crossing or standing near center lane
        # Visible from frame 20 onwards
        if i >= 15:
            cow_progress = min(1.0, (i - 15) / 100.0)
            cow_y = int(horizon_y + 20 + cow_progress * (h * 0.35))
            cow_x = int(w * 0.62 - cow_progress * (w * 0.12))  # Slowly wandering into driving path
            cow_w = int(35 + cow_progress * 85)
            cow_h = int(25 + cow_progress * 65)

            # Cow body (white / light grey zebu cattle)
            cv2.ellipse(frame, (cow_x + int(cow_w * 0.5), cow_y + int(cow_h * 0.55)), (int(cow_w * 0.42), int(cow_h * 0.32)), 0, 0, 360, (220, 220, 225), -1)
            # Cow head & hump
            cv2.circle(frame, (cow_x + int(cow_w * 0.18), cow_y + int(cow_h * 0.35)), int(cow_h * 0.22), (230, 230, 235), -1)
            cv2.ellipse(frame, (cow_x + int(cow_w * 0.38), cow_y + int(cow_h * 0.28)), (int(cow_w * 0.12), int(cow_h * 0.14)), -20, 0, 360, (210, 210, 215), -1)
            # Horns
            cv2.line(frame, (cow_x + int(cow_w * 0.15), cow_y + int(cow_h * 0.25)), (cow_x + int(cow_w * 0.10), cow_y + int(cow_h * 0.12)), (80, 80, 80), 2)
            cv2.line(frame, (cow_x + int(cow_w * 0.22), cow_y + int(cow_h * 0.25)), (cow_x + int(cow_w * 0.26), cow_y + int(cow_h * 0.12)), (80, 80, 80), 2)
            # Legs
            for lx in [0.25, 0.40, 0.65, 0.78]:
                cv2.line(frame, (cow_x + int(cow_w * lx), cow_y + int(cow_h * 0.70)), (cow_x + int(cow_w * lx), cow_y + cow_h), (180, 180, 185), 3)

        # 6. Indian Road Hazard 3: Stray Dog on roadside / shoulder
        dog_x = int(w * 0.18 + math.sin(i * 0.1) * 8)
        dog_y = int(h * 0.68)
        cv2.ellipse(frame, (dog_x + 18, dog_y + 12), (16, 10), 0, 0, 360, (60, 120, 190), -1)  # brown fur
        cv2.circle(frame, (dog_x + 32, dog_y + 6), 7, (70, 130, 200), -1)
        cv2.line(frame, (dog_x + 30, dog_y + 14), (dog_x + 30, dog_y + 24), (50, 100, 170), 2)
        cv2.line(frame, (dog_x + 12, dog_y + 14), (dog_x + 12, dog_y + 24), (50, 100, 170), 2)

        # 7. Indian Road Object 4: Auto-rickshaw (three-wheeler) ahead on right
        auto_y = int(horizon_y + 35 + (i * 0.8))
        auto_x = int(w * 0.68)
        aw, ah = 55, 48
        # Yellow and green body
        cv2.rectangle(frame, (auto_x, auto_y + 18), (auto_x + aw, auto_y + ah), (40, 140, 40), -1)  # Green bottom
        cv2.rectangle(frame, (auto_x, auto_y), (auto_x + aw, auto_y + 18), (30, 210, 240), -1)     # Yellow top
        # Black windshield / canvas canopy
        cv2.rectangle(frame, (auto_x + 4, auto_y + 6), (auto_x + aw - 4, auto_y + 24), (30, 30, 30), -1)
        # Rear wheels
        cv2.circle(frame, (auto_x + 8, auto_y + ah), 6, (20, 20, 20), -1)
        cv2.circle(frame, (auto_x + aw - 8, auto_y + ah), 6, (20, 20, 20), -1)

        # 8. Indian Road Hazard 5: Barricade / Road Barrier
        if i >= 60:
            barr_y = int(h * 0.58)
            barr_x = int(w * 0.76)
            bw, bh = 70, 32
            # Yellow and black stripes
            cv2.rectangle(frame, (barr_x, barr_y), (barr_x + bw, barr_y + bh), (20, 220, 240), -1)
            for stripe_x in range(barr_x, barr_x + bw, 16):
                pts = np.array([[stripe_x, barr_y], [stripe_x + 8, barr_y], [stripe_x + 4, barr_y + bh], [stripe_x - 4, barr_y + bh]], np.int32)
                cv2.fillPoly(frame, [pts], (25, 25, 25))
            # Stand legs
            cv2.line(frame, (barr_x + 6, barr_y + bh), (barr_x + 6, barr_y + bh + 14), (40, 40, 40), 3)
            cv2.line(frame, (barr_x + bw - 6, barr_y + bh), (barr_x + bw - 6, barr_y + bh + 14), (40, 40, 40), 3)

        writer.write(frame)

    writer.release()
    print(f"Test video created successfully: {output_path}")
    return output_path
