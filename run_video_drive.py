"""
LayaVision Indian Road Autonomous Driving Video Runner
======================================================
Plays real-time video in an interactive window on screen, performing
live object detection, bounding box drawing, and Laya System-1 driving
intelligence simultaneously as the video streams.
"""

import argparse
import os
import sys
import time
import cv2

from layavision.detector import IndianRoadDetector
from layavision.driving_agent import LayaDrivingAgent
from layavision.video_stream import VideoStreamProcessor
from layavision.video_generator import create_indian_road_test_video


def parse_args():
    parser = argparse.ArgumentParser(description="LayaVision Live Video Autonomous Driving Perception")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (defaults to gemini video)")
    parser.add_argument("--camera", type=int, default=None, help="Webcam device ID (e.g. 0)")
    parser.add_argument("--demo", action="store_true", help="Generate and run on synthetic road clip")
    parser.add_argument("--output", type=str, default="output_live_drive_annotated.mp4", help="Output annotated video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to play (default: entire video)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--loop", action="store_true", default=False, help="Loop playback continuously")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print(" LayaVision: Live Autonomous Driving Video Perception & Detection")
    print("=" * 72)

    # 1. Determine video source
    if args.camera is not None:
        video_src = args.camera
        print(f"Source: Live Camera Stream [Device ID: {args.camera}]")
    elif args.video is not None:
        video_src = args.video
        if not os.path.exists(video_src):
            print(f"Error: Video file not found: {video_src}")
            sys.exit(1)
        print(f"Source: Video File [{video_src}]")
    elif args.demo:
        video_src = "sample_indian_road_drive.mp4"
        if not os.path.exists(video_src):
            create_indian_road_test_video(video_src, num_frames=120, fps=30)
        print(f"Source: Simulation Clip [{video_src}]")
    else:
        # Default: Use gemini generated video if available, else synthetic
        if os.path.exists("gemini_generated_video_f3cc846e.mp4"):
            video_src = "gemini_generated_video_f3cc846e.mp4"
            print(f"Source: Gemini Indian Road Driving Video [{video_src}]")
        else:
            video_src = "sample_indian_road_drive.mp4"
            if not os.path.exists(video_src):
                create_indian_road_test_video(video_src, num_frames=120, fps=30)
            print(f"Source: Default Driving Video [{video_src}]")

    # 2. Initialize Detector & Laya Driving Agent
    detector = IndianRoadDetector()
    driving_agent = LayaDrivingAgent(ego_speed_kmh=45.0)
    processor = VideoStreamProcessor(detector=detector, driving_agent=driving_agent)

    # 3. Setup Video Capture
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Failed to open video source: {video_src}")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[Video Specs: {w}x{h} @ {fps:.1f} FPS | Total Frames: {total_video_frames}]")

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # Setup Display Window
    window_name = "LayaVision Live Autonomous Driving (Press Q to Exit, Space to Pause)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Scale window comfortably to fit desktop screen
        display_w = min(1280, w)
        display_h = int(display_w * (h / w))
        cv2.resizeWindow(window_name, display_w, display_h)
        print(f"Opening Live Interactive Video Window: {display_w}x{display_h}...")
        print("Controls: [SPACE] = Pause/Resume | [R] = Restart | [Q] / [ESC] = Quit")

    frame_idx = 0
    total_latency_ms = 0.0
    detected_classes_tally = {}
    actions_tally = {}
    paused = False

    t_start_all = time.perf_counter()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    else:
                        break

                t0 = time.perf_counter()
                annotated_frame, telemetry = processor.process_frame(frame, fps=fps)
                frame_ms = (time.perf_counter() - t0) * 1000.0
                total_latency_ms += frame_ms

                # Write to output file
                writer.write(annotated_frame)

                decision = telemetry["decision"]
                action = decision["action"]
                actions_tally[action] = actions_tally.get(action, 0) + 1

                for d in telemetry["detections"]:
                    cname = d["class_name"]
                    detected_classes_tally[cname] = detected_classes_tally.get(cname, 0) + 1

                frame_idx += 1
                if frame_idx % 24 == 0 or frame_idx == total_video_frames:
                    live_fps = 1000.0 / max(1.0, frame_ms)
                    print(f"Playing Frame {frame_idx:03d}/{total_video_frames} | {live_fps:4.1f} FPS | Action: {action.upper():<16} | Hazards: {len(telemetry['detections'])}")

                if args.max_frames and frame_idx >= args.max_frames:
                    break

            if not args.headless:
                cv2.imshow(window_name, annotated_frame)
                # Synchronize playback timing to match video's native frame rate
                target_wait = max(1, int(1000.0 / fps) - int(frame_ms))
                key = cv2.waitKey(target_wait if not paused else 50) & 0xFF

                if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                    print("\nUser requested exit.")
                    break
                elif key == 32:  # SPACEBAR = Pause / Resume
                    paused = not paused
                    state_str = "PAUSED" if paused else "RESUMING"
                    print(f"\n[{state_str}]")
                elif key in (ord("r"), ord("R")):  # Restart
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    print("\n[Restarting Video from Beginning...]")

    finally:
        cap.release()
        writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    total_time_s = time.perf_counter() - t_start_all
    avg_fps = frame_idx / max(1e-4, total_time_s)
    avg_latency = total_latency_ms / max(1, frame_idx)

    # 4. Display Final Summary
    print("\n" + "=" * 72)
    print(" LayaVision Live Playback Complete")
    print("=" * 72)
    print(f"Total Frames Streamed  : {frame_idx}")
    print(f"Real-Time Speed        : {avg_fps:.1f} FPS (Inference Latency: {avg_latency:.2f} ms/frame)")
    print(f"Annotated Video Saved  : {os.path.abspath(args.output)}")

    print("\n--- [Detected Objects Summary] ---")
    for cname, count in sorted(detected_classes_tally.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cname:<24}: {count} sightings")

    print("\n--- [Laya System-1 Driving Actions Dispatched] ---")
    for act, count in sorted(actions_tally.items(), key=lambda x: x[1], reverse=True):
        pct = (count / max(1, frame_idx)) * 100.0
        bar = "#" * int(pct / 4)
        print(f"  - {act.upper():<20}: {count:3d} frames ({pct:5.1f}%) | {bar}")
    print("=" * 72)


if __name__ == "__main__":
    main()
