"""
LayaVision Object Classification & Multi-Question Demo
======================================================
Demonstrates running LayaVision for fast, calibrated vision decisions
on NVIDIA RTX 3050 GPU.
"""

import os
import sys
import time
import torch
from PIL import Image

from layavision import LayaVision
from layavision.dataset import generate_synthetic_image


def run_demo():
    print("=" * 65)
    print(" LayaVision: Fast System 1 Vision Model for Object Classification")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    # 1. Initialize LayaVision model
    print("\n[1] Initializing LayaVision Classifier...")
    t_init = time.perf_counter()
    classifier = LayaVision(device=device)
    print(f"Model loaded in {(time.perf_counter() - t_init):.2f}s")
    if device == "cuda":
        print(f"GPU VRAM in use: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    # 2. Generate sample images for testing
    os.makedirs("sample_images", exist_ok=True)
    test_cases = [
        ("sports car", "sample_images/car.png"),
        ("golden retriever dog", "sample_images/dog.png"),
        ("tabby cat", "sample_images/cat.png"),
        ("red apple fruit", "sample_images/apple.png"),
    ]

    for label, path in test_cases:
        img = generate_synthetic_image(label)
        img.save(path)

    # 3. Demonstration A: Open-Vocabulary Object Classification
    print("\n" + "=" * 65)
    print(" [DEMO A] Open-Vocabulary Object Classification with Calibrated Confidence")
    print("=" * 65)

    candidate_classes = [
        "car",
        "dog",
        "cat",
        "apple",
        "airplane",
        "bicycle",
        "boat",
        "laptop",
    ]

    criteria = {
        "car": "motor vehicle with wheels and windshield",
        "dog": "canine animal with fur and snout",
        "cat": "feline animal with pointy ears and whiskers",
        "apple": "round red fruit with a stem",
        "airplane": "aircraft with wings and jet engines",
        "bicycle": "two-wheeled pedal-driven cycle",
        "boat": "watercraft floating on water",
        "laptop": "portable computer with keyboard and screen",
    }

    for label, path in test_cases:
        res = classifier.classify(
            image=path,
            candidate_classes=candidate_classes,
            criteria=criteria,
            top_k=3,
        )

        print(f"\nTarget Image : {label.upper()} ({path})")
        print(f"Top Class    : {res['top_class']}")
        print(f"Confidence   : {res['confidence'] * 100:.1f}% (Normalized Entropy)")
        print(f"Latency      : {res['latency_ms']:.2f} ms")
        print("Top 3 Probabilities:")
        for cls_name, prob in res["probabilities"].items():
            bar = "#" * int(prob * 30)
            print(f"  - {cls_name:<10}: {prob * 100:5.1f}% | {bar}")

    # 4. Demonstration B: Full Laya Typed Decision Schema
    print("\n" + "=" * 65)
    print(" [DEMO B] Multi-Question Typed Visual Inspection (Choice, Score, Noul)")
    print("=" * 65)

    # Let's inspect the car image with multiple questions in parallel
    inspection_questions = {
        "primary_entity": {
            "type": "choice",
            "instructions": "What is the primary subject in the image?",
            "criteria": {
                "vehicle": "automobile, truck, or cycle",
                "animal": "living creature like cat or dog",
                "food": "edible item like fruit or dish",
                "other": "everything else",
            },
        },
        "visual_condition": {
            "type": "score",
            "instructions": "Rate the visible wear or damage level of the object",
            "criteria": [
                "pristine, clean and like new",
                "minor wear or light surface scratches",
                "severe damage, broken or defective",
            ],
        },
        "is_living_organism": {
            "type": "noul",
            "instructions": "Is the object in the image a living biological creature?",
            "criteria": {
                "true": "a living biological animal, creature, or human",
                "false": "an inanimate mechanical machine, object, or vehicle",
            },
        },
        "is_transportation": {
            "type": "noul",
            "instructions": "Is the object used for transportation or moving people?",
            "criteria": {
                "true": "a transportation vehicle such as a car, bike, or bus",
                "false": "not a vehicle for transportation",
            },
        },
    }

    test_img = "sample_images/car.png"
    multi_res = classifier.predict(test_img, inspection_questions)

    print(f"\nInspection Results for '{test_img}':")
    print(f"Total Latency: {multi_res['total_latency_ms']:.2f} ms across {multi_res['num_questions']} questions")

    answers = multi_res["answers"]
    print(f"1. Primary Entity : {answers['primary_entity']['choice']} (conf: {answers['primary_entity']['confidence']*100:.1f}%)")
    print(f"2. Visual Condition: Score {answers['visual_condition']['score']:.2f} / 2.0 (conf: {answers['visual_condition']['confidence']*100:.1f}%)")
    print(f"3. Living Organism : {'YES' if answers['is_living_organism']['decision'] else 'NO'} (prob: {answers['is_living_organism']['probability']*100:.1f}%)")
    print(f"4. Transportation  : {'YES' if answers['is_transportation']['decision'] else 'NO'} (prob: {answers['is_transportation']['probability']*100:.1f}%)")

    # 5. Demonstration C: Latency Benchmark
    print("\n" + "=" * 65)
    print(" [DEMO C] GPU Inference Speed Benchmark (10 Warm Iterations)")
    print("=" * 65)

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = classifier.classify(
            image="sample_images/dog.png",
            candidate_classes=candidate_classes,
            top_k=1,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    print(f"Average Latency: {avg_lat:.2f} ms")
    print(f"Fastest Latency: {min_lat:.2f} ms")
    print(f"Throughput     : {1000.0 / avg_lat:.1f} images/sec")
    print("=" * 65)
    print(" LayaVision demo completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
