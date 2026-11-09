"""
Laya Demo & Test Script
======================
This script demonstrates how to use Laya (a non-autoregressive System 1 decision engine)
for ultra-fast, calibrated classification, scoring, and binary decision-making.
"""

import sys
import time

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch is not yet installed.")

try:
    import laya
    from laya import Router

    print(f"Laya version: {getattr(laya, '__version__', 'installed')}")

    # Set device to cuda if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Laya Router on {device.upper()}...")

    # Router detects language and dispatches to English or Multilingual checkpoint
    router = Router(preload=True, device=device)

    # 1. Example state (Customer Support Email / Ticket)
    sample_state = {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate amount today or we will cancel our subscription."
    }

    # 2. Define typed decision questions
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "criteria": {
                "billing": "invoices, payments, refunds, duplicate charges",
                "technical": "bugs, outages, system errors, login problems",
                "sales": "pricing inquiries, enterprise quotes, new contracts",
                "other": "general inquiries and everything else"
            }
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": [
                "not urgent",
                "moderate priority",
                "critical deadline or blocking issue"
            ]
        },
        "churn_risk": {
            "type": "noul",  # Binary yes/no/neutral decision
            "instructions": "Does the customer threaten to cancel or leave?"
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer explicitly request a refund?"
        }
    }

    # 3. Predict English state
    start = time.perf_counter()
    res_en = router.predict(sample_state, questions)
    elapsed_en = (time.perf_counter() - start) * 1000

    print("\n--- [English Result] ---")
    print(f"Inference Time : {elapsed_en:.2f} ms")
    print(f"Model Routed   : {res_en.get('routing', {}).get('model')}")
    print(f"Department     : {res_en['answers']['department']}")
    print(f"Urgency        : {res_en['answers']['urgency']}")
    print(f"Churn Risk     : {res_en['answers']['churn_risk']}")
    print(f"Refund Request : {res_en['answers']['refund_requested']}")

    # 4. Predict Multilingual state (Hindi)
    hindi_state = {
        "body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"
    }
    start = time.perf_counter()
    res_hi = router.predict(hindi_state, questions)
    elapsed_hi = (time.perf_counter() - start) * 1000

    print("\n--- [Multilingual (Hindi) Result] ---")
    print(f"Inference Time : {elapsed_hi:.2f} ms")
    print(f"Model Routed   : {res_hi.get('routing', {}).get('model')}")
    print(f"Department     : {res_hi['answers']['department']}")

except ImportError as e:
    print(f"Laya package import note: {e}")
    print("Run '.venv\\Scripts\\python.exe example_usage.py' once dependencies finish installing.")
