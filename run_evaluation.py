#!/usr/bin/env python3
"""
Entry point for the Embedding Model Evaluation System.

Usage:
    python run_evaluation.py
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluate import run_evaluation


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(project_root, "data", "acme_enterprise.txt")
    results_dir = os.path.join(project_root, "results")

    if not os.path.exists(doc_path):
        print(f"ERROR: Document not found at {doc_path}")
        sys.exit(1)

    results = run_evaluation(doc_path, results_dir)

    print("\nDone! Check results/ directory for detailed output.")


if __name__ == "__main__":
    main()
