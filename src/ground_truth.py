"""
Ground Truth Dataset for embedding model evaluation.

Loads questions and relevance annotations from data/ground_truth.json.

Each question maps to a set of chunk IDs with graded relevance:
  - 2: Highly relevant (directly answers the question)
  - 1: Partially relevant (supporting context)
  - 0: Not relevant (default for all other chunks)

Ground truth is generated via expert annotation by reading each chunk
and determining relevance to each question. This avoids circular bias
from using embeddings to generate the ground truth.
"""

import json
import os
from dataclasses import dataclass


@dataclass
class EvalQuestion:
    qid: int
    question: str
    # Maps chunk_id -> relevance grade (2=high, 1=partial)
    # Chunks not in this dict are assumed grade 0
    relevant_chunks: dict[int, int]
    description: str  # Short label for reporting


def _load_ground_truth_json(json_path: str) -> dict:
    """Load the ground truth JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ground_truth(chunks: list, json_path: str = None) -> list[EvalQuestion]:
    """
    Build ground truth from the JSON annotation file.

    The JSON file contains chunk IDs that correspond to the output of the
    chunker. If chunks have been re-generated (e.g., chunking params changed),
    the JSON ground truth should be re-validated.

    Falls back to section-based annotation if JSON is not found.
    """
    if json_path is None:
        json_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "ground_truth.json"
        )

    if os.path.exists(json_path):
        return _build_from_json(json_path)
    else:
        print(f"  WARNING: {json_path} not found, falling back to section-based annotation")
        return _build_from_sections(chunks)


def _build_from_json(json_path: str) -> list[EvalQuestion]:
    """Build ground truth from JSON file."""
    data = _load_ground_truth_json(json_path)
    questions = []

    for q_data in data["questions"]:
        # JSON keys are strings, convert to int chunk_id -> int grade
        relevant = {
            int(chunk_id): info["grade"]
            for chunk_id, info in q_data["relevant_chunks"].items()
        }
        questions.append(EvalQuestion(
            qid=q_data["qid"],
            question=q_data["question"],
            relevant_chunks=relevant,
            description=q_data["description"],
        ))

    return questions


def _build_from_sections(chunks: list) -> list[EvalQuestion]:
    """
    Fallback: annotate ground truth by matching chunk content to questions
    using section IDs and keywords. Used when ground_truth.json is not available.
    """
    # Build a lookup: section_id -> list of chunk_ids
    section_to_chunks: dict[str, list[int]] = {}
    for chunk in chunks:
        sid = chunk.section_id
        if sid not in section_to_chunks:
            section_to_chunks[sid] = []
        section_to_chunks[sid].append(chunk.chunk_id)

    chunk_texts = {c.chunk_id: c.text.lower() for c in chunks}

    raw_questions = [
        (1, "Describe the specific policies and controls available to prevent data loss "
            "and control information flow from the Acme application on unmanaged, "
            "employee-owned mobile devices. Detail how data can be contained within the "
            "application and what actions can be taken if a device is compromised or lost.",
         "Mobile DLP & BYOD controls"),
        (2, "What is Acme's guiding architectural philosophy for security? Describe the "
            "core principles of this architecture, such as how it redefines the security "
            "perimeter and its approach to access control.",
         "Zero Trust architecture"),
        (3, "Our security operations team requires deep integration for monitoring user "
            "and administrative activity. Describe the platform's native capabilities for "
            "exporting detailed audit logs and the specific mechanisms or connectors "
            "provided for integration with enterprise Security Information and Event "
            "Management (SIEM) platforms.",
         "Audit logs & SIEM integration"),
        (4, "For compliance purposes, we must enforce policies that actively prevent "
            "communication between specific user groups. What platform feature allows an "
            "administrator to create such a policy, and what is the user experience for "
            "individuals in groups where this communication control is enforced?",
         "Information Barriers / Ethical Walls"),
        (5, "Beyond the governance of customer data used by AI features, what is Acme's "
            "framework for the responsible development of the AI models themselves? "
            "Specifically, what is your policy regarding the sourcing of training data for "
            "global models, and what governance processes are in place to validate models "
            "for fairness and bias before deployment?",
         "Responsible AI & model governance"),
    ]

    questions = []
    for qid, question, description in raw_questions:
        relevance = {}

        if qid == 1:
            for cid in section_to_chunks.get("15.1", []):
                relevance[cid] = 2
            for cid in section_to_chunks.get("15.2", []):
                relevance[cid] = 2
            for cid in section_to_chunks.get("6.4", []):
                relevance[cid] = 1
            for cid in section_to_chunks.get("15", []):
                if "byod" in chunk_texts.get(cid, "") or "mobile" in chunk_texts.get(cid, ""):
                    relevance[cid] = 1
        elif qid == 2:
            for cid in section_to_chunks.get("12.1", []):
                relevance[cid] = 2
            for cid in section_to_chunks.get("5", []):
                if "defense-in-depth" in chunk_texts.get(cid, ""):
                    relevance[cid] = 1
            for cid in section_to_chunks.get("5.2", []):
                if "access" in chunk_texts.get(cid, ""):
                    relevance[cid] = 1
        elif qid == 3:
            for cid in section_to_chunks.get("13.3", []):
                relevance[cid] = 2
            for cid in section_to_chunks.get("12.2", []):
                if "siem" in chunk_texts.get(cid, ""):
                    relevance[cid] = 1
        elif qid == 4:
            for cid in section_to_chunks.get("13.2", []):
                relevance[cid] = 2
        elif qid == 5:
            for cid in section_to_chunks.get("14.1", []):
                relevance[cid] = 2
            for cid in section_to_chunks.get("14.2", []):
                relevance[cid] = 2
            for cid in section_to_chunks.get("4.4", []):
                relevance[cid] = 1
            for cid in section_to_chunks.get("14", []):
                if "responsible" in chunk_texts.get(cid, "") or "ethical" in chunk_texts.get(cid, ""):
                    relevance[cid] = 1

        questions.append(EvalQuestion(
            qid=qid,
            question=question,
            relevant_chunks=relevance,
            description=description,
        ))

    return questions
