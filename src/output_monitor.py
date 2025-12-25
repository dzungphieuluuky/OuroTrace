import torch
import time
import re
from typing import List, Optional, Dict, Any
from difflib import SequenceMatcher
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from tqdm.auto import tqdm


class ExperimentFailureException(Exception):
    """Custom exception for experiment failures"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class OutputQualityMonitor:
    """Monitor output quality and detect systematic failures"""

    def __init__(
        self,
        garbage_threshold: float = 0.3,  # 30% garbage outputs
        example_similarity_threshold: float = 0.85,  # 85% similarity to example
        min_samples_before_check: int = 10,
        window_size: int = 20,
    ):
        self.garbage_threshold = garbage_threshold
        self.example_similarity_threshold = example_similarity_threshold
        self.min_samples_before_check = min_samples_before_check
        self.window_size = window_size

        # Tracking metrics
        self.results_buffer = []
        self.example_responses = {}
        self.failure_samples = []

    def set_example_response(self, task_type: str, example_text: str):
        """Store example response for similarity checking"""
        self.example_responses[task_type] = example_text.strip()

    def add_result(self, result: Dict[str, Any], task_type: str) -> None:
        """Add a result to the monitoring buffer"""
        self.results_buffer.append(
            {
                "task_type": task_type,
                "is_degenerate": result.get("is_degenerate", False),
                "full_response": result.get("full_response", ""),
                "prediction": result.get("prediction", ""),
                "test_input": result.get("test_input", ""),
                "expected_answer": result.get("expected_answer", ""),
            }
        )

        # Keep buffer at window size
        if len(self.results_buffer) > self.window_size * 3:
            self.results_buffer = self.results_buffer[-self.window_size * 2 :]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts"""
        return SequenceMatcher(None, text1, text2).ratio()

    def _check_example_memorization(self) -> Dict[str, Any]:
        """Check if model is just copying examples"""
        if len(self.results_buffer) < self.min_samples_before_check:
            return {"is_memorizing": False}

        recent_results = self.results_buffer[-self.window_size :]
        memorization_cases = []

        for result in recent_results:
            task_type = result["task_type"]
            response = result["full_response"].strip()

            if task_type in self.example_responses:
                example = self.example_responses[task_type]
                similarity = self._calculate_similarity(response, example)

                if similarity >= self.example_similarity_threshold:
                    memorization_cases.append(
                        {
                            "task_type": task_type,
                            "similarity": similarity,
                            "response": response[:200],
                            "example": example[:200],
                            "test_input": result["test_input"][:100],
                        }
                    )

        memorization_rate = len(memorization_cases) / len(recent_results)

        return {
            "is_memorizing": memorization_rate > 0.5,  # >50% are memorized
            "memorization_rate": memorization_rate,
            "cases": memorization_cases[:5],  # Keep top 5 examples
        }

    def _check_garbage_rate(self) -> Dict[str, Any]:
        """Check if garbage output rate exceeds threshold"""
        if len(self.results_buffer) < self.min_samples_before_check:
            return {"is_excessive_garbage": False}

        recent_results = self.results_buffer[-self.window_size :]
        garbage_count = sum(1 for r in recent_results if r["is_degenerate"])
        garbage_rate = garbage_count / len(recent_results)

        garbage_samples = [
            {
                "task_type": r["task_type"],
                "response": r["full_response"][:300],
                "test_input": r["test_input"][:100],
            }
            for r in recent_results
            if r["is_degenerate"]
        ][:5]

        return {
            "is_excessive_garbage": garbage_rate > self.garbage_threshold,
            "garbage_rate": garbage_rate,
            "garbage_count": garbage_count,
            "total_count": len(recent_results),
            "samples": garbage_samples,
        }

    def check_failure_conditions(self) -> Optional[ExperimentFailureException]:
        """Check if experiment should be terminated due to quality issues"""

        if len(self.results_buffer) < self.min_samples_before_check:
            return None

        # Check for garbage outputs
        garbage_check = self._check_garbage_rate()
        if garbage_check["is_excessive_garbage"]:
            failure_stats = {
                "failure_type": "EXCESSIVE_GARBAGE",
                "garbage_rate": f"{garbage_check['garbage_rate'] * 100:.1f}%",
                "threshold": f"{self.garbage_threshold * 100:.1f}%",
                "garbage_count": garbage_check["garbage_count"],
                "total_samples": garbage_check["total_count"],
                "example_outputs": garbage_check["samples"],
            }
            raise ExperimentFailureException(
                f"EXCESSIVE GARBAGE OUTPUTS: {garbage_check['garbage_rate'] * 100:.1f}% "
                f"(threshold: {self.garbage_threshold * 100:.1f}%)"
            )

        # Check for example memorization
        memorization_check = self._check_example_memorization()
        if memorization_check["is_memorizing"]:
            failure_stats = {
                "failure_type": "EXAMPLE_MEMORIZATION",
                "memorization_rate": f"{memorization_check['memorization_rate'] * 100:.1f}%",
                "threshold": f"{self.example_similarity_threshold * 100:.1f}% similarity",
                "example_cases": memorization_check["cases"],
            }
            raise ExperimentFailureException(
                f"EXAMPLE MEMORIZATION DETECTED: {memorization_check['memorization_rate'] * 100:.1f}% "
                f"of responses are copying examples"
            )

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        if not self.results_buffer:
            return {"total_samples": 0}

        total = len(self.results_buffer)
        garbage = sum(1 for r in self.results_buffer if r["is_degenerate"])

        return {
            "total_samples": total,
            "garbage_count": garbage,
            "garbage_rate": garbage / total if total > 0 else 0,
            "buffer_size": len(self.results_buffer),
        }
