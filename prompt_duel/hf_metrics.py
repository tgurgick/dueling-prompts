#!/usr/bin/env python3
"""
Hugging Face Evaluation Metrics Integration
"""

import re
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np

# Import the base Metric class
from prompt_duel.metrics import Metric


class BLEUMetric(Metric):
    """BLEU (Bilingual Evaluation Understudy) scoring for text generation."""
    
    def __init__(self, **kwargs):
        try:
            from evaluate import load
            self.metric = load("bleu")
            self.metric_name = "bleu"
        except ImportError:
            print("Warning: evaluate not installed. Install with: pip install evaluate")
            self.metric = None
    
    def score(self, response: str, expected: str) -> float:
        """Score a single response against expected output using BLEU."""
        if self.metric is None:
            return 0.5  # Fallback score
        
        try:
            result = self.metric.compute(
                predictions=[response], 
                references=[expected]
            )
            return float(result[self.metric_name])
        except Exception as e:
            print(f"Warning: BLEU calculation failed: {e}")
            return 0.5
    
    def score_comparative(self, response_a: str, response_b: str, expected: str) -> Tuple[float, float]:
        """Score two responses comparatively using BLEU."""
        score_a = self.score(response_a, expected)
        score_b = self.score(response_b, expected)
        return score_a, score_b


class ROUGEMetric(Metric):
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scoring."""
    
    def __init__(self, rouge_type: str = "rouge1", **kwargs):
        try:
            from evaluate import load
            self.metric = load("rouge")
            self.rouge_type = rouge_type
        except ImportError:
            print("Warning: evaluate not installed. Install with: pip install evaluate")
            self.metric = None
    
    def score(self, response: str, expected: str) -> float:
        """Score a single response against expected output using ROUGE."""
        if self.metric is None:
            return 0.5  # Fallback score
        
        try:
            result = self.metric.compute(
                predictions=[response], 
                references=[expected]
            )
            # ROUGE returns simple dict with float values
            score = float(result[self.rouge_type])
            return score
        except Exception as e:
            print(f"Warning: ROUGE calculation failed: {e}")
            return 0.5
    
    def score_comparative(self, response_a: str, response_b: str, expected: str) -> Tuple[float, float]:
        """Score two responses comparatively using ROUGE."""
        score_a = self.score(response_a, expected)
        score_b = self.score(response_b, expected)
        return score_a, score_b


class METEORMetric(Metric):
    """METEOR (Metric for Evaluation of Translation with Explicit ORdering) scoring."""
    
    def __init__(self, **kwargs):
        try:
            from evaluate import load
            self.metric = load("meteor")
            self.metric_name = "meteor"
        except ImportError:
            print("Warning: evaluate not installed. Install with: pip install evaluate")
            self.metric = None
    
    def score(self, response: str, expected: str) -> float:
        """Score a single response against expected output using METEOR."""
        if self.metric is None:
            return 0.5  # Fallback score
        
        try:
            result = self.metric.compute(
                predictions=[response], 
                references=[expected]
            )
            return float(result[self.metric_name])
        except Exception as e:
            print(f"Warning: METEOR calculation failed: {e}")
            return 0.5
    
    def score_comparative(self, response_a: str, response_b: str, expected: str) -> Tuple[float, float]:
        """Score two responses comparatively using METEOR."""
        score_a = self.score(response_a, expected)
        score_b = self.score(response_b, expected)
        return score_a, score_b


class BERTScoreMetric(Metric):
    """BERTScore evaluation using BERT embeddings."""
    
    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli", **kwargs):
        try:
            from evaluate import load
            self.metric = load("bertscore")
            self.model_type = model_type
        except ImportError:
            print("Warning: evaluate not installed. Install with: pip install evaluate")
            self.metric = None
    
    def score(self, response: str, expected: str) -> float:
        """Score a single response against expected output using BERTScore."""
        if self.metric is None:
            return 0.5  # Fallback score
        
        try:
            result = self.metric.compute(
                predictions=[response], 
                references=[expected],
                model_type=self.model_type
            )
            # Return the F1 score (harmonic mean of precision and recall)
            return float(result["f1"][0])
        except Exception as e:
            print(f"Warning: BERTScore calculation failed: {e}")
            return 0.5
    
    def score_comparative(self, response_a: str, response_b: str, expected: str) -> Tuple[float, float]:
        """Score two responses comparatively using BERTScore."""
        score_a = self.score(response_a, expected)
        score_b = self.score(response_b, expected)
        return score_a, score_b


class BLEURTMetric(Metric):
    """BLEURT (BLEU + BERT) evaluation metric."""
    
    def __init__(self, **kwargs):
        try:
            from evaluate import load
            self.metric = load("bleurt")
            self.metric_name = "bleurt"
        except ImportError:
            print("Warning: evaluate not installed. Install with: pip install evaluate")
            self.metric = None
    
    def score(self, response: str, expected: str) -> float:
        """Score a single response against expected output using BLEURT."""
        if self.metric is None:
            return 0.5  # Fallback score
        
        try:
            result = self.metric.compute(
                predictions=[response], 
                references=[expected]
            )
            return float(result[self.metric_name])
        except Exception as e:
            print(f"Warning: BLEURT calculation failed: {e}")
            return 0.5
    
    def score_comparative(self, response_a: str, response_b: str, expected: str) -> Tuple[float, float]:
        """Score two responses comparatively using BLEURT."""
        score_a = self.score(response_a, expected)
        score_b = self.score(response_b, expected)
        return score_a, score_b


class TERMetric(Metric):
    """TER (Translation Edit Rate) evaluation metric."""
    
    def __init__(self, **kwargs):
        try:
            from evaluate import load
            self.metric = load("ter")
            self.metric_name = "ter"
        except ImportError:
            print("Warning: evaluate not installed. Install with: pip install evaluate")
            self.metric = None
    
    def score(self, response: str, expected: str) -> float:
        """Score a single response against expected output using TER."""
        if self.metric is None:
            return 0.5  # Fallback score
        
        try:
            result = self.metric.compute(
                predictions=[response], 
                references=[expected]
            )
            # TER is a distance metric (lower is better), so we invert it
            ter_score = float(result[self.metric_name])
            return max(0.0, 1.0 - ter_score)  # Convert to similarity score
        except Exception as e:
            print(f"Warning: TER calculation failed: {e}")
            return 0.5
    
    def score_comparative(self, response_a: str, response_b: str, expected: str) -> Tuple[float, float]:
        """Score two responses comparatively using TER."""
        score_a = self.score(response_a, expected)
        score_b = self.score(response_b, expected)
        return score_a, score_b


# Registry of available Hugging Face metrics
HF_METRICS = {
    "bleu": BLEUMetric,
    "rouge": ROUGEMetric,
    "meteor": METEORMetric,
    "bertscore": BERTScoreMetric,
    "bleurt": BLEURTMetric,
    "ter": TERMetric,
}


def get_hf_metric(metric_name: str, **kwargs) -> Metric:
    """Get a Hugging Face metric by name."""
    if metric_name not in HF_METRICS:
        raise ValueError(f"Unknown Hugging Face metric: {metric_name}")
    
    return HF_METRICS[metric_name](**kwargs)


def list_available_hf_metrics() -> Dict[str, str]:
    """List all available Hugging Face metrics with descriptions."""
    return {
        "bleu": "Bilingual Evaluation Understudy - measures n-gram overlap",
        "rouge": "Recall-Oriented Understudy for Gisting Evaluation - measures word overlap",
        "meteor": "Metric for Evaluation of Translation with Explicit ORdering - considers synonyms",
        "bertscore": "BERT-based evaluation - uses contextual embeddings",
        "bleurt": "BLEU + BERT - combines n-gram overlap with BERT embeddings",
        "ter": "Translation Edit Rate - measures edit distance",
    } 