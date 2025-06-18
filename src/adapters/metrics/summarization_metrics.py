from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from rouge_score import rouge_scorer
from src.ports.metrics_calculator import IMetricsCalculator
from src.domain.models import EvaluationResult, EvaluationMetric

class SummarizationMetrics(IMetricsCalculator):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_score(self, expected_output: str, llm_output: str) -> EvaluationResult:
        scores = self.scorer.score(expected_output, llm_output)
        # For simplicity, we'll use ROUGE-L F-measure as the primary metric
        rouge_l_score = scores['rougeL'].fmeasure
        return EvaluationResult(
            metric_name=EvaluationMetric.ROUGE_L,
            metric_value=rouge_l_score,
            details={
                "rouge1_fmeasure": scores['rouge1'].fmeasure,
                "rouge2_fmeasure": scores['rouge2'].fmeasure,
                "rougeL_fmeasure": rouge_l_score
            }
        )

    def supports_metric_type(self, metric_type: EvaluationMetric) -> bool:
        return metric_type in [EvaluationMetric.ROUGE_L, EvaluationMetric.BLEU] # BLEU can be added later