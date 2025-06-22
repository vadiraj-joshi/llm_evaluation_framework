from rouge_score import rouge_scorer
from typing import List

from ports.metrics_calculator import IMetricsCalculator
from domain.models import EvaluationResult, EvaluationMetric

class SummarizationMetrics(IMetricsCalculator):
    """
    Metrics calculator specifically for summarization tasks, using ROUGE-L.
    Now considered a core domain capability.
    """
    def __init__(self):
        """
        Initializes the SummarizationMetrics calculator using ROUGE-L.
        """
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_score(self, expected_output: str, llm_output: str) -> EvaluationResult:
        """
        Calculates ROUGE-L F-measure score for summarization.
        :param expected_output: The reference summary.
        :param llm_output: The generated summary by the LLM.
        :return: An EvaluationResult containing ROUGE-L score and details.
        """
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
        """
        Checks if the calculator supports the given metric type.
        :param metric_type: The metric to check.
        :return: True if ROUGE_L is supported, False otherwise.
        """
        return metric_type == EvaluationMetric.ROUGE_L
