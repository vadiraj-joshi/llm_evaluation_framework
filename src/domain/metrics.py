from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from typing import List, Dict, Optional, Any

from ports.metrics_calculator import IMetricsCalculator
from domain.models import EvaluationResult, EvaluationMetric, EvaluationDataType

# Ensure NLTK data for BLEU is available (punkt tokenizer and averaged_perceptron_tagger for tokenization if needed)
# In a real setup, this would be part of build/deployment steps.
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('taggers/averaged_perceptron_tagger') # Needed for some tokenizers, not strictly BLEU itself
# except nltk.downloader.DownloadError:
#     nltk.download('averaged_perceptron_tagger')


class SummarizationMetrics(IMetricsCalculator):
    """
    Metrics calculator specifically for summarization tasks, using ROUGE-L.
    Now considered a core domain capability.
    """
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_score(self, expected_output: str, llm_output: str, **kwargs) -> EvaluationResult:
        scores = self.scorer.score(expected_output, llm_output)
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
        return metric_type in [EvaluationMetric.ROUGE_L, EvaluationMetric.ROUGE_1, EvaluationMetric.ROUGE_2]

class TranslationMetrics(IMetricsCalculator):
    """
    Calculates BLEU score for translation.
    """
    def __init__(self):
        self.smoothie = SmoothingFunction().method1 # Or method4 for common usage

    def calculate_score(self, expected_output: str, llm_output: str, **kwargs) -> EvaluationResult:
        # BLEU expects lists of tokens, so simple split for demo
        reference = [expected_output.lower().split()]
        candidate = llm_output.lower().split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=self.smoothie)
        return EvaluationResult(
            metric_name=EvaluationMetric.BLEU,
            metric_value=bleu_score,
            details={"smooth_method": "method1"} # Indicate smoothing method used
        )

    def supports_metric_type(self, metric_type: EvaluationMetric) -> bool:
        return metric_type == EvaluationMetric.BLEU

class ClassificationMetrics(IMetricsCalculator):
    """
    Calculates Accuracy, F1-score, Precision, and Recall for classification.
    Assumes `expected_labels` (list of strings) and `llm_output` (single string predicted label).
    """
    def calculate_score(self, expected_output: str, llm_output: str, **kwargs) -> EvaluationResult:
        expected_labels = kwargs.get('expected_labels')
        
        if not expected_labels:
            raise ValueError("expected_labels must be provided for ClassificationMetrics.")

        # For simplicity, assuming LLM outputs a single label that matches one of the expected_labels
        # In a real scenario, you might need more sophisticated parsing/mapping of LLM output.
        predicted_label = llm_output.strip().lower()
        true_label = expected_output.strip().lower() # Assuming expected_output holds the true label as string
        
        # Or, if expected_labels is the canonical list and true_label is derived from it:
        # For simplicity, we compare single predicted label to a single true label.
        # If multi-label classification is needed, this logic needs significant expansion.

        # Convert to binary for scikit-learn metrics for single-label comparison
        # This assumes a binary classification context or one-vs-rest for multi-class
        # A full multi-class setup needs true_labels and predicted_labels to be arrays
        # representing all classes. For demo, we simplify to single label matching.
        is_correct = 1 if predicted_label == true_label else 0
        
        # Create dummy binary arrays for accuracy calculation, assuming a single class scenario
        # This is very simplistic and should be adapted for multi-class/multi-label.
        y_true = [1] if predicted_label == true_label else [0]
        y_pred = [1] # Always predict 1 for correctness comparison here. More robust for multi-class.

        # If comparing against a specific label presence, need to adjust
        # For simple accuracy on a single data point, a boolean match is enough.
        
        # For the F1, Precision, Recall metrics in sklearn, we need arrays.
        # Here's a placeholder if we treat it as binary correct/incorrect for a sample.
        # This part is highly simplified for a single data point.
        # A more complex setup would involve aggregating results for a batch/dataset.
        
        # Simpler approach for a single data point's "metrics"
        accuracy = 1.0 if predicted_label == true_label else 0.0
        
        # F1, Precision, Recall are more meaningful across a dataset.
        # For a single point, they often become 0 or 1, or undefined.
        # We'll use 0.0 if not a match, and 1.0 if match, for demo purposes for single point.
        # This is not how these metrics are typically used for a single sample.
        # A better approach would be to aggregate `y_true` and `y_pred` across the entire dataset.
        
        # Let's adjust to return 0/1 for binary correctness, and indicate if other metrics are not applicable directly per sample
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0) if true_label in expected_labels else 0.0
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0) if true_label in expected_labels else 0.0
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0) if true_label in expected_labels else 0.0

        # We'll report accuracy as the primary for simplicity per data point
        primary_score = accuracy

        return EvaluationResult(
            metric_name=EvaluationMetric.ACCURACY, # Default to Accuracy as primary
            metric_value=primary_score,
            details={
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
        )

    def supports_metric_type(self, metric_type: EvaluationMetric) -> bool:
        return metric_type in [EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE,
                               EvaluationMetric.PRECISION, EvaluationMetric.RECALL]

class RAGMetrics(IMetricsCalculator):
    """
    Conceptual RAG pipeline metrics (Faithfulness, Relevance).
    These often require another LLM (LLM-as-a-judge) or sophisticated NLP techniques.
    For this demo, they are simplified placeholders.
    """
    def calculate_score(self, expected_output: str, llm_output: str, **kwargs) -> EvaluationResult:
        # RAG metrics are complex. For a demo, use dummy values or very simple checks.
        # Faithfulness: Does LLM output only contain information from the provided context?
        # Relevance: Is the LLM output directly relevant to the question/query?
        
        # Dummy example: Assuming expected_output contains keywords that should be in llm_output
        faithfulness_score = 0.8 # Placeholder
        relevance_score = 0.9 # Placeholder

        # Example: Simple check for expected_output presence in llm_output
        is_relevant_simple = 1.0 if expected_output.lower() in llm_output.lower() else 0.0


        return EvaluationResult(
            metric_name=EvaluationMetric.RELEVANCE, # Arbitrarily pick one as primary
            metric_value=relevance_score, # Use dummy score
            details={
                "faithfulness": faithfulness_score,
                "relevance": relevance_score,
                "simple_relevance_check": is_relevant_simple
            }
        )

    def supports_metric_type(self, metric_type: EvaluationMetric) -> bool:
        return metric_type in [EvaluationMetric.FAITHFULNESS, EvaluationMetric.RELEVANCE]


class MetricCalculatorFactory:
    """
    A factory to provide the correct IMetricsCalculator implementation
    based on the AI task name and requested metric type.
    This acts as a central registry within the domain.
    """
    def __init__(self):
        # Register calculators for different tasks
        self._calculators: Dict[str, Dict[EvaluationMetric, IMetricsCalculator]] = {
            "summarization": {
                metric_type: SummarizationMetrics() for metric_type in [
                    EvaluationMetric.ROUGE_L, EvaluationMetric.ROUGE_1, EvaluationMetric.ROUGE_2
                ]
            },
            "translation": {
                EvaluationMetric.BLEU: TranslationMetrics()
            },
            "classification": {
                metric_type: ClassificationMetrics() for metric_type in [
                    EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE,
                    EvaluationMetric.PRECISION, EvaluationMetric.RECALL
                ]
            },
            "rag": {
                metric_type: RAGMetrics() for metric_type in [
                    EvaluationMetric.FAITHFULNESS, EvaluationMetric.RELEVANCE
                ]
            }
            # Add other tasks and their supported metrics here
        }

    def get_calculator(self, ai_task_name: str, metric_type: EvaluationMetric) -> IMetricsCalculator:
        """
        Retrieves the appropriate metric calculator.
        :param ai_task_name: The name of the AI task.
        :param metric_type: The specific metric requested.
        :return: An IMetricsCalculator instance.
        :raises ValueError: If no calculator is found for the given task/metric.
        """
        task_calculators = self._calculators.get(ai_task_name.lower())
        if not task_calculators:
            raise ValueError(f"No metric calculators registered for AI task: {ai_task_name}")
        
        calculator = task_calculators.get(metric_type)
        if not calculator:
            raise ValueError(f"Metric type {metric_type.value} not supported for task {ai_task_name}.")
        
        return calculator

