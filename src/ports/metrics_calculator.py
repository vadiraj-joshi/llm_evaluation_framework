from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.domain.models import EvaluationResult, EvaluationMetric
class IMetricsCalculator(ABC):
    @abstractmethod
    def calculate_score(self, expected_output: str, llm_output: str) -> EvaluationResult:
        pass

    @abstractmethod
    def supports_metric_type(self, metric_type: EvaluationMetric) -> bool:
        pass