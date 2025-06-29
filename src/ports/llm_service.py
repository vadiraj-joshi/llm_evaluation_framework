
from abc import ABC, abstractmethod
from domain.models import EvaluationMetric, EvaluationResult

class ILLMService(ABC):
    @abstractmethod
    def get_llm_response(self, prompt: str, model_name: str) -> str:
        pass

# ports/metrics_calculator.py
class IMetricsCalculator(ABC):
    @abstractmethod
    def calculate_score(self, expected_output: str, llm_output: str) -> EvaluationResult:
        pass

    @abstractmethod
    def supports_metric_type(self, metric_type: EvaluationMetric) -> bool:
        pass