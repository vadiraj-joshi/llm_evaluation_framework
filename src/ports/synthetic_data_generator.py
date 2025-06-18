from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.domain.models import EvaluationData
class ISyntheticDataGenerator(ABC):
    @abstractmethod
    def generate_synthetic_data(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        pass

    @abstractmethod
    def validate_synthetic_data(self, data: List[EvaluationData]) -> bool:
        pass