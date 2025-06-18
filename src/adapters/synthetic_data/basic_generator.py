
from src.domain.models import EvaluationData, EvaluationDataType, ExpectedResult, InputData, EvaluationStatus
from typing import List, Optional, Dict, Any
from src.ports.synthetic_data_generator import ISyntheticDataGenerator
import uuid
class BasicSyntheticDataGenerator(ISyntheticDataGenerator):
    def generate_synthetic_data(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        # This is a placeholder. In a real scenario, this would generate more complex data.
        synthetic_data = []
        for i in range(num_samples):
            input_data = InputData(data_type=EvaluationDataType.TEXT, decoded_data=f"Synthetic input text {i} for {task_name}.")
            expected_result = ExpectedResult(decoded_result=f"Synthetic expected output {i} for {task_name}.")
            eval_data = EvaluationData(
                evaluation_id=str(uuid.uuid4()),
                input_data=input_data,
                expected_result=expected_result,
                status=EvaluationStatus.COMPLETED # Assuming synthetic data is 'pre-evaluated' or for generation
            )
            synthetic_data.append(eval_data)
        return synthetic_data

    def validate_synthetic_data(self, data: List[EvaluationData]) -> bool:
        return all(isinstance(d, EvaluationData) for d in data)