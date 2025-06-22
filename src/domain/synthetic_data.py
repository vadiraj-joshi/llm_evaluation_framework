import uuid
from typing import List
from ports.synthetic_data_generator import ISyntheticDataGenerator
from domain.models import EvaluationData, InputData, ExpectedResult, EvaluationDataType, EvaluationStatus

class BasicSyntheticDataGenerator(ISyntheticDataGenerator):
    """
    A basic synthetic data generator, now considered a core domain capability.
    In a real scenario, this would generate more complex and varied data.
    """
    def generate_synthetic_data(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        """
        Generates a list of simple synthetic EvaluationData instances.
        :param num_samples: Number of samples to generate.
        :param task_name: The task name for which to generate data.
        :return: A list of synthetic EvaluationData objects.
        """
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
        """
        Performs a basic validation check to ensure data instances are of correct type.
        :param data: The list of EvaluationData objects to validate.
        :return: True if all items are EvaluationData instances, False otherwise.
        """
        return all(isinstance(d, EvaluationData) for d in data)
