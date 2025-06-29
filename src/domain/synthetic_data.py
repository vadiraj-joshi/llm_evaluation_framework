import uuid
from typing import List, Dict
from ports.synthetic_data_generator import ISyntheticDataGenerator
from domain.models import EvaluationData, InputData, ExpectedResult, EvaluationDataType, EvaluationStatus

class BasicSyntheticDataGenerator(ISyntheticDataGenerator):
    """
    A basic synthetic data generator for general text-based tasks.
    """
    def generate_synthetic_data(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        synthetic_data = []
        for i in range(num_samples):
            input_data = InputData(data_type=EvaluationDataType.TEXT, decoded_data=f"Synthetic {task_name} input text {i}.")
            expected_result = ExpectedResult(decoded_result=f"Synthetic {task_name} expected output {i}.")
            eval_data = EvaluationData(
                evaluation_id=str(uuid.uuid4()),
                input_data=input_data,
                expected_result=expected_result,
                status=EvaluationStatus.PENDING # Initial status, will be evaluated later
            )
            synthetic_data.append(eval_data)
        return synthetic_data

    def validate_synthetic_data(self, data: List[EvaluationData]) -> bool:
        return all(isinstance(d, EvaluationData) for d in data)

class TranslationSyntheticDataGenerator(ISyntheticDataGenerator):
    """
    Synthetic data generator for translation tasks.
    """
    def generate_synthetic_data(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        synthetic_data = []
        for i in range(num_samples):
            input_data = InputData(data_type=EvaluationDataType.TEXT, decoded_data=f"This is input text {i} to be translated.")
            expected_result = ExpectedResult(decoded_result=f"Ceci est le texte d'entr�e {i} � traduire.") # Example French
            eval_data = EvaluationData(
                evaluation_id=str(uuid.uuid4()),
                input_data=input_data,
                expected_result=expected_result,
                status=EvaluationStatus.PENDING
            )
            synthetic_data.append(eval_data)
        return synthetic_data

    def validate_synthetic_data(self, data: List[EvaluationData]) -> bool:
        return all(isinstance(d, EvaluationData) for d in data)

class ClassificationSyntheticDataGenerator(ISyntheticDataGenerator):
    """
    Synthetic data generator for classification tasks.
    """
    def generate_synthetic_data(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        synthetic_data = []
        labels = ["positive", "negative", "neutral"]
        texts = [
            "I love this product!",
            "This is terrible service.",
            "It's an okay experience."
        ]
        for i in range(num_samples):
            text_idx = i % len(texts)
            label_idx = i % len(labels)
            input_data = InputData(data_type=EvaluationDataType.TEXT, decoded_data=texts[text_idx])
            expected_result = ExpectedResult(decoded_result=labels[label_idx], labels=[labels[label_idx]]) # Expected label also as string
            eval_data = EvaluationData(
                evaluation_id=str(uuid.uuid4()),
                input_data=input_data,
                expected_result=expected_result,
                status=EvaluationStatus.PENDING
            )
            synthetic_data.append(eval_data)
        return synthetic_data

    def validate_synthetic_data(self, data: List[EvaluationData]) -> bool:
        return all(isinstance(d, EvaluationData) for d in data)


class RAGSyntheticDataGenerator(ISyntheticDataGenerator):
    """
    Synthetic data generator for RAG tasks (context + query).
    """
    def generate_synthetic_data(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        synthetic_data = []
        contexts = [
            "The capital of France is Paris. Paris is known for its Eiffel Tower.",
            "Mount Everest is the highest mountain in the world, located in the Himalayas."
        ]
        questions = [
            "What is the capital of France?",
            "Where is Mount Everest located?"
        ]
        answers = [
            "Paris",
            "In the Himalayas"
        ]

        for i in range(num_samples):
            idx = i % len(contexts)
            input_data = InputData(data_type=EvaluationDataType.TEXT, decoded_data=questions[idx], context=contexts[idx])
            expected_result = ExpectedResult(decoded_result=answers[idx])
            eval_data = EvaluationData(
                evaluation_id=str(uuid.uuid4()),
                input_data=input_data,
                expected_result=expected_result,
                status=EvaluationStatus.PENDING
            )
            synthetic_data.append(eval_data)
        return synthetic_data

    def validate_synthetic_data(self, data: List[EvaluationData]) -> bool:
        return all(isinstance(d, EvaluationData) for d in data)


class SyntheticDataGeneratorFactory:
    """
    A factory to provide the correct ISyntheticDataGenerator implementation
    based on the AI task name.
    """
    def __init__(self):
        self._generators: Dict[str, ISyntheticDataGenerator] = {
            "summarization": BasicSyntheticDataGenerator(),
            "translation": TranslationSyntheticDataGenerator(),
            "classification": ClassificationSyntheticDataGenerator(),
            "rag": RAGSyntheticDataGenerator(),
            # Add other task generators here
        }

    def get_generator(self, ai_task_name: str) -> ISyntheticDataGenerator:
        """
        Retrieves the appropriate synthetic data generator.
        :param ai_task_name: The name of the AI task.
        :return: An ISyntheticDataGenerator instance.
        :raises ValueError: If no generator is found for the given task.
        """
        generator = self._generators.get(ai_task_name.lower())
        if not generator:
            raise ValueError(f"No synthetic data generator registered for AI task: {ai_task_name}")
        return generator

