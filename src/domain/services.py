from typing import List
from domain.models import EvaluationDataset, EvaluationMetric, EvaluationData, EvaluationStatus, AITask
from ports.llm_service import ILLMService
from ports.metrics_calculator import IMetricsCalculator # Still for type hinting interface
from ports.synthetic_data_generator import ISyntheticDataGenerator # Still for type hinting interface
from ports.repositories import IEvaluationDomainDataRepository

# NEW: Import the metric and synthetic data factories/registries
from domain.metrics import MetricCalculatorFactory
from domain.synthetic_data import SyntheticDataGeneratorFactory


class DatasetDomainService:
    """
    Service responsible for domain-level calculations related to datasets.
    """
    @staticmethod
    def calculate_overall_dataset_score(dataset: EvaluationDataset, metric: EvaluationMetric) -> float:
        """
        Calculates the overall score for a dataset based on a specific metric.
        This is a simple average for demonstration.
        :param dataset: The EvaluationDataset to score.
        :param metric: The EvaluationMetric to use for calculation.
        :return: The calculated overall score.
        """
        if not dataset.evaluation_data:
            return 0.0

        total_score = 0.0
        count = 0
        for eval_data in dataset.evaluation_data:
            if eval_data.metric_result and eval_data.metric_result.metric_name == metric:
                total_score += eval_data.metric_result.metric_value
                count += 1
        return total_score / count if count > 0 else 0.0

class DomainEvaluationService:
    """
    Service encapsulating the core business logic for evaluating a single piece of evaluation data.
    This includes calling the LLM and orchestrating metric calculation and synthetic data generation.
    It acts as a single entry point for these domain capabilities.
    """
    def __init__(
        self,
        llm_service: ILLMService,
        metric_calculator_factory: MetricCalculatorFactory, # Now depends on the factory
        synthetic_data_generator_factory: SyntheticDataGeneratorFactory # Now depends on the factory
    ):
        self.llm_service = llm_service
        self.metric_calculator_factory = metric_calculator_factory
        self.synthetic_data_generator_factory = synthetic_data_generator_factory

    def evaluate_single_data_point(
        self,
        eval_data: EvaluationData,
        llm_model_name: str,
        ai_task_name: str, # Pass task name to select correct metric
        metric_type: EvaluationMetric
    ) -> EvaluationData:
        """
        Processes a single EvaluationData item: gets LLM response, calculates metric, and updates status.
        This method represents a core business process of evaluating one test case.

        Reliability & Resiliency:
        - LLM calls include retries.
        - Robust error handling for each data point to prevent full dataset failure.

        :param eval_data: The EvaluationData entity to process.
        :param llm_model_name: The name of the LLM model to use.
        :param ai_task_name: The name of the AI task to determine prompt structure and metric.
        :param metric_type: The metric to calculate.
        :return: The updated EvaluationData entity.
        :raises ValueError: If the metric type is not supported.
        """
        # Get the correct metric calculator based on the AI Task Name and Metric Type
        metrics_calculator: IMetricsCalculator = self.metric_calculator_factory.get_calculator(ai_task_name, metric_type)

        if not metrics_calculator.supports_metric_type(metric_type):
            raise ValueError(f"Metric type {metric_type} not supported by the {ai_task_name} metrics calculator.")

        eval_data.set_status(EvaluationStatus.IN_PROGRESS)

        try:
            # 1. Prepare LLM Prompt based on Task Type
            prompt = self._prepare_llm_prompt(eval_data.input_data.decoded_data, ai_task_name, eval_data.input_data.context)
            
            # 2. Get LLM Response
            llm_response = self.llm_service.get_llm_response(prompt, llm_model_name)
            eval_data.record_llm_response(llm_response)

            # 3. Calculate Metric Score - pass additional context for metrics if needed
            metric_result = metrics_calculator.calculate_score(
                expected_output=eval_data.expected_result.decoded_result,
                llm_output=llm_response,
                expected_labels=eval_data.expected_result.labels # For classification
            )
            eval_data.add_metric_result(metric_result)
            eval_data.set_status(EvaluationStatus.COMPLETED)
        except Exception as e:
            eval_data.set_status(EvaluationStatus.FAILED, str(e))
            print(f"Error evaluating data {eval_data.evaluation_id}: {e}")
        
        return eval_data

    def _prepare_llm_prompt(self, input_text: str, ai_task_name: str, context: str = None) -> str:
        """
        Internal helper to construct prompts based on the AI task.
        This is a simple example; complex prompt engineering would involve more logic.
        """
        if ai_task_name.lower() == "summarization":
            return f"Summarize the following text: {input_text}"
        elif ai_task_name.lower() == "translation":
            return f"Translate the following text to English: {input_text}" # Assuming target language is English
        elif ai_task_name.lower() == "classification":
            # For classification, assuming the LLM should output a class label
            # A more robust solution might provide labels as part of the prompt
            return f"Classify the following text into one category: {input_text}"
        elif ai_task_name.lower() == "rag":
            if not context:
                raise ValueError("Context is required for RAG evaluation.")
            return f"Given the context: '{context}', answer the following question: {input_text}"
        else:
            return input_text # Default to plain text for unknown tasks


    def generate_and_validate_synthetic_data_domain(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        """
        Generates and validates synthetic data within the domain service,
        using the appropriate generator from the factory.

        :param num_samples: The number of synthetic samples to generate.
        :param task_name: The name of the AI task for which to generate data.
        :return: A list of validated synthetic EvaluationData objects.
        :raises ValueError: If synthetic data generation or validation fails.
        """
        synthetic_data_generator: ISyntheticDataGenerator = self.synthetic_data_generator_factory.get_generator(task_name)
        
        synthetic_data = synthetic_data_generator.generate_synthetic_data(num_samples, task_name)
        if not synthetic_data_generator.validate_synthetic_data(synthetic_data):
            raise ValueError("Generated synthetic data failed validation within domain service.")
        return synthetic_data

