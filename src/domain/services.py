from domain.models import EvaluationDataset, EvaluationMetric, EvaluationData, EvaluationStatus
from ports.llm_service import ILLMService
from ports.metrics_calculator import IMetricsCalculator # Still imports port for type hinting
from ports.synthetic_data_generator import ISyntheticDataGenerator # Still imports port for type hinting
from ports.repositories import IEvaluationDomainDataRepository

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
    This logic includes calling the LLM and calculating metrics, representing a key domain process.
    It now acts as a single entry point for using both LLM services and metric calculation within the domain.
    """
    def __init__(
        self,
        llm_service: ILLMService,
        metrics_calculator: IMetricsCalculator, # Dependency on the port
        synthetic_data_generator: ISyntheticDataGenerator # Dependency on the port
    ):
        self.llm_service = llm_service
        self.metrics_calculator = metrics_calculator
        self.synthetic_data_generator = synthetic_data_generator

    def evaluate_single_data_point(
        self,
        eval_data: EvaluationData,
        llm_model_name: str,
        metric_type: EvaluationMetric
    ) -> EvaluationData:
        """
        Processes a single EvaluationData item: gets LLM response, calculates metric, and updates status.
        This method represents a core business process of evaluating one test case.

        :param eval_data: The EvaluationData entity to process.
        :param llm_model_name: The name of the LLM model to use.
        :param metric_type: The metric to calculate.
        :return: The updated EvaluationData entity.
        :raises ValueError: If the metric type is not supported.
        """
        if not self.metrics_calculator.supports_metric_type(metric_type):
            raise ValueError(f"Metric type {metric_type} not supported by the current metrics calculator.")

        eval_data.set_status(EvaluationStatus.IN_PROGRESS)

        try:
            # 1. Get LLM Response
            prompt = eval_data.input_data.decoded_data
            llm_response = self.llm_service.get_llm_response(prompt, llm_model_name)
            eval_data.record_llm_response(llm_response)

            # 2. Calculate Metric Score
            expected_output = eval_data.expected_result.decoded_result
            metric_result = self.metrics_calculator.calculate_score(expected_output, llm_response)
            eval_data.add_metric_result(metric_result)
            eval_data.set_status(EvaluationStatus.COMPLETED)
        except Exception as e:
            eval_data.set_status(EvaluationStatus.FAILED, str(e))
            print(f"Error evaluating data {eval_data.evaluation_id}: {e}")
        
        return eval_data

    def generate_and_validate_synthetic_data_domain(self, num_samples: int, task_name: str) -> List[EvaluationData]:
        """
        Generates and validates synthetic data within the domain service.
        This provides a single entry point for synthetic data generation logic.

        :param num_samples: The number of synthetic samples to generate.
        :param task_name: The name of the AI task for which to generate data.
        :return: A list of validated synthetic EvaluationData objects.
        :raises ValueError: If synthetic data generation or validation fails.
        """
        synthetic_data = self.synthetic_data_generator.generate_synthetic_data(num_samples, task_name)
        if not self.synthetic_data_generator.validate_synthetic_data(synthetic_data):
            raise ValueError("Generated synthetic data failed validation within domain service.")
        return synthetic_data
