import uuid
from typing import List, Optional

from domain.models import (
    EvaluationDataset, EvaluationData, EvaluationStatus, EvaluationMetric,
    InputData, ExpectedResult, EvaluationDataType, MetricsData
)
from domain.services import DatasetDomainService
from ports.llm_service import ILLMService
from ports.metrics_calculator import IMetricsCalculator
from ports.repositories import (
    IEvaluationDomainDataRepository, IEvaluationDatasetRepository, IMetricsDetailRepository
)
from ports.synthetic_data_generator import ISyntheticDataGenerator
from adapters.parsers.excel_parser import ExcelParser # Used by ImportEvaluationDataUseCase

class EvaluateDatasetUseCase:
    """
    Use case for evaluating a given dataset against an LLM and calculating metrics.
    """
    def __init__(self,
                 llm_service: ILLMService,
                 metrics_calculator: IMetricsCalculator,
                 evaluation_data_repo: IEvaluationDomainDataRepository,
                 evaluation_dataset_repo: IEvaluationDatasetRepository):
        self.llm_service = llm_service
        self.metrics_calculator = metrics_calculator
        self.evaluation_data_repo = evaluation_data_repo
        self.evaluation_dataset_repo = evaluation_dataset_repo

    def execute(self, dataset_id: str, llm_model_name: str, metric_type: EvaluationMetric) -> EvaluationDataset:
        """
        Executes the evaluation process for a specified dataset.
        :param dataset_id: The ID of the dataset to evaluate.
        :param llm_model_name: The name of the LLM model to use for generating responses.
        :param metric_type: The metric to use for calculating scores (e.g., ROUGE_L).
        :return: The updated EvaluationDataset after evaluation.
        :raises ValueError: If the dataset is not found or metric type is unsupported.
        """
        dataset = self.evaluation_dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found.")

        if not self.metrics_calculator.supports_metric_type(metric_type):
            raise ValueError(f"Metric type {metric_type} not supported by the current metrics calculator.")

        # Update dataset status to indicate evaluation is in progress
        dataset.set_status(EvaluationStatus.IN_PROGRESS)
        self.evaluation_dataset_repo.update(dataset)

        for eval_data in dataset.evaluation_data:
            eval_data.set_status(EvaluationStatus.IN_PROGRESS)
            self.evaluation_data_repo.update(eval_data)
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
            finally:
                self.evaluation_data_repo.update(eval_data)

        # Calculate overall score for the dataset
        overall_score = DatasetDomainService.calculate_overall_dataset_score(dataset, metric_type)
        dataset.overall_score = overall_score
        dataset.set_status(EvaluationStatus.COMPLETED)
        # Store the LLM model name in dataset metadata for leaderboard
        if "llm_model_name" not in dataset.metadata:
            dataset.metadata["llm_model_name"] = llm_model_name
        self.evaluation_dataset_repo.update(dataset)
        return dataset

class ManageDomainDatasetsUseCase:
    """
    Use case for managing evaluation datasets, including creation and synthetic data generation.
    """
    def __init__(self,
                 evaluation_dataset_repo: IEvaluationDatasetRepository,
                 metrics_detail_repo: IMetricsDetailRepository,
                 synthetic_data_generator: ISyntheticDataGenerator):
        self.evaluation_dataset_repo = evaluation_dataset_repo
        self.metrics_detail_repo = metrics_detail_repo
        self.synthetic_data_generator = synthetic_data_generator

    def create_evaluation_dataset(self,
                                  sub_domain_id: str,
                                  ai_task_name: str,
                                  task_description: str,
                                  evaluation_data: Optional[List[EvaluationData]] = None,
                                  theme: Optional[str] = None,
                                  metadata: Optional[dict] = None) -> EvaluationDataset:
        """
        Creates a new evaluation dataset.
        :param sub_domain_id: The ID of the subdomain (e.g., 'healthcare', 'finance').
        :param ai_task_name: The name of the AI task (e.g., 'Summarization', 'Sentiment Analysis').
        :param task_description: A description of the task.
        :param evaluation_data: Optional list of EvaluationData to include initially.
        :param theme: Optional theme for the dataset.
        :param metadata: Optional additional metadata.
        :return: The newly created EvaluationDataset.
        """
        if evaluation_data is None:
            evaluation_data = []
        if metadata is None:
            metadata = {}

        dataset = EvaluationDataset(
            dataset_id=str(uuid.uuid4()),
            sub_domain_id=sub_domain_id,
            ai_task_name=ai_task_name,
            task_description=task_description,
            theme=theme,
            metadata=metadata,
            evaluation_data=evaluation_data,
            status=EvaluationStatus.PENDING
        )
        self.evaluation_dataset_repo.save(dataset)
        return dataset

    def validate_evaluation_data_to_dataset(self, dataset_id: str) -> bool:
        """
        Validates if all data entries within a dataset are of the correct type.
        :param dataset_id: The ID of the dataset to validate.
        :return: True if all data is valid, False otherwise.
        :raises ValueError: If the dataset is not found.
        """
        dataset = self.evaluation_dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found.")
        return all(isinstance(ed, EvaluationData) for ed in dataset.evaluation_data)

    def generate_and_add_synthetic_data(self, dataset_id: str, num_samples: int):
        """
        Generates synthetic data and adds it to an existing dataset.
        :param dataset_id: The ID of the dataset to add data to.
        :param num_samples: The number of synthetic samples to generate.
        :raises ValueError: If dataset not found or synthetic data fails validation.
        """
        dataset = self.evaluation_dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found.")

        synthetic_data = self.synthetic_data_generator.generate_synthetic_data(num_samples, dataset.ai_task_name)
        if not self.synthetic_data_generator.validate_synthetic_data(synthetic_data):
            raise ValueError("Generated synthetic data failed validation.")

        for data in synthetic_data:
            dataset.add_evaluation_data(data)
        self.evaluation_dataset_repo.update(dataset)

    def register_ai_task_metrics(self, task_name: str, metrics: List[EvaluationMetric], description: Optional[str] = None):
        """
        Registers available metrics for a specific AI task.
        :param task_name: The name of the AI task.
        :param metrics: A list of EvaluationMetric types applicable to the task.
        :param description: Optional description for the metrics.
        """
        metrics_data = MetricsData(
            metrics_id=str(uuid.uuid4()),
            ai_task_name=task_name,
            available_metrics=metrics,
            description=description
        )
        self.metrics_detail_repo.save(metrics_data)

class LeaderboardGenerationUseCase:
    """
    Use case for generating a leaderboard of LLM model performance.
    """
    def __init__(self, evaluation_dataset_repo: IEvaluationDatasetRepository):
        self.evaluation_dataset_repo = evaluation_dataset_repo

    def execute(self, ai_task_name: str, metric_type: EvaluationMetric) -> List[dict]:
        """
        Generates a leaderboard based on a specific AI task and metric.
        :param ai_task_name: The AI task for which to generate the leaderboard.
        :param metric_type: The metric by which to rank models.
        :return: A sorted list of dictionaries representing leaderboard entries.
        """
        all_datasets = self.evaluation_dataset_repo.get_all()
        leaderboard_entries = []

        for dataset in all_datasets:
            # Only include completed datasets for the specified task
            if dataset.ai_task_name == ai_task_name and dataset.status == EvaluationStatus.COMPLETED:
                # Ensure overall_score is calculated for the specified metric
                # If not present or if the metric used for stored score differs, recalculate
                if dataset.overall_score is None or dataset.metadata.get("metric_for_score") != metric_type.value:
                    dataset.overall_score = DatasetDomainService.calculate_overall_dataset_score(dataset, metric_type)
                    dataset.metadata["metric_for_score"] = metric_type.value
                    self.evaluation_dataset_repo.update(dataset) # Persist the score

                leaderboard_entries.append({
                    "dataset_id": dataset.dataset_id,
                    "sub_domain_id": dataset.sub_domain_id,
                    "task_description": dataset.task_description,
                    "llm_model_name": dataset.metadata.get("llm_model_name", "N/A"), # Get model name from metadata
                    "score": dataset.overall_score
                })
        # Sort by score in descending order for a leaderboard
        leaderboard_entries.sort(key=lambda x: x["score"], reverse=True)
        return leaderboard_entries


class ImportEvaluationDataUseCase:
    """
    Use case for importing evaluation data from external sources like Excel files.
    """
    def __init__(self,
                 excel_parser: ExcelParser,
                 manage_domain_datasets_use_case: ManageDomainDatasetsUseCase):
        self.excel_parser = excel_parser
        self.manage_domain_datasets_use_case = manage_domain_datasets_use_case

    def execute(self,
                file_content: bytes,
                task_name: str,
                sub_domain_id: str,
                sheet_name: Optional[str] = None) -> EvaluationDataset:
        """
        Imports evaluation data from an Excel file, creates EvaluationData objects,
        and adds them to a new or existing EvaluationDataset.
        :param file_content: The binary content of the Excel file.
        :param task_name: The AI task name for the imported data.
        :param sub_domain_id: The subdomain ID for the imported data.
        :param sheet_name: Optional. The specific sheet name to read from.
        :return: The EvaluationDataset containing the imported data.
        :raises Exception: If file parsing or dataset creation fails.
        """
        parsed_data = self.excel_parser.parse_excel_to_evaluation_data(file_content, sheet_name)

        evaluation_data_list = []
        for item in parsed_data:
            input_data = InputData(data_type=EvaluationDataType.TEXT, decoded_data=item['input_text'])
            expected_result = ExpectedResult(decoded_result=item['expected_output'])
            eval_data = EvaluationData(
                evaluation_id=str(uuid.uuid4()),
                input_data=input_data,
                expected_result=expected_result,
                status=EvaluationStatus.PENDING
            )
            evaluation_data_list.append(eval_data)

        # Create a new dataset (for simplicity, always creating a new one here)
        dataset = self.manage_domain_datasets_use_case.create_evaluation_dataset(
            sub_domain_id=sub_domain_id,
            ai_task_name=task_name,
            task_description=f"Evaluation dataset for {task_name} imported from Excel",
            evaluation_data=evaluation_data_list
        )
        return dataset