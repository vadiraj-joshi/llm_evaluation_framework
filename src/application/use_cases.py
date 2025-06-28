import uuid
from typing import List, Optional, Any

from domain.models import (
    EvaluationDataset, EvaluationData, EvaluationStatus, EvaluationMetric,
    InputData, ExpectedResult, EvaluationDataType, MetricsData
)
from domain.services import DatasetDomainService, DomainEvaluationService
from ports.repositories import (
    IEvaluationDomainDataRepository, IEvaluationDatasetRepository, IMetricsDetailRepository
)
from adapters.parsers.excel_parser import ExcelParser

class EvaluateDatasetUseCase:
    """
    Use case for evaluating a given dataset against an LLM and calculating metrics.
    Orchestrates the evaluation process, delegating single data point evaluation
    to the DomainEvaluationService.
    """
    def __init__(self,
                 domain_evaluation_service: DomainEvaluationService,
                 evaluation_data_repo: IEvaluationDomainDataRepository,
                 evaluation_dataset_repo: IEvaluationDatasetRepository):
        self.domain_evaluation_service = domain_evaluation_service
        self.evaluation_data_repo = evaluation_data_repo
        self.evaluation_dataset_repo = evaluation_dataset_repo

    def execute(self, dataset_id: str, llm_model_name: str, ai_task_name: str, metric_type: EvaluationMetric) -> EvaluationDataset:
        """
        Executes the evaluation process for a specified dataset.
        For each evaluation data point, it calls the DomainEvaluationService
        to get LLM responses and calculate metric scores.

        Scalability, Reliability & Resiliency Note:
        For production systems with large datasets (e.g., >100 rows), this synchronous
        loop within an API request is a bottleneck. Consider:
        1.  **Asynchronous Task Queues (Scalability/Reliability/Availability):** Offload each `evaluate_single_data_point` call to a message queue (e.g., Kafka, RabbitMQ) and process them with dedicated worker services (e.g., Celery workers, Kubernetes Jobs). The API would then return an immediate "Evaluation initiated" status. This provides resilience to API timeouts and allows horizontal scaling of workers.
        2.  **Distributed Processing:** For very large datasets, use frameworks like Apache Spark or Dask to process evaluation data in parallel across a cluster.
        3.  **Idempotency & Checkpointing (Reliability):** Ensure the evaluation process is idempotent. If a worker fails, it can pick up from where it left off without duplicating effort.
        4.  **Circuit Breakers/Bulkheads (Resiliency):** For LLM calls, implement more sophisticated patterns to isolate failures and prevent cascading issues.

        :param dataset_id: The ID of the dataset to evaluate.
        :param llm_model_name: The name of the LLM model to use for generating responses.
        :param ai_task_name: The specific AI task (e.g., "Summarization", "Translation").
        :param metric_type: The metric to use for calculating scores (e.g., ROUGE_L, BLEU, ACCURACY).
        :return: The updated EvaluationDataset after evaluation.
        :raises ValueError: If the dataset is not found or metric type is unsupported by the domain service.
        """
        dataset = self.evaluation_dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found.")
        
        # Override dataset's AI task name if the user provides it, to ensure consistency in evaluation
        dataset.ai_task_name = ai_task_name 

        dataset.set_status(EvaluationStatus.IN_PROGRESS)
        self.evaluation_dataset_repo.update(dataset) # Persist status change

        for eval_data in dataset.evaluation_data:
            # Delegate the core evaluation logic for a single data point to the domain service.
            updated_eval_data = self.domain_evaluation_service.evaluate_single_data_point(
                eval_data=eval_data,
                llm_model_name=llm_model_name,
                ai_task_name=ai_task_name, # Pass task name for prompt building and metric selection
                metric_type=metric_type
            )
            # Persist the updated evaluation data point state
            self.evaluation_data_repo.update(updated_eval_data)

        # Calculate overall score for the dataset (using domain service)
        overall_score = DatasetDomainService.calculate_overall_dataset_score(dataset, metric_type)
        dataset.overall_score = overall_score
        dataset.set_status(EvaluationStatus.COMPLETED)
        # Store the LLM model name and metric used in dataset metadata for leaderboard
        dataset.metadata["llm_model_name"] = llm_model_name
        dataset.metadata["metric_for_score"] = metric_type.value
        self.evaluation_dataset_repo.update(dataset) # Persist final dataset status and score
        return dataset

class ManageDomainDatasetsUseCase:
    """
    Use case for managing evaluation datasets, including creation and synthetic data generation.
    This now uses the DomainEvaluationService as a single entry point for synthetic data generation.
    """
    def __init__(self,
                 evaluation_dataset_repo: IEvaluationDatasetRepository,
                 metrics_detail_repo: IMetricsDetailRepository,
                 domain_evaluation_service: DomainEvaluationService): # Depends on the unified domain service
        self.evaluation_dataset_repo = evaluation_dataset_repo
        self.metrics_detail_repo = metrics_detail_repo
        self.domain_evaluation_service = domain_evaluation_service

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
        Generates synthetic data (via DomainEvaluationService) and adds it to an existing dataset.
        :param dataset_id: The ID of the dataset to add data to.
        :param num_samples: The number of synthetic samples to generate.
        :raises ValueError: If dataset not found or synthetic data fails validation.
        """
        dataset = self.evaluation_dataset_repo.get_by_id(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found.")

        # Call the domain service for synthetic data generation and validation
        synthetic_data = self.domain_evaluation_service.generate_and_validate_synthetic_data_domain(
            num_samples=num_samples,
            task_name=dataset.ai_task_name
        )

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
            # Only include completed datasets for the specified task and if they have score for chosen metric
            if dataset.ai_task_name == ai_task_name and \
               dataset.status == EvaluationStatus.COMPLETED and \
               dataset.metadata.get("metric_for_score") == metric_type.value: # Check if evaluated with this metric
                
                # We assume overall_score is already calculated and stored for this metric type.
                # If not, DatasetDomainService.calculate_overall_dataset_score would be called here.
                # For this iteration, we rely on `evaluate_dataset_endpoint` to store the score.

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
        Handles parsing specific columns for different task types.
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
            input_data_kwargs = {
                "data_type": EvaluationDataType.TEXT,
                "decoded_data": item['input_text'],
                "context": item.get('context') # Pass context if available
            }
            input_data = InputData(**input_data_kwargs)

            expected_result_kwargs = {
                "decoded_result": item['expected_output'],
                "labels": item.get('labels') # Pass labels if available
            }
            expected_result = ExpectedResult(**expected_result_kwargs)
            
            eval_data = EvaluationData(
                evaluation_id=str(uuid.uuid4()),
                input_data=input_data,
                expected_result=expected_result,
                status=EvaluationStatus.PENDING
            )
            evaluation_data_list.append(eval_data)

        # Create a new dataset
        dataset = self.manage_domain_datasets_use_case.create_evaluation_dataset(
            sub_domain_id=sub_domain_id,
            ai_task_name=task_name,
            task_description=f"Evaluation dataset for {task_name} imported from Excel",
            evaluation_data=evaluation_data_list
        )
        return dataset
