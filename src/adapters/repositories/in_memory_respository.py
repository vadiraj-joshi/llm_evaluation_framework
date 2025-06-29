
from openai import OpenAI
from openai import APIConnectionError, APIStatusError # Import specific exceptions
import time
from ports.repositories import IEvaluationDomainDataRepository, IEvaluationDatasetRepository, IMetricsDetailRepository
import uuid
from typing import List, Optional, Dict, Any
from domain.models import EvaluationData, EvaluationDataset, MetricsData

class InMemoryEvaluationDomainDataRepository(IEvaluationDomainDataRepository):
    def __init__(self):
        self.data: List[EvaluationData] = []

    def save(self, evaluation_data: EvaluationData):
        if not evaluation_data.evaluation_id:
            evaluation_data.evaluation_id = str(uuid.uuid4())
        self.data.append(evaluation_data)

    def get_by_id(self, evaluation_id: str) -> Optional[EvaluationData]:
        return next((d for d in self.data if d.evaluation_id == evaluation_id), None)

    def get_all(self) -> List[EvaluationData]:
        return list(self.data)

    def update(self, evaluation_data: EvaluationData):
        for i, d in enumerate(self.data):
            if d.evaluation_id == evaluation_data.evaluation_id:
                self.data[i] = evaluation_data
                return
        self.save(evaluation_data) # If not found, save it (upsert-like behavior)

class InMemoryEvaluationDatasetRepository(IEvaluationDatasetRepository):
    def __init__(self):
        self.datasets: List[EvaluationDataset] = []

    def save(self, dataset: EvaluationDataset):
        if not dataset.dataset_id:
            dataset.dataset_id = str(uuid.uuid4())
        self.datasets.append(dataset)

    def get_by_id(self, dataset_id: str) -> Optional[EvaluationDataset]:
        return next((d for d in self.datasets if d.dataset_id == dataset_id), None)

    def get_all(self) -> List[EvaluationDataset]:
        return list(self.datasets)

    def update(self, dataset: EvaluationDataset):
        for i, d in enumerate(self.datasets):
            if d.dataset_id == dataset.dataset_id:
                self.datasets[i] = dataset
                return
        self.save(dataset)

class InMemoryMetricsDetailRepository(IMetricsDetailRepository):
    def __init__(self):
        self.metrics_data: List[MetricsData] = []

    def save(self, metrics_data: MetricsData):
        if not metrics_data.metrics_id:
            metrics_data.metrics_id = str(uuid.uuid4())
        self.metrics_data.append(metrics_data)

    def get_by_id(self, metrics_id: str) -> Optional[MetricsData]:
        return next((m for m in self.metrics_data if m.metrics_id == metrics_id), None)

    def get_by_task_name(self, task_name: str) -> Optional[MetricsData]:
        return next((m for m in self.metrics_data if m.ai_task_name == task_name), None)

    def get_all(self) -> List[MetricsData]:
        return list(self.metrics_data)