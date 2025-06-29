
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from domain.models import EvaluationDataset, EvaluationData, MetricsData

class IEvaluationDomainDataRepository(ABC):
    @abstractmethod
    def save(self, evaluation_data: EvaluationData):
        pass

    @abstractmethod
    def get_by_id(self, evaluation_id: str) -> Optional[EvaluationData]:
        pass

    @abstractmethod
    def get_all(self) -> List[EvaluationData]:
        pass

    @abstractmethod
    def update(self, evaluation_data: EvaluationData):
        pass

class IEvaluationDatasetRepository(ABC):
    @abstractmethod
    def save(self, dataset: EvaluationDataset):
        pass

    @abstractmethod
    def get_by_id(self, dataset_id: str) -> Optional[EvaluationDataset]:
        pass

    @abstractmethod
    def get_all(self) -> List[EvaluationDataset]:
        pass

    @abstractmethod
    def update(self, dataset: EvaluationDataset):
        pass

class IMetricsDetailRepository(ABC):
    @abstractmethod
    def save(self, metrics_data: MetricsData):
        pass

    @abstractmethod
    def get_by_id(self, metrics_id: str) -> Optional[MetricsData]:
        pass

    @abstractmethod
    def get_by_task_name(self, task_name: str) -> Optional[MetricsData]:
        pass

    @abstractmethod
    def get_all(self) -> List[MetricsData]:
        pass