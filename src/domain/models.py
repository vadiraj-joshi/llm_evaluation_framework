import uuid
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
class EvaluationStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class EvaluationMetric(str, Enum):
    ROUGE_L = "ROUGE_L"
    BLEU = "BLEU"
    EM = "Exact Match"

class EvaluationDataType(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"

class InputData(BaseModel):
    data_base64: Optional[str] = None # For future use cases like image/audio
    data_type: EvaluationDataType
    decoded_data: str

class ExpectedResult(BaseModel):
    expected_result_base64: Optional[str] = None
    decoded_result: str

class EvaluationResult(BaseModel):
    metric_name: EvaluationMetric
    metric_value: float
    details: dict = {}

class EvaluationData(BaseModel):
    evaluation_id: str = "" # Default to empty, will be set by repo if not provided
    input_data: InputData
    expected_result: ExpectedResult
    llm_response: Optional[str] = None
    metric_result: Optional[EvaluationResult] = None
    status: EvaluationStatus = EvaluationStatus.PENDING
    error_message: Optional[str] = None

    def record_llm_response(self, response: str):
        self.llm_response = response

    def add_metric_result(self, result: EvaluationResult):
        self.metric_result = result

    def set_status(self, status: EvaluationStatus, error_msg: Optional[str] = None):
        self.status = status
        self.error_message = error_msg

class EvaluationDataset(BaseModel):
    dataset_id: str = "" # Default to empty, will be set by repo if not provided
    sub_domain_id: str
    ai_task_name: str # e.g., Summarization, Translation, Q&A
    task_description: str
    theme: Optional[str] = None
    metadata: dict = {}
    version: int = 1
    evaluation_data: List[EvaluationData] = []
    status: EvaluationStatus = EvaluationStatus.PENDING
    error_message: Optional[str] = None
    overall_score: Optional[float] = None

    def add_evaluation_data(self, data: EvaluationData):
        self.evaluation_data.append(data)

    def set_status(self, status: EvaluationStatus, error_msg: Optional[str] = None):
        self.status = status
        self.error_message = error_msg

    def get_evaluation_status(self) -> EvaluationStatus:
        return self.status

class MetricsData(BaseModel):
    metrics_id: str = "" # Default to empty, will be set by repo if not provided
    ai_task_name: str
    available_metrics: List[EvaluationMetric]
    description: Optional[str] = None

class AITask(BaseModel):
    task_id: str = "" # Default to empty, will be set by repo if not provided
    task_family: str # e.g., Generative, Discriminative
    ai_task_name: str # e.g., Summarization, Sentiment Analysis
    evaluation_metrics: List[EvaluationMetric]
    method_type: Optional[str] = None # e.g., few-shot, zero-shot
