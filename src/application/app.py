import yaml
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# Import domain models
from src.domain.models import EvaluationStatus, EvaluationMetric, EvaluationDataset

# Import domain services
from src.domain.services import DatasetDomainService

# Import ports
from src.ports.llm_service import ILLMService
from src.ports.metrics_calculator import IMetricsCalculator
from src.ports.repositories import IEvaluationDomainDataRepository, IEvaluationDatasetRepository, IMetricsDetailRepository
from src.ports.synthetic_data_generator import ISyntheticDataGenerator

# Import adapters
from src.adapters.llm.openai_adapter import OpenAIAdapter
from src.adapters.metrics.summarization_metrics import SummarizationMetrics
from src.adapters.repositories.in_memory_respository import (
    InMemoryEvaluationDomainDataRepository,
    InMemoryEvaluationDatasetRepository,
    InMemoryMetricsDetailRepository
)
from adapters.parsers.excel_parser import ExcelParser
from adapters.synthetic_data.basic_generator import BasicSyntheticDataGenerator

# Import use cases
from application.use_cases import (
    EvaluateDatasetUseCase,
    ManageDomainDatasetsUseCase,
    LeaderboardGenerationUseCase,
    ImportEvaluationDataUseCase
)

# Import console printer for leaderboard
from src.interfaces.console_leaderboard import ConsoleLeaderboardPrinter

# --- Configuration Loading ---
def load_config(config_path: str = "configs/config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}. Please create it.")
        # Exit to force user to create config
        os._exit(1) # Using os._exit to immediately terminate process
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        os._exit(1) # Using os._exit to immediately terminate process

# Load configuration globally for the application
config = load_config()

# --- Dependency Initialization ---
# Initialize Repositories (In-memory for demonstration)
evaluation_data_repo = InMemoryEvaluationDomainDataRepository()
evaluation_dataset_repo = InMemoryEvaluationDatasetRepository()
metrics_detail_repo = InMemoryMetricsDetailRepository()

# Initialize Adapters
openai_api_key = config.get("openai_api_key")
if not openai_api_key or openai_api_key == "YOUR_OPENAI_API_KEY_HERE":
    raise ValueError(
        "OpenAI API key not found or is default in config.yaml. "
        "Please set 'openai_api_key' in configs/config.yaml with your actual key."
    )
llm_service: ILLMService = OpenAIAdapter(api_key=openai_api_key)
metrics_calculator: IMetricsCalculator = SummarizationMetrics()
excel_parser = ExcelParser()
synthetic_data_generator: ISyntheticDataGenerator = BasicSyntheticDataGenerator()

# Initialize Use Cases
manage_domain_datasets_uc = ManageDomainDatasetsUseCase(
    evaluation_dataset_repo=evaluation_dataset_repo,
    metrics_detail_repo=metrics_detail_repo,
    synthetic_data_generator=synthetic_data_generator
)

evaluate_dataset_uc = EvaluateDatasetUseCase(
    llm_service=llm_service,
    metrics_calculator=metrics_calculator,
    evaluation_data_repo=evaluation_data_repo,
    evaluation_dataset_repo=evaluation_dataset_repo
)

import_evaluation_data_uc = ImportEvaluationDataUseCase(
    excel_parser=excel_parser,
    manage_domain_datasets_use_case=manage_domain_datasets_uc
)

leaderboard_generation_uc = LeaderboardGenerationUseCase(
    evaluation_dataset_repo=evaluation_dataset_repo
)

# --- FastAPI Application Setup ---
app = FastAPI(title="LLM Evaluation Framework")

# Dependency Injection functions for FastAPI
def get_manage_domain_datasets_uc() -> ManageDomainDatasetsUseCase:
    return manage_domain_datasets_uc

def get_evaluate_dataset_uc() -> EvaluateDatasetUseCase:
    return evaluate_dataset_uc

def get_import_evaluation_data_uc() -> ImportEvaluationDataUseCase:
    return import_evaluation_data_uc

def get_leaderboard_generation_uc() -> LeaderboardGenerationUseCase:
    return leaderboard_generation_uc

def get_evaluation_dataset_repo() -> IEvaluationDatasetRepository:
    return evaluation_dataset_repo

# --- API Endpoints ---

class FileUploadRequest(BaseModel):
    task_name: str = "Summarization"
    sub_domain_id: str = "general"
    sheet_name: Optional[str] = None

@app.post("/upload-evaluation-data/", summary="Upload Excel data for evaluation")
async def upload_evaluation_data(
    file: UploadFile = File(..., description="Excel file containing 'input_text' and 'expected_output' columns."),
    task_name: str = "Summarization",
    sub_domain_id: str = "general",
    sheet_name: Optional[str] = None,
    import_uc: ImportEvaluationDataUseCase = Depends(get_import_evaluation_data_uc)
):
    """
    Uploads an Excel file containing 'input_text' and 'expected_output'
    to create an evaluation dataset.
    """
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only Excel files (.xlsx, .xls) are allowed.")

    try:
        file_content = await file.read()
        dataset = import_uc.execute(file_content, task_name, sub_domain_id, sheet_name)
        return JSONResponse(status_code=201, content={
            "message": "Evaluation data imported successfully.",
            "dataset_id": dataset.dataset_id,
            "task_name": dataset.ai_task_name,
            "num_samples": len(dataset.evaluation_data)
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

class EvaluateDatasetRequest(BaseModel):
    dataset_id: str
    llm_model_name: str
    metric_type: EvaluationMetric = EvaluationMetric.ROUGE_L

@app.post("/evaluate-dataset/", summary="Trigger evaluation of a dataset")
async def evaluate_dataset_endpoint(
    request: EvaluateDatasetRequest,
    evaluate_uc: EvaluateDatasetUseCase = Depends(get_evaluate_dataset_uc)
):
    """
    Triggers evaluation of a dataset using a specified LLM model and metric.
    The LLM responses are generated, and metric scores are calculated.
    """
    try:
        evaluated_dataset = evaluate_uc.execute(request.dataset_id, request.llm_model_name, request.metric_type)
        return JSONResponse(status_code=200, content={
            "message": "Dataset evaluation initiated/completed.",
            "dataset_id": evaluated_dataset.dataset_id,
            "status": evaluated_dataset.status,
            "overall_score": evaluated_dataset.overall_score
        })
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {e}")

@app.get("/datasets/", summary="Get all evaluation datasets")
async def get_all_datasets(
    repo: IEvaluationDatasetRepository = Depends(get_evaluation_dataset_repo)
):
    """
    Retrieves a list of all available evaluation datasets.
    """
    datasets = repo.get_all()
    # Convert Enum to string for JSON serialization
    return JSONResponse(status_code=200, content=[d.model_dump(mode='json') for d in datasets])

@app.get("/datasets/{dataset_id}", summary="Get a specific evaluation dataset by ID")
async def get_dataset_by_id(
    dataset_id: str,
    repo: IEvaluationDatasetRepository = Depends(get_evaluation_dataset_repo)
):
    """
    Retrieves a specific evaluation dataset by its ID.
    """
    dataset = repo.get_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    # Convert Enum to string for JSON serialization
    return JSONResponse(status_code=200, content=dataset.model_dump(mode='json'))

@app.get("/leaderboard/", summary="Generate and display the leaderboard")
async def get_leaderboard_endpoint(
    ai_task_name: str = "Summarization",
    metric_type: EvaluationMetric = EvaluationMetric.ROUGE_L,
    leaderboard_uc: LeaderboardGenerationUseCase = Depends(get_leaderboard_generation_uc)
):
    """
    Generates and returns the leaderboard for a specific AI task and metric.
    The detailed leaderboard output is printed to the console where the FastAPI app is running.
    """
    try:
        leaderboard = leaderboard_uc.execute(ai_task_name, metric_type)
        
        # Print beautiful console output
        console_leaderboard_printer = ConsoleLeaderboardPrinter()
        console_leaderboard_printer.print_leaderboard(leaderboard, ai_task_name, metric_type)
        
        # Return a simple JSON response indicating success
        return JSONResponse(status_code=200, content={"message": "Leaderboard printed to console."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating leaderboard: {e}")

