import yaml
import os
import sys
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import domain models
from domain.models import EvaluationStatus, EvaluationMetric, EvaluationDataset

# Import domain services
from domain.services import DatasetDomainService, DomainEvaluationService

# Import domain-internal metrics and synthetic data factories
from domain.metrics import MetricCalculatorFactory # NEW Import
from domain.synthetic_data import SyntheticDataGeneratorFactory # NEW Import

# Import ports
from ports.authentication_service import IAuthenticationService
from ports.llm_service import ILLMService
from ports.metrics_calculator import IMetricsCalculator # Still for type hinting interface
from ports.repositories import IEvaluationDomainDataRepository, IEvaluationDatasetRepository, IMetricsDetailRepository
from ports.synthetic_data_generator import ISyntheticDataGenerator # Still for type hinting interface

# Import adapters
from adapters.authentication.basic_auth_adapter import BasicAuthAdapter
from adapters.llm.openai_adapter import OpenAIAdapter
from adapters.repositories.in_memory_repository import (
    InMemoryEvaluationDomainDataRepository,
    InMemoryEvaluationDatasetRepository,
    InMemoryMetricsDetailRepository
)
from adapters.parsers.excel_parser import ExcelParser

# Import use cases
from application.use_cases import (
    EvaluateDatasetUseCase,
    ManageDomainDatasetsUseCase,
    LeaderboardGenerationUseCase,
    ImportEvaluationDataUseCase
)

# Import console printer for leaderboard
from interfaces.console_leaderboard import ConsoleLeaderboardPrinter

# --- Configuration Loading ---
def load_config(config_path: str = "configs/config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}. Please create it.")
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

# Initialize Adapters (External Dependencies)
openai_api_key = config.get("openai_api_key")
if not openai_api_key or openai_api_key == "YOUR_OPENAI_API_KEY_HERE":
    raise ValueError(
        "OpenAI API key not found or is default in config.yaml. "
        "Please set 'openai_api_key' in configs/config.yaml with your actual key."
    )
llm_service: ILLMService = OpenAIAdapter(api_key=openai_api_key)

# NEW: Initialize domain-internal components (implementing ports)
# These are instantiated here in the application layer and then passed into domain services.
metric_calculator_factory = MetricCalculatorFactory() # NEW
synthetic_data_generator_factory = SyntheticDataGeneratorFactory() # NEW

excel_parser = ExcelParser()

# Initialize Authentication Adapter
auth_config = config.get("auth", {})
USERS_DB = {auth_config.get("username"): auth_config.get("password")}
authentication_service: IAuthenticationService = BasicAuthAdapter(users=USERS_DB)
security = HTTPBasic() # FastAPI's built-in Basic Auth utility

# Initialize Domain Services
domain_evaluation_service = DomainEvaluationService(
    llm_service=llm_service, # External adapter
    metric_calculator_factory=metric_calculator_factory, # NEW: Factory for metrics
    synthetic_data_generator_factory=synthetic_data_generator_factory # NEW: Factory for synthetic data
)

# Initialize Use Cases
manage_domain_datasets_uc = ManageDomainDatasetsUseCase(
    evaluation_dataset_repo=evaluation_dataset_repo,
    metrics_detail_repo=metrics_detail_repo,
    domain_evaluation_service=domain_evaluation_service # Now depends on the unified domain service
)

evaluate_dataset_uc = EvaluateDatasetUseCase(
    domain_evaluation_service=domain_evaluation_service,
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

# Dependency for Authentication
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Authenticates the user based on Basic HTTP credentials.
    Returns the username if authenticated, otherwise raises an HTTPException.
    """
    user_id = authentication_service.authenticate_user(credentials.username, credentials.password)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user_id

# --- API Endpoints ---

class FileUploadRequest(BaseModel):
    task_name: str = "Summarization" # Default task
    sub_domain_id: str = "general"
    sheet_name: Optional[str] = None

@app.post("/upload-evaluation-data/", summary="Upload Excel data for evaluation")
async def upload_evaluation_data(
    file: UploadFile = File(..., description="Excel file containing 'input_text', 'expected_output'. Optional: 'context', 'labels'."),
    task_name: str = "Summarization",
    sub_domain_id: str = "general",
    sheet_name: Optional[str] = None,
    import_uc: ImportEvaluationDataUseCase = Depends(get_import_evaluation_data_uc),
    current_user: str = Depends(get_current_user) # PROTECTED endpoint
):
    """
    Uploads an Excel file containing evaluation data for various LLM tasks.
    Required columns: 'input_text', 'expected_output'.
    Optional for specific tasks: 'context' (for RAG), 'labels' (for Classification - comma separated).
    This endpoint is protected by basic authentication.
    """
    print(f"Authenticated user: {current_user} is uploading data for task: {task_name}.")
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
    ai_task_name: str # e.g., Summarization, Translation, Classification, RAG
    metric_type: EvaluationMetric = EvaluationMetric.ROUGE_L # Default, but should match task

@app.post("/evaluate-dataset/", summary="Trigger evaluation of a dataset")
async def evaluate_dataset_endpoint(
    request: EvaluateDatasetRequest,
    evaluate_uc: EvaluateDatasetUseCase = Depends(get_evaluate_dataset_uc),
    current_user: str = Depends(get_current_user) # PROTECTED endpoint
):
    """
    Triggers evaluation of a dataset using a specified LLM model, AI task, and metric.
    The LLM responses are generated, and metric scores are calculated.
    This endpoint is protected by basic authentication.
    """
    print(f"Authenticated user: {current_user} is triggering evaluation for dataset {request.dataset_id} "
          f"with model {request.llm_model_name} for task {request.ai_task_name} using metric {request.metric_type.value}.")
    try:
        evaluated_dataset = evaluate_uc.execute(request.dataset_id, request.llm_model_name, request.ai_task_name, request.metric_type)
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
    This endpoint is intentionally left unprotected for easy viewing of available datasets.
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
    This endpoint is intentionally left unprotected for easy viewing of available datasets.
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
    This endpoint is intentionally left unprotected for public viewing of results.
    """
    print(f"Generating leaderboard for task: {ai_task_name} with metric: {metric_type.value}.")
    try:
        leaderboard = leaderboard_uc.execute(ai_task_name, metric_type)
        
        # Print beautiful console output
        console_leaderboard_printer = ConsoleLeaderboardPrinter()
        console_leaderboard_printer.print_leaderboard(leaderboard, ai_task_name, metric_type)
        
        # Return a simple JSON response indicating success
        return JSONResponse(status_code=200, content={"message": "Leaderboard printed to console."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating leaderboard: {e}")

# Basic token endpoint for demonstration (optional, just to show auth flow)
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@app.post("/token", response_model=TokenResponse, summary="Get an authentication token (Basic Auth)")
async def login_for_access_token(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Authenticate with username/password to get a dummy access token.
    Use this token for subsequent protected requests.
    """
    user_id = authentication_service.authenticate_user(credentials.username, credentials.password)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    # In a real app, generate a JWT here
    token = f"dummy_token_for_{user_id}"
    # The BasicAuthAdapter already "stores" the dummy token associated with the user.
    return {"access_token": token}
