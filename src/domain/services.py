from src.domain.models import EvaluationDataset, EvaluationMetric

class DatasetDomainService:
    @staticmethod
    def calculate_overall_dataset_score(dataset: EvaluationDataset, metric: EvaluationMetric) -> float:
        """
        Calculates the overall score for a dataset based on a specific metric.
        This is a simple average for demonstration.
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