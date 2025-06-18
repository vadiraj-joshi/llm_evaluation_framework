from typing import List, Dict, Any
from domain.models import EvaluationMetric # Import necessary for type hints

class ConsoleLeaderboardPrinter:
    """
    A utility class to print a formatted LLM evaluation leaderboard to the console.
    """
    def print_leaderboard(self, leaderboard_data: List[Dict[str, Any]], task_name: str, metric_type: EvaluationMetric):
        """
        Prints a beautifully formatted leaderboard to the console,
        including a comparison of the top two model performances if available.

        :param leaderboard_data: A list of dictionaries, where each dictionary represents
                                 a leaderboard entry with keys like 'dataset_id', 'sub_domain_id',
                                 'llm_model_name', and 'score'.
        :param task_name: The name of the AI task being evaluated (e.g., "Summarization").
        :param metric_type: The metric used for evaluation (e.g., EvaluationMetric.ROUGE_L).
        """
        print("\n" + "="*80)
        print(f"{'LLM Evaluation Leaderboard'.center(80)}")
        print(f"{f'Task: {task_name} | Metric: {metric_type.value}'.center(80)}")
        print("="*80)

        if not leaderboard_data:
            print(f"{'No completed evaluations for this task and metric yet.'.center(80)}")
            print("="*80)
            return

        # Prepare data for display
        rows = []
        # Header row
        rows.append(["Rank", "Dataset ID", "Sub-Domain", "LLM Model", "Score"])
        # Separator row
        rows.append(["-"*4, "-"*10, "-"*10, "-"*10, "-"*7])

        for i, entry in enumerate(leaderboard_data):
            rows.append([
                str(i + 1),
                entry['dataset_id'][:8] + "...", # Truncate ID for cleaner display
                entry['sub_domain_id'],
                entry['llm_model_name'],
                f"{entry['score']:.4f}" # Format score to 4 decimal places
            ])

        # Determine optimal column widths based on content
        column_widths = [max(len(str(item)) for item in col) for col in zip(*rows)]
        formatted_rows = []

        for row in rows:
            # Join columns with a separator, left-justifying each item to its determined width
            formatted_row = " | ".join(str(item).ljust(width) for item, width in zip(row, column_widths))
            formatted_rows.append(formatted_row)

        print("\n".join(formatted_rows))
        print("="*80)

        # Optional: Comparison of top 2 models
        if len(leaderboard_data) >= 2:
            model1 = leaderboard_data[0]
            model2 = leaderboard_data[1]
            print("\n" + "~"*80)
            print(f"{'Model Performance Comparison (Top 2)'.center(80)}")
            print("~"*80)
            print(f"Model 1: {model1['llm_model_name']} (Dataset: {model1['dataset_id'][:8]}...) - Score: {model1['score']:.4f}")
            print(f"Model 2: {model2['llm_model_name']} (Dataset: {model2['dataset_id'][:8]}...) - Score: {model2['score']:.4f}")
            score_diff = abs(model1['score'] - model2['score'])
            if model1['score'] > model2['score']:
                print(f"'{model1['llm_model_name']}' performed better by {score_diff:.4f} points.")
            elif model2['score'] > model1['score']:
                print(f"'{model2['llm_model_name']}' performed better by {score_diff:.4f} points.")
            else:
                print("Both models performed equally.")
            print("~"*80 + "\n")
