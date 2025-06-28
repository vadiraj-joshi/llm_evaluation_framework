import pandas as pd
from io import BytesIO
from typing import List, Dict, Any, Optional

class ExcelParser:
    """
    Parses Excel files to extract evaluation data, now supporting various columns
    for different AI tasks like summarization, translation, classification, and RAG.
    """
    def parse_excel_to_evaluation_data(self, file_content: bytes, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parses an Excel file and extracts relevant columns for evaluation.
        Expected columns: 'input_text', 'expected_output'.
        Optional columns: 'context' (for RAG), 'labels' (for classification, comma-separated).

        Scalability Note: For very large Excel files (e.g., millions of rows),
        reading the entire file into memory using `pd.read_excel` might be inefficient or
        lead to out-of-memory errors. For such cases, consider:
        1. Streaming parsers (e.g., `openpyxl.read_only` mode) to process row by row.
        2. Processing in chunks using `pd.read_excel(..., chunksize=...)`.
        3. Storing large datasets in a dedicated data lake/warehouse and processing them
           with distributed computing frameworks.

        :param file_content: The binary content of the Excel file.
        :param sheet_name: Optional. The specific sheet name to read from.
        :return: A list of dictionaries, each containing extracted data.
        :raises Exception: If parsing fails or required columns are missing.
        """
        try:
            df = pd.read_excel(BytesIO(file_content), sheet_name=sheet_name)
            required_columns = ['input_text', 'expected_output']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Excel file must contain '{required_columns[0]}' and '{required_columns[1]}' columns.")

            parsed_data = []
            for index, row in df.iterrows():
                row_data = {
                    "input_text": str(row['input_text']),
                    "expected_output": str(row['expected_output'])
                }
                # Optional columns for specific tasks
                if 'context' in df.columns and pd.notna(row['context']):
                    row_data['context'] = str(row['context'])
                if 'labels' in df.columns and pd.notna(row['labels']):
                    # Assuming labels are comma-separated in the Excel cell
                    row_data['labels'] = [label.strip() for label in str(row['labels']).split(',')]
                
                parsed_data.append(row_data)
            return parsed_data
        except Exception as e:
            raise Exception(f"Failed to parse Excel file: {e}")
