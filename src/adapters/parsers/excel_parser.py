

import pandas as pd
from io import BytesIO
from typing import List
class ExcelParser:
    def parse_excel_to_evaluation_data(self, file_content: bytes, sheet_name: str = None) -> List[dict]:
        """
        Parses an Excel file and extracts 'input_text' and 'expected_output'.
        Returns a list of dictionaries.
        """
        try:
            df = pd.read_excel(BytesIO(file_content), sheet_name=sheet_name)
            required_columns = ['input_text', 'expected_output']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Excel file must contain '{required_columns[0]}' and '{required_columns[1]}' columns.")

            parsed_data = []
            for index, row in df.iterrows():
                parsed_data.append({
                    "input_text": str(row['input_text']),
                    "expected_output": str(row['expected_output'])
                })
            return parsed_data
        except Exception as e:
            raise Exception(f"Failed to parse Excel file: {e}")