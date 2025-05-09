from typing import List
import json
from utils.enums_dcs import TSData

def load_tsdata_list(file_path: str) -> List[TSData]:
    """
    Load a list of TSData objects from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[TSData]: List of TSData objects.
    """
    ts_data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ts_data_list.append(TSData(**data))
    return ts_data_list
