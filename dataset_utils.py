"""
Simple Dataset Utilities for Bio-Medical AI Competition

This module contains only the essential CureBenchDataset class and related utilities
for loading bio-medical datasets in the competition starter kit.

Note: Data should be preprocessed using preprocess_data.py to add dataset_type fields
before using this module.
"""

import json
import os
import sys

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Warning: PyTorch not available. Some features may not work.")
    # Create dummy classes for basic functionality
    class Dataset:
        pass

    class DataLoader:
        def __init__(self, *args, **kwargs):
            pass


def read_and_process_json_file(file_path):
    """
    Reads a JSON file and processes it into a standardized format.
    Handles both single JSON objects and line-delimited JSON files.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # Try to read as line-delimited JSON first
            try:
                data = [json.loads(line) for line in file if line.strip()]
                # If first item is a list, flatten it
                if data and isinstance(data[0], list):
                    data = [item for sublist in data for item in sublist]
                return data
            except json.JSONDecodeError:
                # If that fails, try reading as single JSON object
                file.seek(0)
                content = file.read()
                data = json.loads(content)
                return data

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error reading {file_path}: {e}")
        return []


class CureBenchDataset(Dataset):
    """
    Dataset class for CureBench data.

    Supports:
    - Multiple choice questions
    - Open-ended questions
    - Open-ended questions with multiple-choice mapping (meta_question)
    """

    def __init__(self, json_file):
        """
        Initialize dataset.

        Args:
            json_file (str): Path to the JSON/JSONL file containing CureBench data
        """
        self.data = read_and_process_json_file(json_file)

        if not self.data:
            print(f"Warning: No data loaded from {json_file}")
            self.data = []
            return

        print(f"CureBenchDataset initialized with {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.

        Returns:
            Tuple: (question_type, id, question, answer, meta_question)
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        item = self.data[idx]

        question_type = item["question_type"]
        question = item.get("question", "")
        answer = item.get("correct_answer", item.get("answer", ""))
        id_value = item["id"]
        meta_question = ""

        if question_type == "multi_choice":
            options = item["options"]
            options_list = "\n".join([f"{opt}: {options[opt]}" for opt in sorted(options.keys())])
            question = f"{question}\n{options_list}"
            meta_question = ""
            return question_type, id_value, question, answer, meta_question

        elif question_type == "open_ended_multi_choice":
            options = item["options"]
            options_list = "\n".join([f"{opt}: {options[opt]}" for opt in sorted(options.keys())])
            # Keep question plain; eval_framework will decide prompting
            meta_question = (
                "The following is a multiple choice question about medicine and the agent's open-ended answer. "
                "Convert the agent's answer to the final answer format using the corresponding option label "
                "(A, B, C, D, or E).\n\n"
                f"Question: {question}\n{options_list}\n\n"
            )
            return question_type, id_value, question, answer, meta_question

        elif question_type == "open_ended":
            # Keep question plain; eval_framework will decide prompting
            meta_question = ""
            return question_type, id_value, question, answer, meta_question

        else:
            raise ValueError(f"Unsupported question type: {question_type}")


def build_dataset(dataset_path=None):
    """
    Build and return CureBenchDataset.
    """
    print("dataset_path:", dataset_path)
    dataset = CureBenchDataset(dataset_path)
    return dataset
