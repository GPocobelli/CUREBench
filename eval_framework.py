"""
Bio-Medical AI Competition Starter Kit

A simple framework for evaluating models on bio-medical datasets.
"""

import json
import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -------------------------
# Prompt templates (override via environment variables)
# -------------------------

DEFAULT_PROMPT_MC = (
    "You are a medical expert.\n"
    "Before answering, analyze the question step by step and break it down into sub-problems.\n"
    "Do NOT reveal your step-by-step reasoning.\n\n"
    "TASK: Select the single best option.\n"
    "STRICT OUTPUT: Return EXACTLY ONE LETTER: A, B, C, D, or E. No punctuation, no other text.\n\n"
    "Question:\n{question}\n\n"
    "Final answer (one letter only):"
)

DEFAULT_PROMPT_OE = (
    "You are a medical expert.\n"
    "Before answering, analyze the question step by step and break it down into sub-problems.\n"
    "Do NOT reveal your step-by-step reasoning.\n\n"
    "Provide a clinically grounded answer.\n"
    "Output format:\n"
    "1) Recommendation (1-3 sentences)\n"
    "2) Key rationale (3-5 bullet points)\n"
    "3) Safety/uncertainty note (1 sentence, only if needed)\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

DEFAULT_PROMPT_OEMC_META = (
    "You are a medical expert evaluator.\n"
    "Before answering, analyze step by step internally.\n"
    "Do NOT reveal reasoning.\n\n"
    "TASK: Map the agent's free-form answer to the best multiple-choice label.\n"
    "STRICT OUTPUT: Return EXACTLY ONE LETTER: A, B, C, D, or E. No other text.\n\n"
    "{meta_question}"
    "Agent answer:\n{agent_answer}\n\n"
    "Final answer (one letter only):"
)


def _get_env_prompt(name: str, default: str) -> str:
    val = os.getenv(name, "").strip()
    return val if val else default


PROMPT_MC = _get_env_prompt("CURE_PROMPT_MC", DEFAULT_PROMPT_MC)
PROMPT_OE = _get_env_prompt("CURE_PROMPT_OE", DEFAULT_PROMPT_OE)
PROMPT_OEMC_META = _get_env_prompt("CURE_PROMPT_OEMC_META", DEFAULT_PROMPT_OEMC_META)


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]
    reasoning_traces: List[Any] = None
    details: Optional[Dict] = None


# -------------------------
# Model Interfaces
# -------------------------

class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, **kwargs):
        pass

    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        pass


class ChatGPTModel(BaseModel):
    """ChatGPT/OpenAI model wrapper (Azure OpenAI in this starter kit)"""

    def load(self, **kwargs):
        api_key = os.getenv("AZURE_OPENAI_API_KEY_O1")
        api_version = "2024-12-01-preview"
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key:
            raise ValueError("API key not found in environment. Set AZURE_OPENAI_API_KEY_O1.")
        if not azure_endpoint:
            raise ValueError("Azure endpoint not found in environment. Set AZURE_OPENAI_ENDPOINT.")

        from openai import AzureOpenAI
        logger.info(f"Initializing AzureOpenAI client with endpoint: {azure_endpoint}")
        logger.info(f"Using API version: {api_version}")
        self.model_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        messages = [{"role": "user", "content": prompt}]
        responses = self.model_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        response = responses.choices[0].message.content
        complete_messages = messages + [{"role": "assistant", "content": response}]
        return response, complete_messages


class LocalModel(BaseModel):
    """Local HuggingFace model wrapper"""

    def load(self, **kwargs):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # IMPORTANT:
        # - Do not force BitsAndBytesConfig by default on Kaggle (often not available / breaks).
        # - Allow user to pass quantization_config via kwargs if they know what they are doing.
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        model_kwargs.update(kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        logger.info(f"Loaded local model: {self.model_name}")

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )

        response = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)

        complete_messages = messages + [{"role": "assistant", "content": response_text}]
        return response_text, complete_messages


class CustomModel(BaseModel):
    """Custom model wrapper for user-defined models"""

    def __init__(self, model_name: str, model_instance, inference_func):
        super().__init__(model_name)
        self.model = model_instance
        self._inference_func = inference_func

    def load(self, **kwargs):
        logger.info(f"Using custom model: {self.model_name}")

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self._inference_func(self.model, prompt, max_tokens)
            complete_messages = messages + [{"role": "assistant", "content": response}]
            return response, complete_messages
        except Exception as e:
            logger.error(f"Custom model inference error: {e}")
            error_messages = messages + [{"role": "assistant", "content": "Error occurred"}]
            return "Error occurred", error_messages


class GPTOSS20BModel(BaseModel):
    """GPT-OSS-20B wrapper"""

    def __init__(
        self,
        model_name: str,
        quantization: str = "auto",
        reasoning_lvl: str = "medium",
        system_identity: str = None,
        developer_instructions: str = None,
    ):
        super().__init__(model_name)
        self.quantization = quantization
        self.enc = None
        self.reasoning_lvl = reasoning_lvl
        self.system_identity = system_identity
        self.developer_instructions = developer_instructions

    def load(self, **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        from openai_harmony import load_harmony_encoding, HarmonyEncodingName

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quantization == "fp16":
            torch_dtype = torch.float16
        elif self.quantization == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"

        model_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto"}
        model_kwargs.update(kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def inference(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        builtin_tools: Optional[List[str]] = None,
        tools: Optional[List[dict]] = None,
    ) -> Tuple[str, List[Dict]]:

        from openai_harmony import Role
        from transformers import AutoTokenizer
        import logging

        messages = []
        if self.system_identity or self.reasoning_lvl:
            sys_content = ""
            if self.system_identity:
                sys_content += self.system_identity + "\n"
            sys_content += f"Reasoning: {self.reasoning_lvl}."
            messages.append({"role": "system", "content": sys_content})

        if self.developer_instructions:
            messages.append({"role": "developer", "content": self.developer_instructions})

        messages.append({"role": "user", "content": prompt})

        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                reasoning_effort=self.reasoning_lvl,
                model_identity=self.system_identity
                or "You are ChatGPT, a large language model trained by OpenAI.",
                builtin_tools=builtin_tools,
                tools=tools,
            ).to(self.model.device)
        except Exception as e:
            logging.warning(
                f"[WARN] chat_template in {self.model_name} failed "
                f"({type(e).__name__}: {e}). Falling back to base GPT-OSS template."
            )
            base_tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
            self.tokenizer.chat_template = base_tok.chat_template
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                reasoning_effort=self.reasoning_lvl,
                model_identity=self.system_identity
                or "You are ChatGPT, a large language model trained by OpenAI.",
                builtin_tools=builtin_tools,
                tools=tools,
            ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            eos_token_id=None if not self.enc else self.enc.stop_tokens()[-1],
        )

        gen_tokens = outputs[0][input_ids.shape[-1]:].tolist()

        try:
            parsed = self.enc.parse_messages_from_completion_tokens(gen_tokens, role=Role.ASSISTANT)
            reasoning_trace = [msg.to_dict() for msg in parsed]

            finals = [msg for msg in parsed if msg.to_dict().get("channel") == "final"]
            if finals:
                final_response = "".join(
                    c.text for c in finals[-1].content if hasattr(c, "text")
                )
            else:
                final_response = "".join(
                    c.text for c in parsed[-1].content if hasattr(c, "text")
                )

        except Exception as e:
            logging.error(f"[Harmony parse error] {e}")
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            final_response = text
            reasoning_trace = [{"role": "assistant", "content": text}]

        return final_response.strip(), reasoning_trace


# -------------------------
# Competition Kit
# -------------------------

class CompetitionKit:
    """Simple competition framework"""

    def __init__(self, config_path: str = None):
        self.model = None
        self.model_name = None
        self.config = json.load(open(config_path, "r")) if config_path else {}
        self.output_dir = self.config.get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)
        self.datasets = self._load_dataset_configs(self.config)

    def load_model(self, model_name: str, model_type: str = "auto", **kwargs):
        self.model_name = model_name

        if model_type == "auto":
            model_type = self._detect_model_type(model_name)

        logger.info(f"Loading model: {model_name} (type: {model_type})")

        if model_type == "chatgpt":
            self.model = ChatGPTModel(model_name)
        elif model_type == "gpt-oss-20b":
            self.model = GPTOSS20BModel(model_name)
        elif model_type == "local":
            self.model = LocalModel(model_name)
        elif model_type == "custom":
            model_instance = kwargs.get("model_instance")
            inference_func = kwargs.get("inference_func")
            if not model_instance or not inference_func:
                raise ValueError("Custom model requires 'model_instance' and 'inference_func'")
            self.model = CustomModel(model_name, model_instance, inference_func)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.load(**kwargs)

    def _load_dataset_configs(self, config) -> Dict:
        if not config:
            print("No config provided, exiting.")
            exit(1)

        if "dataset" in config:
            dataset_config = config["dataset"]
            dataset_name = dataset_config.get("dataset_name", "cure_bench_phase_1")
            return {dataset_name: dataset_config}

        print("No dataset found in config, exiting.")
        exit(1)

    def _detect_model_type(self, model_name: str) -> str:
        if "gpt-oss-20b" in model_name.lower():
            return "gpt-oss-20b"
        if any(name in model_name.lower() for name in ["gpt", "chatgpt", "openai", "o1", "o3", "o4"]):
            return "chatgpt"
        return "local"

    def evaluate(self, dataset_name: str, subset_size: int = None) -> EvaluationResult:
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")

        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")

        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config.get('description','')}")

        dataset = self._load_dataset(dataset_config)
        self._last_dataset_examples = dataset

        if subset_size is not None and subset_size > 0:
            dataset = dataset[:subset_size]
            logger.info(f"Subset size applied: {len(dataset)} examples")

        predictions = []
        reasoning_traces = []
        total_count = len(dataset)

        accuracy_correct_count = 0
        accuracy_total_count = 0

        logger.info(f"Running evaluation on {total_count} examples...")
        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                prediction, reasoning_trace = self._get_prediction_with_trace(example)
                predictions.append(prediction)
                reasoning_traces.append(reasoning_trace)

                question_type = example["question_type"]
                expected_answer = example.get("answer", "")

                if question_type in ["multi_choice", "open_ended_multi_choice"]:
                    if expected_answer != "":
                        is_correct = prediction["choice"] == expected_answer
                    else:
                        is_correct = False
                    accuracy_total_count += 1
                    if is_correct:
                        accuracy_correct_count += 1

            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                predictions.append({"choice": "NOTAVALUE", "open_ended_answer": "Error"})
                reasoning_traces.append("Error occurred during inference")

        accuracy = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0

        result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,
            total_examples=accuracy_total_count,
            predictions=predictions,
            reasoning_traces=reasoning_traces,
        )

        logger.info(
            f"Evaluation completed: {accuracy:.2%} accuracy ({accuracy_correct_count}/{accuracy_total_count}) "
            f"- excluding open-ended"
        )
        return result

    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        from dataset_utils import build_dataset
        from torch.utils.data import DataLoader

        dataset = build_dataset(dataset_config.get("dataset_path"))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_list = []

        for batch in dataloader:
            question_type = batch[0][0]
            if question_type == "multi_choice":
                dataset_list.append(
                    {
                        "question_type": batch[0][0],
                        "id": batch[1][0],
                        "question": batch[2][0],
                        "answer": batch[3][0],
                    }
                )
            elif question_type == "open_ended_multi_choice":
                dataset_list.append(
                    {
                        "question_type": batch[0][0],
                        "id": batch[1][0],
                        "question": batch[2][0],
                        "answer": batch[3][0],
                        "meta_question": batch[4][0],
                    }
                )
            elif question_type == "open_ended":
                dataset_list.append(
                    {
                        "question_type": batch[0][0],
                        "id": batch[1][0],
                        "question": batch[2][0],
                        "answer": batch[3][0],
                    }
                )

        return dataset_list

    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, Any]:
        question = example["question"]
        question_type = example["question_type"]

        if question_type == "multi_choice":
            prompt = PROMPT_MC.format(question=question)
        elif question_type == "open_ended":
            prompt = PROMPT_OE.format(question=question)
        elif question_type == "open_ended_multi_choice":
            prompt = PROMPT_OE.format(question=question)
        else:
            raise ValueError(f"Unsupported question_type: {question_type}")

        response, reasoning_trace = self.model.inference(prompt)

        prediction = {"choice": "", "open_ended_answer": ""}

        if question_type == "multi_choice":
            choice = self._extract_multiple_choice_answer(response)
            prediction["choice"] = choice if choice else ""
            prediction["open_ended_answer"] = response.strip()

        elif question_type == "open_ended":
            prediction["choice"] = "NOTAVALUE"
            prediction["open_ended_answer"] = response.strip()

        elif question_type == "open_ended_multi_choice":
            prediction["open_ended_answer"] = response.strip()

            meta_question = example.get("meta_question", "")
            if meta_question:
                meta_prompt = PROMPT_OEMC_META.format(meta_question=meta_question, agent_answer=response.strip())
                meta_response, meta_reasoning = self.model.inference(meta_prompt)
                try:
                    reasoning_trace = list(reasoning_trace) + list(meta_reasoning)
                except Exception:
                    pass
                choice = self._extract_multiple_choice_answer(meta_response)
                prediction["choice"] = choice if choice else ""
            else:
                choice = self._extract_multiple_choice_answer(response)
                prediction["choice"] = choice if choice else ""

        return prediction, reasoning_trace

    def _extract_multiple_choice_answer(self, response: str) -> str:
        if not response:
            return ""

        r = response.strip().upper()

        if len(r) >= 1 and r[0] in ["A", "B", "C", "D", "E"]:
            return r[0]

        import re
        patterns = [
            r"(?:FINAL ANSWER|ANSWER)\s*[:\-]?\s*([ABCDE])\b",
            r"(?:ANSWER IS|THE ANSWER IS|IS)\s*([ABCDE])\b",
            r"\b([ABCDE])\)",
            r"\b([ABCDE])\b",
        ]
        for p in patterns:
            m = re.search(p, r)
            if m:
                return m.group(1)

        return ""

    def save_submission(
        self,
        results: List[EvaluationResult],
        filename: str = "submission.csv",
        metadata: Dict = None,
        dataset_examples: List[Dict] = None,
        config_path: str = None,
        args: argparse.Namespace = None,
    ):
        import pandas as pd
        import zipfile

        metadata = self.get_metadata(config_path, args, metadata)

        submission_data = []
        for result in results:
            examples = dataset_examples if dataset_examples else []
            for i, (prediction, example) in enumerate(zip(result.predictions, examples)):
                reasoning_trace = json.dumps(result.reasoning_traces[i]) if result.reasoning_traces else "No reasoning available"

                prediction_text = prediction.get("open_ended_answer", "") or ""
                if not prediction_text.strip():
                    prediction_text = "No prediction available"

                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ["NULL", "NONE", "NAN"] or str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"
                else:
                    choice_clean = str(choice_raw).strip()

                if not reasoning_trace or reasoning_trace.strip() == "" or reasoning_trace.strip().lower() == "null":
                    reasoning_trace = "No reasoning available"

                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace),
                }
                submission_data.append(row)

        df = pd.DataFrame(submission_data)
        for col in df.columns:
            df[col] = df[col].astype(str)

        # Strong null cleaning
        df["choice"] = df["choice"].replace(["", " ", "None", "none", "NULL", "null", "NaN", "nan", "<NA>"], "NOTAVALUE")
        df["prediction"] = df["prediction"].replace(["", " "], "No prediction available")
        df["reasoning"] = df["reasoning"].replace(["", " "], "No reasoning available")

        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=False, na_rep="NOTAVALUE", quoting=1)

        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        zip_filename = filename.replace(".csv", ".zip")
        zip_path = os.path.join(self.output_dir, zip_filename)

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, filename)
            zipf.write(metadata_path, metadata_filename)

        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(f"Overall accuracy (excluding open-ended questions): {overall_accuracy:.2%} ({total_correct}/{total_examples})")

        return zip_path

    def save_submission_with_metadata(
        self,
        results: List[EvaluationResult],
        metadata: Dict = None,
        filename: str = "submission.csv",
        config_path: str = None,
        args: argparse.Namespace = None,
    ):
        dataset_examples = getattr(self, "_last_dataset_examples", [])
        return self.save_submission(results, filename, metadata, dataset_examples, config_path, args)

    def load_metadata_from_config(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        _, ext = os.path.splitext(config_path)
        with open(config_path, "r") as f:
            if ext.lower() in [".json"]:
                config = json.load(f)
            elif ext.lower() in [".yaml", ".yml"]:
                import yaml
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {ext}")

        metadata = config.get("metadata", config.get("meta_data", {}))
        return metadata

    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        metadata = {}
        arg_mapping = {
            "model_name": "model_name",
            "model_type": "model_type",
            "track": "track",
            "base_model_type": "base_model_type",
            "base_model_name": "base_model_name",
            "dataset": "dataset",
            "additional_info": "additional_info",
        }
        for arg_name, meta_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                metadata[meta_field] = getattr(args, arg_name)
        return metadata

    def get_metadata(self, config_path: str = None, args: argparse.Namespace = None, fallback_metadata: Dict = None) -> Dict:
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework",
        }

        if fallback_metadata:
            metadata.update(fallback_metadata)

        if config_path:
            try:
                config_metadata = self.load_metadata_from_config(config_path)
                metadata.update(config_metadata)
                logger.info(f"Loaded metadata from config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")

        if args:
            arg_metadata = self.parse_metadata_from_args(args)
            metadata.update(arg_metadata)

        return metadata


def create_metadata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation Framework with Metadata Support")

    parser.add_argument("--model-name", type=str, help="Name of the model")
    parser.add_argument("--model-type", type=str, help="Type of model wrapper")
    parser.add_argument("--base-model-name", type=str, help="Name of the base model")
    parser.add_argument("--base-model-type", type=str, choices=["API", "OpenWeighted"], help="Type of base model")
    parser.add_argument("--track", type=str, choices=["internal_reasoning", "agentic_reasoning"], default="internal_reasoning")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--additional-info", type=str, help="Additional information about the submission")
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON or YAML)")
    parser.add_argument("--output-dir", type=str, default="competition_results", help="Output directory")
    parser.add_argument("--output-file", type=str, default="submission.csv", help="Output CSV filename")
    parser.add_argument("--subset-size", type=int, help="Limit evaluation to N examples")

    return parser


def load_config_file(config_path):
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_and_merge_config(args):
    if not args.config:
        return args

    config = load_config_file(args.config)

    if "metadata" in config:
        md = config["metadata"]
        for key, value in md.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    def add_config_to_args(config_dict, prefix=""):
        for key, value in config_dict.items():
            if key in ["metadata", "dataset"]:
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)

    add_config_to_args(config)
    return args
