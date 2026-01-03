"""
Bio-Medical AI Competition Starter Kit

A simple framework for evaluating models on bio-medical datasets.
Perfect for getting started quickly in the competition.

Key Features:
- Easy model loading (ChatGPT, GPT-OSS-20B, Local models, Custom models)
- Simple dataset loading
- Automatic evaluation and scoring
- Submission file generation

Usage:
    framework = CompetitionKit()
    framework.load_model("gpt-4o-mini")
    results = framework.evaluate("quick_test")
    framework.save_submission(results, "my_submission.json")
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
import re  # needed for fallback and existing extractor patterns
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]  # Changed from List[str] to List[Dict]
    reasoning_traces: List[str] = None  # Add reasoning traces
    details: Optional[Dict] = None


# Model Classes
class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass

    @abstractmethod
    def inference(
    self,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,      # added
    top_p: float = 1.0,            # added
    top_k: Optional[int] = None,   # added
) -> Tuple[str, List[Dict]]:
        """Run inference on the model

        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class ChatGPTModel(BaseModel):
    """ChatGPT/OpenAI model wrapper"""

    def load(self, **kwargs):
        api_key = os.getenv("AZURE_OPENAI_API_KEY_O1")
        api_version = "2024-12-01-preview"

        if not api_key:
            raise ValueError("API key not found in environment. Please set the appropriate environment variable.")

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        from openai import AzureOpenAI
        logger.info(f"Initializing AzureOpenAI client with endpoint: {azure_endpoint}")
        logger.info(f"Using API version: {api_version}")

        self.model_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def inference(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,  # Azure chat.completions doesn't support top_k; kept for API compatibility
    ) -> Tuple[str, List[Dict]]:
        messages = [{"role": "user", "content": prompt}]
        responses = self.model_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,  # FIX: respect caller's max_tokens
            temperature=temperature,
            top_p=top_p,
        )
        response = responses.choices[0].message.content or ""
        complete_messages = messages + [{"role": "assistant", "content": response}]
        return response, complete_messages


# -----------------------------------------------------------------------------------------
# Qwen


class LocalModel(BaseModel):
    """Local HuggingFace model wrapper"""

    def load(self, **kwargs):
        """Load local HuggingFace model.

        Notes:
        - Tries to use 8-bit quantization only when CUDA is available and bitsandbytes is installed.
        - Falls back to non-quantized loading otherwise (important for CPU-only environments / Kaggle CPU).
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            model_kwargs = dict(
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model_kwargs.update(kwargs)

            # Optional 8-bit quantization (CUDA + bitsandbytes)
            use_8bit = False
            try:
                if torch.cuda.is_available():
                    from transformers import BitsAndBytesConfig  # noqa: F401
                    import bitsandbytes  # noqa: F401
                    use_8bit = True
            except Exception:
                use_8bit = False

            if use_8bit:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            logger.info(f"Loaded local model: {self.model_name} (8bit={use_8bit})")

        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise

    def inference(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
    ):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Some tokenizers don't support enable_thinking; fall back gracefully.
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.model.device)
        except TypeError:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if temperature and temperature > 0:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)
        else:
            gen_kwargs.update(dict(do_sample=False))

        outputs = self.model.generate(input_ids, **gen_kwargs)
        response = outputs[0][input_ids.shape[-1] :]
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
        """Custom models are already loaded"""
        logger.info(f"Using custom model: {self.model_name}")

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Custom model inference"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._inference_func(self.model, prompt, max_tokens)
            complete_messages = messages + [{"role": "assistant", "content": response}]
            return response, complete_messages
        except Exception as e:
            logger.error(f"Custom model inference error: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Error occurred"}
            ]
            return "Error occurred", error_messages


class GPTOSS20BModel(BaseModel):
    """GPT-OSS-20B wrapper"""

    def __init__(
        self,
        model_name: str,
        quantization: str = "auto",          # auto | fp16 | bf16 | 8bit
        reasoning_lvl: str = "medium",       # low | medium | high
        system_identity: str = None,         # optional system override
        developer_instructions: str = None,  # optional developer message
    ):
        super().__init__(model_name)
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.enc = None
        self.reasoning_lvl = reasoning_lvl
        self.system_identity = system_identity
        self.developer_instructions = developer_instructions

    def load(self, **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        from openai_harmony import load_harmony_encoding, HarmonyEncodingName

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quantization == "fp16":
            torch_dtype = torch.float16
            quant_config = None
        elif self.quantization == "bf16":
            torch_dtype = torch.bfloat16
            quant_config = None
        elif self.quantization == "8bit":
            torch_dtype = torch.bfloat16
            quant_config = None
        else:
            torch_dtype = "auto"
            quant_config = None

        model_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto", **kwargs}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

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
        import logging
        from transformers import AutoTokenizer

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
                f"[WARN] Custom chat_template in {self.model_name} failed "
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
                final_response = "".join(c.text for c in finals[-1].content if hasattr(c, "text"))
            else:
                final_response = "".join(c.text for c in parsed[-1].content if hasattr(c, "text"))

        except Exception as e:
            logging.error(f"[Harmony parse error] {e}")
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            final_response = text
            reasoning_trace = [{"role": "assistant", "content": text}]

        return final_response.strip(), reasoning_trace


class CompetitionKit:
    """
    Simple competition framework - everything you need in one class!
    """

    def __init__(self, config_path: str = None):
        self.model = None
        self.model_name = None

        self.config = json.load(open(config_path, 'r')) if config_path else {}

        self.self_consistency = self.config.get("self_consistency", {})
        self.sc_enabled = bool(self.self_consistency.get("enabled", False))
        self.sc_num_paths = int(self.self_consistency.get("num_paths", 10))
        self.sc_temperature = float(self.self_consistency.get("temperature", 0.7))
        self.sc_top_k = self.self_consistency.get("top_k", 40)  # can be None
        self.sc_top_p = float(self.self_consistency.get("top_p", 1.0))

        self.output_dir = self.config.get('output_dir', 'results')
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
                raise ValueError("Custom model requires 'model_instance' and 'inference_func' parameters")
            self.model = CustomModel(model_name, model_instance, inference_func)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.load(**kwargs)

    def _load_dataset_configs(self, config) -> Dict:
        if not config:
            print("Not config provided, existing.")
            exit(1)

        if 'dataset' in config:
            dataset_config = config['dataset']
            dataset_name = dataset_config.get('dataset_name', 'treatment')
            return {dataset_name: dataset_config}
        else:
            print("Not config found, existing.")
            exit(1)

    def _detect_model_type(self, model_name: str) -> str:
        if "gpt-oss-20b" in model_name.lower():
            return "gpt-oss-20b"
        if any(name in model_name.lower() for name in ["gpt", "chatgpt", "openai", 'o1', 'o3', 'o4']):
            return "chatgpt"
        else:
            return "local"

    def evaluate(self, dataset_name: str, subset_size: int = None) -> EvaluationResult:
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")

        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")

        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config['description']}")

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

                is_correct = False
                question_type = example["question_type"]
                expected_answer = example.get("answer")
                print("expected_answer:", expected_answer)

                if question_type == "multi_choice" or question_type == "open_ended_multi_choice":
                    if expected_answer != '':
                        is_correct = prediction["choice"] == expected_answer
                    else:
                        is_correct = False
                    accuracy_total_count += 1
                    if is_correct:
                        accuracy_correct_count += 1
                elif question_type == "open_ended":
                    if expected_answer != '':
                        is_correct = prediction["open_ended_answer"] == expected_answer
                    else:
                        is_correct = False

                if (i + 1) % 10 == 0:
                    current_acc = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
                    logger.info(f"Progress: {i+1}/{total_count}, Accuracy: {current_acc:.2%} (excluding open-ended)")

            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                error_prediction = {
                    "choice": "NOTAVALUE",
                    "open_ended_answer": "Error"
                }
                predictions.append(error_prediction)
                reasoning_traces.append("Error occurred during inference")

        accuracy = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0

        result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,
            total_examples=accuracy_total_count,
            predictions=predictions,
            reasoning_traces=reasoning_traces
        )

        logger.info(
            f"Evaluation completed: {accuracy:.2%} accuracy ({accuracy_correct_count}/{accuracy_total_count}) - excluding open-ended questions"
        )
        logger.info(f"Total examples processed: {total_count} (including {total_count - accuracy_total_count} open-ended questions)")

        return result

    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        from dataset_utils import build_dataset
        from torch.utils.data import DataLoader

        dataset = build_dataset(
            dataset_config.get("dataset_path"),
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_list = []

        for batch in dataloader:
            question_type = batch[0][0]

            if question_type == "multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })
            elif question_type == "open_ended_multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                    "meta_question": batch[4][0],
                })
            elif question_type == "open_ended":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })

        return dataset_list















    
    # =====================================================================
    # FIXED: CoT-safe prompts + syntactically safe meta_prompt
    # - Replaces _get_prediction_with_trace with CoT-safe prompts
    # - Uses a syntactically safe meta_prompt (no triple-quote nesting)
    # - Keeps _extract_multiple_choice_answer intact
    # =====================================================================
    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
        prediction = {"choice": "", "open_ended_answer": ""}
        question = example["question"]
        question_type = example["question_type"]



        # ------------------------------------------------------------------------------------
        # multi_choice
        # ------------------------------------------------------------------------------------
        
        if question_type == "multi_choice":
            prompt = (
                "The following is a multi_choice question.\n"
                "You are a medical expert that answers multiple choice questions about medical knowledge.\n\n"
                "INSTRUCTIONS:\n"
                "- Before answering, let's think step by step and break it down into sub-problems.\n"
                "- Then provide the final answer in exactly this format:\n"
                "  The answer is X.\n"
                "  where X is one of A, B, C, D, E.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
            )

            
            if self.sc_enabled:
               # Self-Consistency: sample multiple reasoning paths, then majority vote on final answers.
                responses, traces = self._sample_responses(prompt, self.sc_num_paths)
                
                choices = [self._extract_multiple_choice_answer(r) for r in responses]
                final_choice = self._majority_vote(choices)
    
                prediction["choice"] = final_choice if final_choice else ""
                # store a representative string (optional): first response matching final_choice
                rep = ""
                for r, c in zip(responses, choices):
                    if c == final_choice and r:
                        rep = r
                        break
                prediction["open_ended_answer"] = rep or (responses[0] if responses else "")
    
                reasoning_trace = {
                    "self_consistency": True,
                    "num_paths": self.sc_num_paths,
                    "votes": dict(Counter(choices)),
                    "samples": [
                        {"response": r, "choice": c, "trace": t}
                        for r, c, t in zip(responses, choices, traces)
                    ],
                }
                return prediction, reasoning_trace
    
            # --- fallback: single decode (your current behavior) ---
            response, reasoning_trace = self.model.inference(prompt, temperature=0.0)
            choice = self._extract_multiple_choice_answer(response)
            prediction["choice"] = choice if choice else ""
            prediction["open_ended_answer"] = (response or "").strip()
            return prediction, reasoning_trace





        
        # ------------------------------------------------------------------------------------
        # OPEN-ENDED 
        # ------------------------------------------------------------------------------------
    
        elif question_type == "open_ended":
            prompt = (
                "The following is a open_ended question.\n"
                "You are a medical expert that answers open-end questions about medical knowledge.\n\n"
                "REASONING (internal):\n"
                "Before answering, let's think step by step and break it down into sub-problems.\n"
                "OUTPUT FORMAT:\n"
                "1) Recommendation (1–3 sentences)\n"
                "2) Key rationale (3–5 bullet points)\n"
                "3) Safety/contraindications/interactions (bullet points, if relevant)\n"
                "If unsure, state assumptions and missing information.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "ANSWER:"
            )
            response, reasoning_trace = self.model.inference(prompt, temperature=0.0)
            prediction["choice"] = "NOTAVALUE"
            prediction["open_ended_answer"] = (response or "").strip()
            return prediction, reasoning_trace



        
        # ------------------------------------------------------------------------------------
        # open_ended_multi_choice 
        # ------------------------------------------------------------------------------------
        
        elif question_type == "open_ended_multi_choice":
            prompt = (
                "The following is a open_ended_multi_choice question.\n"
                "You are a medical expert that answers open-end questions about medical knowledge.\n\n"
                "REASONING (internal):\n"
                "Before answering, let's think step by step and break it down into sub-problems.\n"
                "INSTRUCTIONS:\n"
                "- Answer in 1-3 bullet points.\n"
                "Each bullet MUST include concrete identifiers from the options (e.g. drug names, regimen, placebo vs active).\n"
                "- Do NOT mention option letters (A, B, C, D, or E) in this step.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "ANSWER:"
            )

            if self.sc_enabled:
                # sample diverse agent answers
                responses, traces = self._sample_responses(prompt, self.sc_num_paths)
    
                # If meta_question exists, we map each sampled response to a letter deterministically
                if "meta_question" in example:
                    choices = []
                    meta_traces = []
    
                    for r in responses:
                        meta_prompt = (
                            f"{example['meta_question']}\n"
                            "You are a helpful assistant who reviews an open-end answer.\n\n"
                            "Given the agent answer below, choose the single best matching option.\n"
                            "Analyze it and choose the single option (A, B, C, D, or E) whose text best matches the agent answer.\n\n"
                            "If uncertain, pick the closest match.\n"
                            "Let's think step by step.\n"

                            "STRICT OUTPUT:\n"
                            "Return exactly one letter A, B, C, D, or E.\n"
                            "No extra text.\n\n"
                            "Agent's answer:\n"
                            f"{r}\n\n"
                        )

                        # mapping can be greedy (temperature=0) to reduce noise in the mapper
                        mr, mt = self.model.inference(meta_prompt, temperature=0.0)
                        meta_traces.append(mt)
                        choices.append(self._extract_multiple_choice_answer(mr))
    
                    final_choice = self._majority_vote(choices)
                    prediction["choice"] = final_choice if final_choice else ""
    
                    # representative open answer whose mapped choice equals the final vote
                    rep = ""
                    for r, c in zip(responses, choices):
                        if c == final_choice and r:
                            rep = r
                            break
                    prediction["open_ended_answer"] = rep or (responses[0] if responses else "")
    
                    reasoning_trace = {
                        "self_consistency": True,
                        "num_paths": self.sc_num_paths,
                        "votes": dict(Counter(choices)),
                        "samples": [
                            {
                                "response": r,
                                "mapped_choice": c,
                                "trace": t,
                                "meta_trace": mt,
                            }
                            for r, c, t, mt in zip(responses, choices, traces, meta_traces)
                        ],
                    }
                    return prediction, reasoning_trace
    
                # no meta_question: fall back to extracting choice directly from responses
                choices = [self._extract_multiple_choice_answer(r) for r in responses]
                final_choice = self._majority_vote(choices)
                prediction["choice"] = final_choice if final_choice else ""
                prediction["open_ended_answer"] = responses[0] if responses else ""
                reasoning_trace = {
                    "self_consistency": True,
                    "num_paths": self.sc_num_paths,
                    "votes": dict(Counter(choices)),
                    "samples": [
                        {"response": r, "choice": c, "trace": t}
                        for r, c, t in zip(responses, choices, traces)
                    ],
                }
                return prediction, reasoning_trace
    
            # --- fallback single decode (your current behavior) ---
            response, reasoning_trace = self.model.inference(prompt, temperature=0.0)
            prediction["open_ended_answer"] = (response or "").strip()

            if "meta_question" in example:
                meta_prompt = (
                    f"{example['meta_question']}\n"
                    "You are a helpful assistant who reviews an open-end answer.\n\n"
                    "Given the agent answer below, choose the single best matching option.\n"
                    "Analyze it and choose the single option (A, B, C, D, or E) whose text best matches the agent answer.\n\n"
                    "If uncertain, pick the closest match.\n"
                    "Let's think step by step.\n"
                    "STRICT OUTPUT:\n"
                    "Return exactly one letter A, B, C, D, or E.\n"
                    "No extra text.\n\n"
                    "Agent's answer:\n"
                    f"{response.strip()}\n\n"
                )
                
                meta_response, meta_reasoning = self.model.inference(meta_prompt, temperature=0.0)
                # Keep traces separate but concatenate for compatibility with your existing storage
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
    
        # Safety fallback
        return prediction, "Unsupported question_type"





        
        
        
    def _majority_vote(self, answers: List[str]) -> str:
        answers = [a for a in answers if a]
        if not answers:
            return ""
        c = Counter(answers)
        # deterministic tie-break: alphabetical
        best_count = max(c.values())
        winners = sorted([k for k, v in c.items() if v == best_count])
        return winners[0]
        
    def _sample_responses(self, prompt: str, m: int) -> Tuple[List[str], List[Any]]:
        responses = []
        traces = []
        for _ in range(m):
            r, t = self.model.inference(
                prompt,
                temperature=self.sc_temperature,
                top_p=self.sc_top_p,
                top_k=self.sc_top_k if self.sc_top_k is not None else None,
            )
            responses.append((r or "").strip())
            traces.append(t)
        return responses, traces



















    def _extract_multiple_choice_answer(self, response: str) -> str:
        """Extract letter answer from model response"""
        if not response or response is None:
            return ""

        response = response.strip().upper()

        # Look for letter at the beginning
        if response and response[0] in ['A', 'B', 'C', 'D', 'E']:
            return response[0]

        # Look for "The answer is X" patterns
        import re
        patterns = [
            r"(?:answer is|answer:|is)\s*([ABCDE])",
            r"([ABCDE])\)",
            r"\b([ABCDE])\b"
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        return ""




    
    # ---------------------------------------------------------------------
    # Everything below here is unchanged from your provided script
    # ---------------------------------------------------------------------
    def save_submission(self, results: List[EvaluationResult], filename: str = "submission.csv",
                        metadata: Dict = None, dataset_examples: List[Dict] = None,
                        config_path: str = None, args: argparse.Namespace = None):
        """
        Save results in competition submission format as CSV file with metadata JSON and zip package
        """
        import pandas as pd
        import zipfile

        metadata = self.get_metadata(config_path, args, metadata)

        submission_data = []

        for result in results:
            examples = dataset_examples if dataset_examples else []

            for i, (prediction, example) in enumerate(zip(result.predictions, examples)):
                rt = result.reasoning_traces[i]
                reasoning_trace = rt if isinstance(rt, str) else json.dumps(rt)

                prediction_text = prediction.get("open_ended_answer", "") or ""
                if not prediction_text or prediction_text.strip() == "":
                    prediction_text = "No prediction available"

                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ['NULL', 'NONE', 'NAN']:
                    choice_clean = "NOTAVALUE"
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"
                else:
                    choice_clean = str(choice_raw).strip()

                if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
                    reasoning_trace = "No reasoning available"

                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace)
                }

                if str(choice_clean).upper() in ['NULL', 'NONE', 'NAN'] or str(choice_clean).strip() == "":
                    logger.warning(
                        f"Found NULL-like or empty choice for row {row['id']}: '{choice_clean}' - replacing with NOTAVALUE"
                    )
                    row["choice"] = "NOTAVALUE"

                submission_data.append(row)

        df = pd.DataFrame(submission_data)

        for col in df.columns:
            df[col] = df[col].astype(str)

        null_replacements = {
            'id': 'unknown_id',
            'prediction': 'No prediction available',
            'choice': 'NOTAVALUE',
            'reasoning': 'No reasoning available'
        }

        for col in df.columns:
            df[col] = df[col].fillna(null_replacements.get(col, 'NOTAVALUE'))

            null_like_values = ['nan', 'NaN', 'None', 'null', 'NULL', '<NA>', 'nat', 'NaT']
            for null_val in null_like_values:
                df[col] = df[col].replace(null_val, null_replacements.get(col, 'NOTAVALUE'))

            if col == 'choice':
                for null_val in null_like_values:
                    df[col] = df[col].replace(null_val, 'NOTAVALUE')
                df[col] = df[col].replace('', 'NOTAVALUE')
                df[col] = df[col].replace(' ', 'NOTAVALUE')

            if col != 'choice' and col in null_replacements:
                df[col] = df[col].replace('', null_replacements[col])
                df[col] = df[col].replace(' ', null_replacements[col])

        csv_path = os.path.join(self.output_dir, filename)

        logger.info(f"Creating CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")

        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Still found {null_count} nulls in column {col}")

        logger.info("Performing final NULL check on choice column...")
        null_patterns = ['NULL', 'null', 'None', 'NaN', 'nan', '<NA>', 'nat', 'NaT', 'NOTAVALUE']
        for pattern in null_patterns:
            count_before = (df['choice'] == pattern).sum()
            if count_before > 0:
                logger.warning(f"Found {count_before} instances of '{pattern}' in choice column, replacing with NOTAVALUE")
                df['choice'] = df['choice'].replace(pattern, 'NOTAVALUE')

        empty_count = (df['choice'] == '').sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty strings in choice column, replacing with NOTAVALUE")
            df['choice'] = df['choice'].replace('', 'NOTAVALUE')

        null_mask = df['choice'].isnull()
        if null_mask.sum() > 0:
            logger.warning(f"Found {null_mask.sum()} pandas null values in choice column, replacing with NOTAVALUE")
            df.loc[null_mask, 'choice'] = 'NOTAVALUE'

        df.to_csv(csv_path, index=False, na_rep='NOTAVALUE', quoting=1)
        logger.info(f"Successfully saved CSV to {csv_path}")

        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        zipfile_name = filename.replace('.csv', '.zip')
        zip_path = os.path.join(self.output_dir, zipfile_name)

        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, filename)
            zipf.write(metadata_path, metadata_filename)

        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        logger.info(f"CSV submission saved to: {csv_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(f"Overall accuracy (excluding open-ended questions): {overall_accuracy:.2%} ({total_correct}/{total_examples})")

        return zip_path

    def save_submission_with_metadata(self, results: List[EvaluationResult],
                                      metadata: Dict = None, filename: str = "submission.csv",
                                      config_path: str = None, args: argparse.Namespace = None):
        dataset_examples = getattr(self, '_last_dataset_examples', [])
        return self.save_submission(results, filename, metadata, dataset_examples, config_path, args)

    def list_datasets(self):
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def load_metadata_from_config(self, config_path: str) -> Dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        _, ext = os.path.splitext(config_path)

        with open(config_path, 'r') as f:
            if ext.lower() in ['.json']:
                config = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            else:
                raise ValueError(f"Unsupported config file format: {ext}")

        metadata = config.get('metadata', config.get('meta_data', {}))

        required_fields = ['model_name', 'track', 'base_model_type', 'base_model_name', 'dataset']
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Required metadata field '{field}' not found in config")

        return metadata

    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        metadata = {}
        arg_mapping = {
            'model_name': 'model_name',
            'model_type': 'model_type',
            'track': 'track',
            'base_model_type': 'base_model_type',
            'base_model_name': 'base_model_name',
            'dataset': 'dataset',
            'additional_info': 'additional_info'
        }

        for arg_name, meta_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                metadata[meta_field] = getattr(args, arg_name)

        return metadata

    def get_metadata(self, config_path: str = None, args: argparse.Namespace = None,
                     fallback_metadata: Dict = None) -> Dict:
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework"
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
            if arg_metadata:
                logger.info("Applied metadata from command line arguments")

        return metadata


def create_metadata_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for metadata
    
    Returns:
        ArgumentParser with metadata-related arguments
    """
    parser = argparse.ArgumentParser(description='Evaluation Framework with Metadata Support')
    
    # Model information
    parser.add_argument('--model-name', type=str, help='Name of the model')
    parser.add_argument('--model-type', type=str, help='Type of model wrapper')
    parser.add_argument('--base-model-name', type=str, help='Name of the base model')
    parser.add_argument('--base-model-type', type=str, choices=['API', 'OpenWeighted'], 
                       help='Type of base model (API or OpenWeighted)')
    
    # Track information
    parser.add_argument('--track', type=str, choices=['internal_reasoning', 'agentic_reasoning'],
                       default='internal_reasoning', help='Competition track')
    
    # Dataset and submission info
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--additional-info', type=str, help='Additional information about the submission')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='competition_results', 
                       help='Output directory for results')
    parser.add_argument('--output-file', type=str, default='submission.csv', 
                       help='Output CSV filename for submission (will be packaged in zip)')
    
    # Evaluation settings
    parser.add_argument('--subset-size', type=int, help='Limit evaluation to N examples')
    
    return parser

def load_config_file(config_path):
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_and_merge_config(args):
    if not args.config:
        return args

    config = load_config_file(args.config)

    if 'metadata' in config:
        metadata = config['metadata']
        for key, value in metadata.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    def add_config_to_args(config_dict, prefix=''):
        for key, value in config_dict.items():
            if key in ['metadata', 'dataset']:
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)

    add_config_to_args(config)
    return args
