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
import re
from typing import List, Tuple, Dict

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
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference on the model

        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class ChatGPTModel(BaseModel):
    """ChatGPT/OpenAI model wrapper"""

    def load(self, **kwargs):
        """Load ChatGPT model"""

        api_key = os.getenv("AZURE_OPENAI_API_KEY_O1")
        api_version = "2024-12-01-preview"  # "2025-03-01-preview"

        if not api_key:
            raise ValueError("API key not found in environment. Please set the appropriate environment variable.")

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        from openai import AzureOpenAI
        print("Initializing AzureOpenAI client with endpoint:", azure_endpoint)
        print("Using API version:", api_version)
        self.model_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """ChatGPT inference"""
        messages = [{"role": "user", "content": prompt}]

        responses = self.model_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=8192,
        )
        response = responses.choices[0].message.content

        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response}]

        return response, complete_messages

# -----------------------------------------------------------------------------------------
# Qwen

class LocalModel(BaseModel):
    """Local HuggingFace model wrapper"""

    def load(self, **kwargs):
        """Load local HuggingFace model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),  # ✅ FIX: add missing comma + keyword style
                **kwargs
            )
            logger.info(f"Loaded local model: {self.model_name}")
        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise




    # -----------------------------------------------------------------------------------------
    # Llama

    # class LocalModel(BaseModel):
    # """Local HuggingFace model wrapper"""

    # def load(self, **kwargs):
    #     try:
    #         from transformers import AutoTokenizer, AutoModelForCausalLM
    #         import torch

    #         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    #         use_4bit = bool(kwargs.pop("use_4bit", False))
    #         use_8bit = bool(kwargs.pop("use_8bit", False))

    #         model_kwargs = {
    #             "device_map": "auto",
    #             "torch_dtype": torch.bfloat16,
    #             **kwargs,
    #         }

    #         # Quantization (requires bitsandbytes)
    #         if use_4bit or use_8bit:
    #             from transformers import BitsAndBytesConfig

    #             if use_4bit:
    #                 model_kwargs["quantization_config"] = BitsAndBytesConfig(
    #                     load_in_4bit=True,
    #                     bnb_4bit_compute_dtype=torch.bfloat16,
    #                     bnb_4bit_use_double_quant=True,
    #                     bnb_4bit_quant_type="nf4",
    #                 )
    #             elif use_8bit:
    #                 model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    #         self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
    #         logger.info(f"Loaded local model: {self.model_name}")

    #     except ImportError as e:
    #         logger.error(f"Failed to import local model dependencies: {e}")
    #         raise










    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Local model inference"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        #print("messages:", messages)                 #--------------------------------------------------------------------------------------

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors='pt', enable_thinking=False
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )

        response = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        #print("response_text:", response_text)      #--------------------------------------------------------------------------------------

        # Create complete conversation history
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

        self.output_dir = self.config.get('output_dir', 'results')
        os.makedirs(self.output_dir, exist_ok=True)

        self.datasets = self._load_dataset_configs(self.config)

        self.prompting_strategy = self.config.get("prompting_strategy", "least_to_most")


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




    
  






    
    # ---------------------------------------------------------------------------------------
    # Least-to-Most prompt contexts
    # ---------------------------------------------------------------------------------------
    
    # NOTE: The paper’s decomposition stage uses the pattern:
    # "To answer the question 'X', we need to know: 'a', 'b', 'c'."
    # and the solving stage is sequential: exemplars + history + next Q.
    # :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

    _L2M_DECOMPOSITION_INSTRUCTIONS = (
        "Decompose the question into 2 to 6 simpler subquestions.\n"
        "Return ONLY the subquestions as a list, one per line.\n"
        "Use either:\n"
        "- <subquestion>\n"
        "or\n"
        "\"<subquestion>\"\n"
        "Do NOT answer them.\n"
    )

    _L2M_SOLVING_INSTRUCTIONS = (
        "You will answer the next question using the previous Q/A context.\n"
        "Be concise and medically accurate.\n"
    )

    def _least_to_most_decompose(self, question: str) -> Tuple[List[str], str]:
        prompt = (
            f"{self._L2M_DECOMPOSITION_INSTRUCTIONS}\n"
            f"QUESTION:\n{question}\n"
            f"SUBQUESTIONS:\n"
        )
        resp, trace = self.model.inference(prompt)
    
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
    
        subqs: List[str] = []
    
        # bullet style "- ..."
        for l in lines:
            if l.startswith("-"):
                sq = l.lstrip("-").strip()
                if sq:
                    subqs.append(sq)
    
        # quoted style
        if not subqs:
            subqs = re.findall(r"\"([^\"]+)\"", resp)
    
        # fallback split
        if not subqs:
            cleaned = re.sub(r"(?is).*we need to know:\s*", "", resp).strip()
            parts = [p.strip(" \n\t-•\".()") for p in cleaned.split(",") if p.strip()]
            subqs = [p for p in parts if len(p) > 5]
    
        # HARD FILTER: rauswerfen, was wie eine direkte Antwort aussieht
        def looks_like_answer(s: str) -> bool:
            s_up = s.upper()
            return (
                "THE CORRECT ANSWER" in s_up
                or "THE ANSWER IS" in s_up
                or re.search(r"\b[A-D]\s*:", s_up) is not None  # option patterns
            )
    
        subqs = [q for q in subqs if not looks_like_answer(q)]
    
        # cap
        subqs = subqs[:6]
    
        trace_text = f"[L2M:DECOMPOSE]\nPROMPT:\n{prompt}\n\nRESPONSE:\n{resp}\n"
        return subqs, trace_text


    
    def _least_to_most_solve(self, original_question: str, subquestions: List[str]) -> Tuple[str, str]:
        """
        Stage 2 (paper): sequential solving.
        Prompt structure: (constant solving instructions/examples) + (Q/A history) + (next Q)
        and iterate. The original question is appended as the final subproblem. :contentReference[oaicite:6]{index=6}
        Returns: (final_response_text, trace_text)
        """
        # Solve subquestions, then original question
        queue = list(subquestions) + [original_question]

        history = ""
        full_trace = "[L2M:SOLVE]\n"
        last_answer = ""
    
        for i, q in enumerate(queue, start=1):
            prompt = (
                f"{self._L2M_SOLVING_INSTRUCTIONS}\n"
                f"{history}"
                f"Q{i}: {q}\n"
                f"A{i}:"
            )
            resp, trace = self.model.inference(prompt)
            ans = resp.strip()
    
            history += f"Q{i}: {q}\nA{i}: {ans}\n\n"
            full_trace += f"\n--- STEP {i}/{len(queue)} ---\nPROMPT:\n{prompt}\n\nRESPONSE:\n{resp}\n"
            last_answer = ans
    
        return last_answer, full_trace





    def _force_mc_letter(self, question_with_options: str, context_answer: str) -> Tuple[str, str]:
        prompt = (
            "You are a medical expert.\n"
            "Use the context if helpful, but answer the multiple-choice question.\n"
            "STRICT OUTPUT: return exactly one letter A, B, C, D, or E.\n\n"
            "CONTEXT:\n"
            f"{context_answer}\n\n"
            "QUESTION:\n"
            f"{question_with_options}\n\n"
            "FINAL ANSWER (one letter only):"
        )
        resp, trace = self.model.inference(prompt)
        choice = self._extract_multiple_choice_answer(resp)
        return choice, f"[L2M:FINAL_MC]\nPROMPT:\n{prompt}\n\nRESPONSE:\n{resp}\n"



    



    def _l2m_map_to_choice(self, question_block: str, agent_answer: str) -> Tuple[str, str]:
        prompt = (
            "You are a medical expert evaluator.\n"
            "Choose the single best option label (A, B, C, D, or E) that matches the agent answer.\n"
            "STRICT OUTPUT: return exactly one letter A, B, C, D, or E.\n\n"
            "QUESTION (with options):\n"
            f"{question_block}\n\n"
            "AGENT ANSWER:\n"
            f"{agent_answer}\n\n"
            "FINAL ANSWER (one letter only):"
        )
        resp, trace = self.model.inference(prompt)
        choice = self._extract_multiple_choice_answer(resp)
        return choice, f"[L2M:MAP]\nPROMPT:\n{prompt}\n\nRESPONSE:\n{resp}\n"

# ---------------------------------------------------------------------------------------------



  



  
    # =====================================================================

  
    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
    
        question = example["question"]
        question_type = example["question_type"]
    
        # Always return these keys
        prediction = {"choice": "", "open_ended_answer": ""}
    
        # ============================================================
        # Least-to-Most prompting (two-stage) branch
        # ============================================================
        # if getattr(self, "prompting_strategy", "cot_safe") == "least_to_most":
        #     subqs, trace_decomp = self._least_to_most_decompose(question)
        #     final_resp, trace_solve = self._least_to_most_solve(question, subqs)
    
        #     reasoning_trace = trace_decomp + "\n" + trace_solve
    
        #     if question_type == "open_ended":
        #         prediction["choice"] = "NOTAVALUE"
        #         prediction["open_ended_answer"] = final_resp
        #         return prediction, reasoning_trace
    
        #     if question_type == "multi_choice":
        #         choice, trace_mc = self._force_mc_letter(question, final_resp)
        #         prediction["choice"] = choice if choice else ""
        #         prediction["open_ended_answer"] = final_resp
        #         return prediction, reasoning_trace + "\n" + trace_mc
    
        #     if question_type == "open_ended_multi_choice":
        #         prediction["open_ended_answer"] = final_resp
            if getattr(self, "prompting_strategy", "cot_safe") == "least_to_most":
            # L2M NUR für open_ended sinnvoll (bei kleinen lokalen Modellen)
            if question_type in ("multi_choice", "open_ended_multi_choice"):
                # fall back auf deinen bestehenden CoT-safe Pfad:
                # (einfach so tun, als wäre prompting_strategy != least_to_most)
                pass
            else:
                subqs, trace_decomp = self._least_to_most_decompose(question)
                final_resp, trace_solve = self._least_to_most_solve(question, subqs)
        
                reasoning_trace = trace_decomp + "\n" + trace_solve
                prediction["choice"] = "NOTAVALUE"
                prediction["open_ended_answer"] = final_resp
                return prediction, reasoning_trace
                # use existing meta_question if provided (preferred)
                if "meta_question" in example:
                    meta_prompt = (
                        f"{example['meta_question']}\n"
                        "You are a medical expert evaluator.\n"
                        "STRICT OUTPUT: Return exactly one letter A, B, C, D, or E.\n"
                        "No explanation.\n\n"
                        "Agent's answer:\n"
                        f"{final_resp}\n\n"
                        "FINAL ANSWER (one letter only):"
                    )
                    meta_resp, meta_trace = self.model.inference(meta_prompt)
                    choice = self._extract_multiple_choice_answer(meta_resp)
                    prediction["choice"] = choice if choice else ""
                    reasoning_trace += "\n[L2M:META]\n" + str(meta_trace)
                else:
                    # fallback: map using options inside the question text
                    choice, trace_map = self._l2m_map_to_choice(question, final_resp)
                    prediction["choice"] = choice if choice else ""
                    reasoning_trace += "\n" + trace_map
    
                return prediction, reasoning_trace
    
            raise ValueError(f"Unsupported question type: {question_type}")
    
        # ============================================================
        # Otherwise: your existing CoT-safe logic (unchanged)
        # ============================================================
        if question_type == "multi_choice":
            prompt = (
                "The following is a multi_choice question.\n"
                "You are a medical expert.\n\n"
                "REASONING (internal):\n"
                "Before answering, analyze the question step by step and break it down into sub-problems.\n"
                "STRICT OUTPUT:\n"
                "Return EXACTLY ONE LETTER: A, B, C, D, or E.\n"
                "No punctuation, no words, no explanation.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "FINAL ANSWER (one letter only):"
            )
    
        elif question_type == "open_ended":
            prompt = (
                "The following is a open_ended question.\n"
                "You are a medical expert.\n\n"
                "REASONING (internal):\n"
                "Before answering, analyze the question step by step and break it down into sub-problems.\n"
                "OUTPUT FORMAT:\n"
                "1) Recommendation (1–3 sentences)\n"
                "2) Key rationale (3–5 bullet points)\n"
                "3) Safety/contraindications/interactions (bullet points, if relevant)\n"
                "If unsure, state assumptions and missing information.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "ANSWER:"
            )
    
        elif question_type == "open_ended_multi_choice":
            prompt = (
                "The following is a open_ended_multi_choice question.\n"
                "You are a medical expert.\n\n"
                "REASONING (internal):\n"
                "Before answering, analyze the question step by step and break it down into sub-problems.\n"
                "OUTPUT FORMAT:\n"
                "- Answer in 1-3 bullet points.\n"
                "Each bullet MUST include concrete identifiers from the options (drug names, regimen, placebo vs active).\n"
                "- Do NOT mention option letters (A–E) in this step.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "ANSWER (bullets):"
            )
    
        else:
            raise ValueError(f"Unsupported question type: {question_type}")
    
        response, reasoning_trace = self.model.inference(prompt)
    
        if question_type == "multi_choice":
            choice = self._extract_multiple_choice_answer(response)
            prediction["choice"] = choice if choice else ""
            prediction["open_ended_answer"] = response.strip()
    
        elif question_type == "open_ended_multi_choice":
            prediction["open_ended_answer"] = response.strip()
    
            if "meta_question" in example:
                meta_prompt = (
                    f"{example['meta_question']}\n"
                    "You are a medical expert evaluator.\n\n"
                    "STRICT OUTPUT:\n"
                    "Return exactly one letter A, B, C, D, or E.\n"
                    "No explanation.\n\n"
                    "Agent's answer:\n"
                    f"{response.strip()}\n\n"
                    "FINAL ANSWER (one letter only):"
                )
                meta_response, meta_reasoning = self.model.inference(meta_prompt)
                reasoning_trace += meta_reasoning
                choice = self._extract_multiple_choice_answer(meta_response)
                prediction["choice"] = choice if choice else ""
            else:
                choice = self._extract_multiple_choice_answer(response)
                prediction["choice"] = choice if choice else ""
    
        elif question_type == "open_ended":
            prediction["choice"] = "NOTAVALUE"
            prediction["open_ended_answer"] = response.strip()
    
        return prediction, reasoning_trace

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
                reasoning_trace = json.dumps(result.reasoning_traces[i])

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
    parser = argparse.ArgumentParser(description='Evaluation Framework with Metadata Support')

    parser.add_argument('--model-name', type=str, help='Name of the model')
    parser.add_argument('--model-type', type=str, help='Type of model wrapper')
    parser.add_argument('--base-model-name', type=str, help='Name of the base model')
    parser.add_argument('--base-model-type', type=str, choices=['API', 'OpenWeighted'],
                        help='Type of base model (API or OpenWeighted)')

    parser.add_argument('--track', type=str, choices=['internal_reasoning', 'agentic_reasoning'],
                        default='internal_reasoning', help='Competition track')

    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--additional-info', type=str, help='Additional information about the submission')

    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')

    parser.add_argument('--output-dir', type=str, default='competition_results',
                        help='Output directory for results')
    parser.add_argument('--output-file', type=str, default='submission.csv',
                        help='Output CSV filename for submission (will be packaged in zip)')

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
