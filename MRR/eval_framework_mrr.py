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
import torch
from dotenv import load_dotenv
import time, random
load_dotenv()







def polite_sleep(base_delay=5.0, jitter=2.0, peak_multiplier=1.0):
    """
    Sleep base_delay seconds + random jitter in [0,jitter].
    peak_multiplier lets you slow down further if site looks stressed.
    """
    wait = max(0.01, base_delay * peak_multiplier) + random.uniform(0, jitter)
    return wait






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
    reasoning_traces: List[List[Dict]] = None  # Add reasoning traces
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
    """ChatGPT/OpenAI model wrapper (OpenAI Platform key)"""

    def load(self, **kwargs):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        from openai import OpenAI
        self.model_client = OpenAI(api_key=api_key)

        debug = kwargs.get("debug") or {}
        self.print_messages = bool(debug.get("print_messages", False))
        self.print_responses = bool(debug.get("print_responses", False))

    def inference(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.8,
        top_k: Optional[int] = None,  # not supported here; kept for compatibility
    ) -> Tuple[str, List[Dict]]:
        
        time.sleep(polite_sleep(base_delay=5.0, jitter=2.0))

        messages = [
            {"role": "user", "content": prompt},
        ]

        if self.print_messages:
            print("messages:", messages)

        resp = self.model_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        response = resp.choices[0].message.content or ""

        if self.print_responses:
            print("response_text:", response)

        complete_messages = messages + [{"role": "assistant", "content": response}]
        return response, complete_messages

# -----------------------------------------------------------------------------------------
# Qwen

# class LocalModel(BaseModel):
#     """Local HuggingFace model wrapper"""

#     def load(self, **kwargs):
#         """Load local HuggingFace model"""
#         try:
#             from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#             import torch

#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 torch_dtype=torch.bfloat16,
#                 device_map="auto",
#                 quantization_config=BitsAndBytesConfig(load_in_8bit=True),  # ✅ FIX: add missing comma + keyword style
#                 **kwargs
#             )
#             logger.info(f"Loaded local model: {self.model_name}")
#         except ImportError as e:
#             logger.error(f"Failed to import local model dependencies: {e}")
#             raise




    # -----------------------------------------------------------------------------------------
    # Llama

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

            # Pull quantization flags from kwargs (so config can control it)
            use_4bit = bool(kwargs.pop("use_4bit", False))
            use_8bit = bool(kwargs.pop("use_8bit", False))

            # Safety: don't allow both at once
            if use_4bit and use_8bit:
                raise ValueError("Config error: use_4bit and use_8bit cannot both be True.")

            # Tokenizer (local path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            model_kwargs = dict(
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model_kwargs.update(kwargs)

            # Quantization config (only if CUDA + bitsandbytes available)
            quantized = False
            if (use_4bit or use_8bit) and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    import bitsandbytes  # noqa: F401

                    if use_4bit:
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
                        quantized = True

                    elif use_8bit:
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True
                        )
                        quantized = True

                except Exception as e:
                    # If bnb isn't available, fall back to non-quantized load
                    logger.warning(f"bitsandbytes quantization requested but unavailable: {e}. Loading without quantization.")
                    quantized = False

            elif (use_4bit or use_8bit) and not torch.cuda.is_available():
                logger.warning("Quantization requested (4bit/8bit) but CUDA is not available. Loading without quantization.")

            # Model (local path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=False,
                **model_kwargs
            )

            logger.info(
                f"Loaded local model: {self.model_name} "
                f"(4bit={use_4bit and quantized}, 8bit={use_8bit and quantized})"
            )

        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise



    
    def inference(self, prompt: str, max_tokens: int = 1024, **kwargs) -> Tuple[str, List[Dict]]:
        """Local model inference"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        print("messages:", messages)

        try:
            enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except TypeError:
            # manche Tokenizer haben andere Signaturen
            enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        input_ids = enc.to(self.model.device)
        attention_mask = torch.ones_like(input_ids, device=self.model.device)

        temperature = float(kwargs.get("temperature", 0.4))
        top_p = float(kwargs.get("top_p", 0.9))

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,   # ✅ FIX
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=(temperature > 0.0),
        )
        
        
        response = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        print("response_text:", response_text)

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
        time.sleep(random.uniform(0.05, 0.25))
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
        temperature: float = 0.6,
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

        debug_cfg = (self.config.get("debug") or {})
        kwargs = dict(kwargs)
        kwargs["debug"] = debug_cfg
        
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

        dataset = build_dataset(dataset_config.get("dataset_path"))
        dataset_list: List[Dict] = []

        for i in range(len(dataset)):
            item = dataset[i]
            # dataset_utils liefert: (question_type, id_value, question, answer, meta_question)
            question_type, id_value, question, answer, meta_question = item

            ex = {
                "question_type": question_type,
                "id": id_value,
                "question": question,
                "answer": answer,
            }
            if question_type == "open_ended_multi_choice":
                ex["meta_question"] = meta_question

            dataset_list.append(ex)

        return dataset_list


















    def _extract_option_letters(self, question_text: str) -> List[str]:
        """
        Extrahiert vorhandene Antwortoptionen aus dem Frage-Text.
        Unterstützt Formate wie:
        A: ..., B: ... / A) ... / A. ...
        Rückgabe: z.B. ["A","B","C","D"].
        """
        if not question_text:
            return ["A", "B", "C", "D", "E"]

        text = question_text.strip()

        # Option-Markierungen am Zeilenanfang: "A:", "B)", "C." etc.
        letters = re.findall(r"(?m)^\s*([A-E])\s*[:\)\.]\s+", text.upper())
        # Deduplizieren in Reihenfolge
        out = []
        seen = set()
        for x in letters:
            if x not in seen:
                seen.add(x)
                out.append(x)

        # Fallback: wenn nichts gefunden, Default
        return out if out else ["A", "B", "C", "D", "E"]





    def _extract_ranking(self, response: str, candidates: List[str]) -> List[str]:
        """
        Parse eines Rankings für die gegebenen candidates.
        Akzeptiert z.B.:
        A > C > B
        A,C,B
        A C B
        > B > A > C  (wird repariert, solange Buchstaben vorhanden)
        Wenn unbrauchbar: gibt candidates in Originalreihenfolge zurück.
        """
        if not response:
            return candidates[:]  # fallback

        text = response.strip().upper()

        text = re.sub(r"\b(NONE|NOTAVALUE|NULL|NAN)\b", " ", text)

        # Nur Kandidatenbuchstaben extrahieren (als einzelne Tokens)
        pattern = r"(?<![A-Z])(" + "|".join(map(re.escape, candidates)) + r")(?![A-Z])"
        found = re.findall(pattern, text)

        ranking = []
        seen = set()
        for x in found:
            if x not in seen:
                seen.add(x)
                ranking.append(x)

        # Wenn GAR nichts gefunden: fallback
        if len(ranking) == 0:
            return candidates[:]

        # Fehlende Kandidaten hinten anfügen
        for c in candidates:
            if c not in seen:
                ranking.append(c)

        return ranking[:len(candidates)]



    def _mrrv_vote(self, ranked_lists: List[List[str]], candidates: List[str]) -> Tuple[str, Dict[str, float]]:
        """
        Mean Reciprocal Rank Voting gemäß:
        MRR(A) = (1/k) * sum_{i=1..k} 1 / rank_A(A_i^r)
        winner = argmax_A MRR(A)

        ranked_lists: Liste von k Rankings, jedes ist z.B. ["A","C","B","D","E"].
        """
        k = len(ranked_lists)
        if k == 0:
            return "NOTAVALUE", {}

        scores = {c: 0.0 for c in candidates}
        for ranking in ranked_lists:
            # Defensive: normalisiere Ranking
            local = []
            seen = set()
            for x in (ranking or []):
                x = str(x).strip().upper()
                if x in scores and x not in seen:
                    seen.add(x)
                    local.append(x)

            # Fehlende Kandidaten hinten anhängen
            for c in candidates:
                if c not in seen:
                    local.append(c)
                    seen.add(c)

            # Positionsmap in einem Pass (O(n))
            pos = {c: idx + 1 for idx, c in enumerate(local[:len(candidates)])}

            for c in candidates:
                scores[c] += 1.0 / float(pos[c])

        for c in candidates:
            scores[c] /= float(k)

        # Tie-break: höchste Score, dann alphabetisch
        winner = sorted(candidates, key=lambda c: (-scores[c], c))[0]
        return winner, scores








    def _aggregate_rankings_and_pick(
        self,
        ranked_lists: List[List[str]],
        candidates: List[str],
        strategy: str = "mrrv",   # "mrrv" oder "majority_top1"
    ) -> Tuple[str, Dict[str, float]]:
        if not ranked_lists or not candidates:
            return "NOTAVALUE", {}

        strategy = (strategy or "mrrv").lower()

        if strategy == "majority_top1":
            counts = {c: 0.0 for c in candidates}
            for r in ranked_lists:
                if r and r[0] in counts:
                    counts[r[0]] += 1.0
            winner = sorted(candidates, key=lambda c: (-counts[c], c))[0]
            return winner, counts

        # default: mrrv
        return self._mrrv_vote(ranked_lists, candidates)


    def _log_vote_summary(
        self,
        prefix: str,
        winner: str,
        scores: Dict[str, float],
        ranked_lists: List[List[str]],
    ):
        logger.info(
            f"{prefix} RANKING-VOTE SUMMARY | winner={winner} | scores={scores} | ranked_lists={ranked_lists}"
        )


    def _log_top1_vote_summary(
        self,
        prefix: str,
        winner: str,
        counts: Dict[str, float],
        votes: List[str],
    ):
        logger.info(
            f"{prefix} TOP1-VOTE SUMMARY | winner={winner} | counts={counts} | votes={votes}"
        )






    def _majority_vote(self, votes: List[str], candidates: List[str]) -> Tuple[str, Dict[str, float]]:
        counts = {c: 0.0 for c in candidates}
        for v in votes:
            if v in counts:
                counts[v] += 1.0
        winner = sorted(candidates, key=lambda c: (-counts[c], c))[0]
        return winner, counts




    def _final_choice_override(self, example: Dict, choice: str, prediction_text: str) -> Tuple[str, str]:
        qtype = example.get("question_type", "")
        if qtype != "open_ended_multi_choice":
            return choice, ""

        meta_q = (example.get("meta_question") or "")
        if not meta_q:
            return choice, ""

        text = (prediction_text or "").lower()

        # very simple safety heuristic
        emergency_kw = ["emergency", "nearest emergency room", "epinephrine", "call 911", "seek emergency"]
        if any(k in text for k in emergency_kw):
            # try to pick the option whose text mentions emergency care
            # Parse meta options like "B: ...", "C: ..." etc.
            candidates = self._extract_option_letters(meta_q)
            opt_map = {}
            for line in meta_q.splitlines():
                m = re.match(r"^\s*([A-E])\s*:\s*(.+)\s*$", line.strip())
                if m:
                    opt_map[m.group(1).upper()] = m.group(2).strip().lower()

            for c in candidates:
                if "emergency" in opt_map.get(c, ""):
                    if c != choice:
                        return c, "Heuristic override: severe allergic reaction keywords -> choose emergency-care option."
                    break

        return choice, ""


    def _log_final_override(self, prefix: str, old: str, new: str, reason: str):
        logger.info(f"{prefix} FINAL-OVERRIDE | old={old} -> new={new} | reason={reason}")








    # =====================================================================
    # FIXED: CoT-safe prompts + syntactically safe meta_prompt
    # - Replaces _get_prediction_with_trace with CoT-safe prompts
    # - Uses a syntactically safe meta_prompt (no triple-quote nesting)
    # - Keeps _extract_multiple_choice_answer intact
    # =====================================================================
    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, List[Dict]]:
        """Get model prediction and reasoning trace for a single example"""
        question = example["question"]
        question_type = example["question_type"]

        prediction = {"choice": "", "open_ended_answer": ""}

        # Voting config IMMER am Anfang (sonst: UnboundLocalError in open_ended_multi_choice)
        voting_cfg = (self.config.get("voting") or {})
        method = (voting_cfg.get("method") or "").lower()          # e.g. "mrrv"
        k_votes = int(voting_cfg.get("k", 1))
        vote_temperature = float(voting_cfg.get("temperature", 0.6))
        vote_top_p = float(voting_cfg.get("top_p", 0.9))
        override_cfg = (self.config.get("override") or {})
        enable_override = bool(override_cfg.get("enable", False))
        rank_temperature = float(voting_cfg.get("rank_temperature", vote_temperature))
        rank_top_p = float(voting_cfg.get("rank_top_p", vote_top_p))
        meta_rank_temperature = float(voting_cfg.get("meta_rank_temperature", vote_temperature))
        meta_rank_top_p = float(voting_cfg.get("meta_rank_top_p", vote_top_p))
        debug_cfg = self.config.get("debug", {}) or {}
        print_prompts = bool(debug_cfg.get("print_prompts", False))
        add_rationale = bool(debug_cfg.get("add_rationale", False))




        # ---------------------------------------------------------
        # MULTI_CHOICE
        # ---------------------------------------------------------
        if question_type == "multi_choice":
            candidates = self._extract_option_letters(question)
            allowed = ", ".join(candidates)

            # MRRV path
            if method == "mrrv" and k_votes > 1:
                cand_str = " > ".join(candidates)
                ranked_lists: List[List[str]] = []
                all_traces: List[Dict] = []

                rank_prompt = (
                    "TASK: Rank the answer options for the following multiple-choice medical question.\n"
                    "You are a medical expert.\n\n"
                    "OUTPUT RULES (STRICT):\n"
                    "1) Output EXACTLY one line.\n"
                    "2) The line must contain ONLY the option letters shown below, each EXACTLY ONCE.\n"
                    "3) Each letter must appear EXACTLY ONCE (no repeats, no omissions)."
                    "4) Return the ranking of options by likelihood from best to worst.\n"
                    f"5) Use EXACTLY this separator: ' > ' (space, greater-than, space).\n"
                    f"6) Output format MUST be exactly: {cand_str}\n"
                    f"7) Valid letters: {', '.join(candidates)}\n"
                    "No explanation.\n\n"
                    "QUESTION:\n"
                    f"{question}\n\n"
                    "FINAL RANKING:"
                )

                for _ in range(k_votes):
                    rank_resp, rank_trace = self.model.inference(
                        rank_prompt,
                        temperature=rank_temperature,
                        top_p=rank_top_p,
                    )
                    all_traces.extend(rank_trace)
                    ranked_lists.append(self._extract_ranking(rank_resp, candidates))

                winner, scores = self._aggregate_rankings_and_pick(
                    ranked_lists=ranked_lists,
                    candidates=candidates,
                    strategy="mrrv",
                )

                self._log_vote_summary(
                    prefix=f"[multi_choice id={example.get('id','?')}]",
                    winner=winner,
                    scores=scores,
                    ranked_lists=ranked_lists,
                )

                prediction["choice"] = winner
                prediction["open_ended_answer"] = (
                    f"MRRV winner={winner}; scores={scores}; rankings={ranked_lists}"
                )
                return prediction, all_traces

            # Single-letter default
            prompt = (
                "The following is a multi_choice question.\n"
                "You are a medical expert.\n\n"
                "STRICT OUTPUT:\n"
                f"Return EXACTLY ONE LETTER: {allowed}.\n"
                "No punctuation, no words, no explanation.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "FINAL ANSWER (one letter only):"
            )
            response, reasoning_trace = self.model.inference(prompt)
            choice = self._extract_multiple_choice_answer(response)
            prediction["choice"] = choice if choice in candidates else "NOTAVALUE"
            prediction["open_ended_answer"] = response.strip()
            return prediction, reasoning_trace


    # ---------------------------------------------------------
    # OPEN_ENDED
    # ---------------------------------------------------------

        if question_type == "open_ended":
            prompt = (
                "The following is a open_ended question.\n"
                "You are a medical expert.\n\n"
                "OUTPUT FORMAT:\n"
                "1) Recommendation (1–3 sentences)\n"
                "2) Key rationale (3–5 bullet points)\n"
                "3) Safety/contraindications/interactions (bullet points, if relevant)\n"
                "If unsure, state assumptions and missing information.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "ANSWER:"
            )
            response, reasoning_trace = self.model.inference(prompt)
            prediction["choice"] = "NOTAVALUE"
            prediction["open_ended_answer"] = response.strip()
            return prediction, reasoning_trace



    # ---------------------------------------------------------
    # OPEN_ENDED_MULTI_CHOICE
    # ---------------------------------------------------------

        if question_type == "open_ended_multi_choice":
            prompt = (
                "The following is a open_ended_multi_choice question.\n"
                "You are a medical expert.\n\n"
                "OUTPUT FORMAT:\n"
                "- Answer in 1-3 bullet points.\n"
                "Each bullet MUST include concrete identifiers from the options (drug names, regimen, placebo vs active).\n"
                "- Do NOT mention option letters (A–E) in this step.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "ANSWER (bullets):"
            )
            response, reasoning_trace = self.model.inference(prompt)
            prediction["open_ended_answer"] = response.strip()

            meta_q = example.get("meta_question")
            if meta_q:
                meta_candidates = self._extract_option_letters(meta_q)
                allowed = ", ".join(meta_candidates)

                # Meta MRRV
                if method == "mrrv" and k_votes > 1:
                    meta_cand_str = " > ".join(meta_candidates)
                    ranked_lists: List[List[str]] = []

                    for _ in range(k_votes):
                        meta_rank_prompt = (
                            f"{meta_q}\n"
                            "You are a helpful assistant who reviews an open-end answer.\n\n"
                            "Task: Rank the options from best match to worst match.\n"
                            "STRICT OUTPUT:\n"
                            "Return EXACTLY one line.\n"
                            "The line must contain ONLY the option letters shown in the meta-question, each EXACTLY ONCE.\n"
                            "Use EXACTLY this separator: ' > '.\n"
                            f"Output format MUST be exactly: {meta_cand_str}\n"
                            "No explanation. No other text.\n\n"
                            "Agent's answer:\n"
                            f"{response.strip()}\n\n"
                            "FINAL RANKING:"
                        )

                        meta_rank_resp, meta_rank_trace = self.model.inference(
                            meta_rank_prompt,
                            temperature=meta_rank_temperature,
                            top_p=meta_rank_top_p,
                        )
                        reasoning_trace.extend(meta_rank_trace)
                        ranked_lists.append(self._extract_ranking(meta_rank_resp, meta_candidates))

                    winner, scores = self._aggregate_rankings_and_pick(
                        ranked_lists=ranked_lists,
                        candidates=meta_candidates,
                        strategy="mrrv",
                    )

                    self._log_vote_summary(
                        prefix=f"[meta_vote id={example.get('id','?')}]",
                        winner=winner,
                        scores=scores,
                        ranked_lists=ranked_lists,
                    )

                    prediction["choice"] = winner
                    prediction["open_ended_answer"] = (
                        prediction["open_ended_answer"].strip()
                        + f"\n\nMRRV(meta) winner={winner}; scores={scores}; rankings={ranked_lists}"
                    ).strip()


                # Meta majority-letter
                else:
                    meta_mode = (voting_cfg.get("meta_mode") or "majority_letter").lower()

                    if meta_mode == "majority_letter" and k_votes > 1:
                        votes: List[str] = []
                
                        meta_prompt = (
                            f"{example['meta_question']}\n"
                            "You are a helpful assistant who reviews an open-end answer.\n\n"
                            "Analyze it and choose the single option (A, B, C, D, or E) whose text best matches the agent answer.\n\n"
                            "If uncertain, pick the closest match.\n"
                            "STRICT OUTPUT:\n"
                            f"Return exactly one letter from: {allowed}.\n"
                            "No explanation.\n\n"
                            "Agent's answer:\n"
                            f"{response.strip()}\n\n"
                            "FINAL ANSWER (one letter only):\n"
                        )
                    
                        for _ in range(k_votes):
                            meta_resp, meta_rank_trace = self.model.inference(
                                meta_prompt,
                                temperature=meta_rank_temperature,
                                top_p=meta_rank_top_p,
                            )
                            reasoning_trace.extend(meta_trace)
                            v = self._extract_multiple_choice_answer(meta_resp)
                            votes.append(v if v in meta_candidates else "")

                        winner, counts = self._majority_vote(votes, meta_candidates)

                        self._log_top1_vote_summary(
                            prefix=f"[meta_choice id={example.get('id','?')}]",
                            winner=winner,
                            counts=counts,
                            votes=votes,
                        )

                        prediction["choice"] = winner
                        prediction["open_ended_answer"] = (
                            prediction["open_ended_answer"].strip()
                            + f"\n\nMETA-CHOICE winner={winner}; counts={counts}; votes={votes}"
                        ).strip()
                    else:
                        # Single meta decision
                        meta_prompt = (
                            f"{meta_q}\n"
                            "You are a helpful assistant who reviews an open-end answer.\n\n"
                            "Choose the single option letter whose text best matches the agent answer.\n"
                            "STRICT OUTPUT:\n"
                            f"Return exactly one letter from: {allowed}.\n"
                            "No explanation.\n\n"
                            "Agent's answer:\n"
                            f"{response.strip()}\n\n"
                            "FINAL ANSWER (one letter only):"
                        )
                        meta_resp, meta_trace = self.model.inference(meta_prompt)
                        reasoning_trace.extend(meta_trace)
                        v = self._extract_multiple_choice_answer(meta_resp)
                        prediction["choice"] = v if v in meta_candidates else "NOTAVALUE"

                # Final heuristic override (wird NICHT wieder überschrieben)
                if enable_override:
                    old = prediction.get("choice") or "NOTAVALUE"
                    new, reason = self._final_choice_override(example, old, prediction.get("open_ended_answer", ""))
                    if reason and new and new != old:
                        self._log_final_override(
                            prefix=f"[id={example.get('id','?')}]",
                            old=old,
                            new=new,
                            reason=reason,
                        )
                        prediction["choice"] = new

                return prediction, reasoning_trace

            # Fallback wenn meta_question fehlt: versuche Letter aus response zu extrahieren
            choice = self._extract_multiple_choice_answer(response)
            prediction["choice"] = choice if choice else "NOTAVALUE"
            return prediction, reasoning_trace

        raise ValueError(f"Unsupported question type: {question_type}")





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
