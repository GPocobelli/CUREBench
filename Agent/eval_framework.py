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
    framework.sa        elif question_type == "open_ended":
            # For open-ended, only return response, use NOTAVALUE for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE"  # Use NOTAVALUE instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()ubmission(results, "my_submission.json")
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
import time, random
import re  # needed for fallback and existing extractor patterns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def polite_sleep(base_delay=5.0, jitter=2.0, peak_multiplier=1.0):
    """
    Sleep base_delay seconds + random jitter in [0,jitter].
    peak_multiplier lets you slow down further if site looks stressed.
    """
    wait = max(0.01, base_delay * peak_multiplier) + random.uniform(0, jitter)
    return wait



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
        
        
        time.sleep(random.uniform(0.05, 0.25))
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
                quantization_config = BitsAndBytesConfig(load_in_8bit=True),
                **kwargs
            )
            logger.info(f"Loaded local model: {self.model_name}")
        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Local model inference"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        print("messages:", messages)
        
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
        print("response_text:", response_text)
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
            # For custom models, we'll create a simple message structure
            messages = [{"role": "user", "content": prompt}]
            
            response = self._inference_func(self.model, prompt, max_tokens)
            
            # Create complete conversation history
            complete_messages = messages + [{"role": "assistant", "content": response}]
            
            return response, complete_messages
        except Exception as e:
            logger.error(f"Custom model inference error: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Error occurred"}
            ]
            return "Error occurred", error_messages


class LangChainAgentModel(BaseModel):
    """LangChain tool-calling agent wrapper.

    This model is intended for the *agentic_reasoning* track. It runs an
    AgentExecutor (tool-calling agent) and returns the final text answer.

    Notes
    -----
    - The evaluation framework controls the prompt format for each question type.
      Therefore, the agent's system instructions must primarily enforce:
        (1) follow the user's prompt instructions, and
        (2) keep the final answer concise and compliant with requested formats.
    - Tooling is imported from tools.py (search, wikipedia, save).
    """

    def __init__(self, model_name: str, verbose: bool = False):
        super().__init__(model_name)
        self.verbose = verbose
        self._agent = None

    def load(self, **kwargs):
        """Build the LangChain agent."""
        try:
            from dotenv import load_dotenv
            from langchain_openai import ChatOpenAI
            from langchain.agents import create_agent
        except ImportError as e:
            raise ImportError(
                "Missing dependencies for LangChainAgentModel. "
                "Install: python -m pip install langchain langchain-openai "
                "langchain-community python-dotenv"
            ) from e

        # Load env vars (OpenAI/Azure OpenAI keys etc.)
        load_dotenv()

        # Allow overriding verbosity via kwargs
        self.verbose = bool(kwargs.get("verbose", self.verbose))

        # Tools are defined in tools.py (DuckDuckGo, Wikipedia, file save)
        try:
            from tools import ALL_TOOLS
            tools = ALL_TOOLS
        except Exception as e:
            raise RuntimeError(
                "Failed to import tools from tools.py. Ensure tools.py is present "
                "and its dependencies are installed."
            ) from e

        # LLM: use the provided model_name from the framework.
        # Users can pass model kwargs through eval_framework.load_model(..., **kwargs).
        llm_kwargs = {}
        # Common optional overrides
        if "temperature" in kwargs:
            llm_kwargs["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            llm_kwargs["max_tokens"] = kwargs["max_tokens"]

        llm = ChatOpenAI(model=self.model_name, **llm_kwargs)

        # In LangChain v1 ist create_agent der Standard.
        # system_prompt ersetzt hier Ihren ChatPromptTemplate-Block.
        self._agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "You are a medical question-answering agent. "
                "For EVERY question, you MUST use at least one tool before answering.\n"
                "Prefer pubmed_search (and pubmed_fetch) for biomedical claims. "
                "Follow the prompt's instructions exactly, especially output format rules. "
                "If the prompt requests a multiple-choice letter, output ONLY that letter."
            ),
        )

    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        
        time.sleep(polite_sleep(base_delay=5.0, jitter=2.0))
        
        if self._agent is None:
            raise RuntimeError("Agent not initialized. Did you call load()?")

        # LangChain v1: call agent with messages
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        # result contains messages-list
        messages = result.get("messages", [])
        
        print("\n==== RAW MESSAGE TYPES ====")
        for m in messages:
            print(type(m), getattr(m, "type", None), getattr(m, "content", None)[:200] if getattr(m, "content", None) else None)
            if hasattr(m, "tool_calls"):
                print("  tool_calls:", m.tool_calls)
            if hasattr(m, "additional_kwargs"):
                print("  additional_kwargs keys:", list(m.additional_kwargs.keys()))
        print("===========================\n")
        
        output_text = ""
        if messages:
            # extract last message
            last = messages[-1]
            # AIMessage has .content, dict has ["content"]
            output_text = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else "")
        output_text = str(output_text or "").strip()

        # build trace
        trace = [{"role": "user", "content": prompt}]

        last_tool_name = None

        for m in messages:
            m_type = getattr(m, "type", None)

            # Tool calls
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    last_tool_name = tc.get("name")
                    trace.append({
                        "role": "tool_call",
                        "tool": last_tool_name,
                        "tool_input": tc.get("args"),
                    })
                continue

            # Tool result
            if m_type == "tool":
                trace.append({
                    "role": "tool_result",
                    "tool": last_tool_name or "unknown_tool",
                    "content": str(getattr(m, "content", "")),
                })
                continue

            # Normal messages
            role = "assistant" if m_type in ("ai", "assistant") else (m_type or "message")
            trace.append({"role": role, "content": str(getattr(m, "content", ""))})

        trace.append({"role": "assistant", "content": output_text})
        return output_text, trace

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
            # this will automatically use MXFP4 weights.
            torch_dtype = "auto"
            quant_config = None

        model_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto", **kwargs}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def inference(self, prompt: str, max_tokens: int = 1024, temperature: float = 1.0, top_p: float = 1.0, 
                  builtin_tools: Optional[List[str]] = None, tools: Optional[List[dict]] = None,) -> Tuple[str, List[Dict]]:
        
        from openai_harmony import Role
        import logging
        from transformers import AutoTokenizer  

        # Build message list
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

        # Apply Hugging Face chat template with fallback
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
            # Reload base tokenizer for Harmony
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
            do_sample=(temperature>0),
            eos_token_id=None if not self.enc else self.enc.stop_tokens()[-1],
        )
        # Parse Harmony messages
        gen_tokens = outputs[0][input_ids.shape[-1]:].tolist()
 
        try:
            parsed = self.enc.parse_messages_from_completion_tokens(gen_tokens, role=Role.ASSISTANT)
            reasoning_trace = [msg.to_dict() for msg in parsed]

            # Prefer "final" channel
            finals = [msg for msg in parsed if msg.to_dict().get("channel") == "final"]
            if finals:
                final_response = "".join(c.text for c in finals[-1].content if hasattr(c, "text"))
            else:
                # Fallback: take last assistant message, but strip to short answer
                final_response = "".join(c.text for c in parsed[-1].content if hasattr(c, "text"))

        except Exception as e:
            logging.error(f"[Harmony parse error] {e}")
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            final_response = text
            reasoning_trace = [{"role": "assistant", "content": text}]

        return final_response.strip(), reasoning_trace

def extract_sources_from_trace(trace: List[Dict[str, Any]]) -> List[str]:
    """
    Extract source information from the LangChain agent trace produced by
    LangChainAgentModel.inference() in this file.

    Expected trace structure:
      - {"role": "tool_call", "tool": "...", "tool_input": ...}
      - {"role": "tool_result", "content": "..."}
    """
    sources = set()

    last_tool = None
    for step in trace:
        if not isinstance(step, dict):
            continue

        role = step.get("role")

        if role == "tool_call":
            last_tool = step.get("tool", "unknown_tool")

            # Optional: capture explicit URLs in tool_input (rare, but possible)
            tool_input = step.get("tool_input")
            if isinstance(tool_input, str):
                for url in re.findall(r"https?://\S+", tool_input):
                    sources.add(f"{last_tool}: {url}")

        elif role == "tool_result":
            tool_name = last_tool or "unknown_tool"
            content = step.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            # URLs
            for url in re.findall(r"https?://\S+", content):
                url = url.rstrip(").,;]\"'")
                # Filter: PubMed DTD ist keine "Quelle"
                if "dtd.nlm.nih.gov" in url and "pubmed" in url:
                    continue
                sources.add(f"{tool_name}: {url}")

            # PubMed IDs
            for pmid in re.findall(r"\b\d{6,9}\b", content):
                # optional: nur wenn Tool pubmed_* war, um False Positives zu vermeiden
                if tool_name in ("pubmed_search", "pubmed_fetch"):
                    sources.add(f"{tool_name}: PMID{pmid}")

            # ClinicalTrials IDs
            for nct in re.findall(r"\bNCT\d{8}\b", content, flags=re.IGNORECASE):
                sources.add(f"{tool_name}: {nct.upper()}")

    return sorted(sources)

class CompetitionKit:
    """
    Simple competition framework - everything you need in one class!
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the competition kit
        
        Args:
            output_dir: Directory to save results and submissions
            config_path: Path to configuration file containing dataset configs
        """
        self.model = None
        self.model_name = None
        
        self.config = json.load(open(config_path, 'r')) if config_path else {}
        
        self.output_dir = self.config.get('output_dir', 'results')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset configurations from config file or use defaults
        self.datasets = self._load_dataset_configs(self.config)
    
    def load_model(self, model_name: str, model_type: str = "auto", **kwargs):
        """
        Load a model for evaluation
        
        Args:
            model_name: Name/path of the model (e.g., "gpt-4o-mini", "meta-llama/Llama-2-7b-chat-hf")
            model_type: Type of model ("chatgpt", "local", "custom", "auto" for auto-detection)
            **kwargs: Additional model configuration
        """
        self.model_name = model_name
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            model_type = self._detect_model_type(model_name)
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        
        if model_type in ["chatgpt", "azure_chatgpt"]:
            self.model = ChatGPTModel(model_name)
        elif model_type in ["agent", "langchain_agent", "agentic"]:
            # Tool-calling agent (LangChain)
            self.model = LangChainAgentModel(model_name, verbose=bool(kwargs.get("verbose", False)))
        elif model_type == "gpt-oss-20b":
            self.model = GPTOSS20BModel(model_name)
        elif model_type == "local":
            self.model = LocalModel(model_name)
        elif model_type == "custom":
            # For custom models, user should provide model_instance and inference_func
            model_instance = kwargs.get("model_instance")
            inference_func = kwargs.get("inference_func")
            if not model_instance or not inference_func:
                raise ValueError("Custom model requires 'model_instance' and 'inference_func' parameters")
            self.model = CustomModel(model_name, model_instance, inference_func)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load the model
        self.model.load(**kwargs)
    
    def _load_dataset_configs(self, config) -> Dict:
        """
        Load dataset configurations from config file or return defaults
        
        Args:
            config: Configuration dictionary

        Returns:
            Dictionary of dataset configurations
        """
        if not config:
            print("Not config provided, existing.")
            exit(1)

        # Check if config has a single dataset configuration
        if 'dataset' in config:
            dataset_config = config['dataset']
            dataset_name = dataset_config.get('dataset_name', 'treatment')
            # Create a dictionary with the dataset name as key
            return {dataset_name: dataset_config}
        else:
            # If no dataset in config, return defaults
            print("Not config found, existing.")
            exit(1)

    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type based on model name"""
        if "gpt-oss-20b" in model_name.lower():
            return "gpt-oss-20b"
        if any(name in model_name.lower() for name in ["gpt", "chatgpt", "openai", 'o1', 'o3', 'o4']):
            return "chatgpt"
        else:
            return "local"
    
    def evaluate(self, dataset_name: str, subset_size: int = None) -> EvaluationResult:
        """
        Evaluate model on a dataset
        
        Args:
            dataset_name: Name of dataset to evaluate on
            
        Returns:
            EvaluationResult object with scores and predictions
        """
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config['description']}")
        
        # Load dataset
        dataset = self._load_dataset(dataset_config)
        
        # Store dataset examples for later use in save_submission
        self._last_dataset_examples = dataset
        
        if subset_size is not None and subset_size > 0:
            dataset = dataset[:subset_size]
            logger.info(f"Subset size applied: {len(dataset)} examples")
        
        # Run evaluation
        predictions = []
        reasoning_traces = []  # Store reasoning traces
        total_count = len(dataset)
        # Track accuracy only for non-open-ended questions
        accuracy_correct_count = 0
        accuracy_total_count = 0
        
        logger.info(f"Running evaluation on {total_count} examples...")
        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Get prediction and reasoning trace
                prediction, reasoning_trace = self._get_prediction_with_trace(example)
                predictions.append(prediction)
                reasoning_traces.append(reasoning_trace)
                
                # Check if correct based on question type
                is_correct = False
                question_type = example["question_type"]
                expected_answer = example.get("answer")
                print("expected_answer:", expected_answer)
                
                if question_type == "multi_choice" or question_type == "open_ended_multi_choice":
                    # For multiple choice, compare the choice field
                    if expected_answer !='':
                        is_correct = prediction["choice"] == expected_answer
                    else:
                        is_correct = False
                    # Count for accuracy calculation (exclude open_ended)
                    accuracy_total_count += 1
                    if is_correct:
                        accuracy_correct_count += 1
                elif question_type == "open_ended":
                    # For open-ended, compare the open_ended_answer field but don't count in accuracy, we have internal evaluation for open-ended questions
                    if expected_answer !='':
                        is_correct = prediction["open_ended_answer"] == expected_answer
                    else:
                        is_correct = False
                
                # Log progress
                if (i + 1) % 10 == 0:
                    current_acc = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
                    logger.info(f"Progress: {i+1}/{total_count}, Accuracy: {current_acc:.2%} (excluding open-ended)")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                error_prediction = {
                    "choice": "NOTAVALUE",  # Use NOTAVALUE instead of empty string
                    "open_ended_answer": "Error"
                }
                predictions.append(error_prediction)
                reasoning_traces.append("Error occurred during inference")
        
        # Calculate final accuracy (excluding open-ended questions)
        accuracy = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
        
        result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,  # Use accuracy-specific count
            total_examples=accuracy_total_count,  # Use accuracy-specific count
            predictions=predictions,
            reasoning_traces=reasoning_traces  # Include reasoning traces
        )
        
        logger.info(f"Evaluation completed: {accuracy:.2%} accuracy ({accuracy_correct_count}/{accuracy_total_count}) - excluding open-ended questions")
        logger.info(f"Total examples processed: {total_count} (including {total_count - accuracy_total_count} open-ended questions)")
        
        return result
    
    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Load dataset based on configuration"""
        from dataset_utils import build_dataset
        from torch.utils.data import DataLoader
        
        # Build dataset
        dataset = build_dataset(
            dataset_config.get("dataset_path"),
        )
        
        # Convert to list of dictionaries for easier processing
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

    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
        question = example["question"]
        question_type = example["question_type"]
        
        # Format prompt
        if question_type == "multi_choice":
            prompt = f"The following is a multiple choice question about medicine. Answer with only the letter (A, B, C, D, or E).\n\nQuestion: {question}\n\nAnswer:"
        elif question_type == "open_ended_multi_choice" or question_type == "open_ended":
            prompt = f"The following is an open-ended question about medicine. Provide a comprehensive answer.\n\nQuestion: {question}\n\nAnswer:"
        
        # Get model response and messages using the model's inference method
        response, reasoning_trace = self.model.inference(prompt)
        
        sources = extract_sources_from_trace(reasoning_trace)
        # if sources:
        #     print("\n---------------- SOURCES ----------------")
        #     for s in sources:
        #         print("-", s)
        
        # Initialize prediction dictionary
        prediction = {
            "choice": "",  # Use empty string instead of None
            "open_ended_answer": ""  # Use empty string instead of None
        }
        
        # Extract answer from response
        if question_type == "multi_choice":
            # For multiple choice, extract the letter
            choice = self._extract_multiple_choice_answer(response)
            # Ensure choice is never None or NULL
            prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            prediction["open_ended_answer"] = response.strip()  # Keep full response too
        elif question_type == "open_ended_multi_choice":
            # First get the detailed response
            prediction["open_ended_answer"] = response.strip()
            
            # Then use meta question to get choice, if available
            if "meta_question" in example:
                meta_prompt = f"{example['meta_question']}Agent's answer: {response.strip()}\n\nMulti-choice answer:"
                meta_response, meta_reasoning = self.model.inference(meta_prompt)
                # Combine reasoning traces
                reasoning_trace += meta_reasoning
                
                sources = extract_sources_from_trace(reasoning_trace)
                if sources:
                    # optional: remove old sources entry to avoid duplicates
                    reasoning_trace = [x for x in reasoning_trace if not (isinstance(x, dict) and x.get("role") == "sources")]
                    reasoning_trace.append({"role": "sources", "content": sources})
                                
                
                # Extract the letter choice
                choice = self._extract_multiple_choice_answer(meta_response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            else:
                # If no meta_question, try to extract choice directly from the response
                choice = self._extract_multiple_choice_answer(response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
        elif question_type == "open_ended":
            # For open-ended, only return response, use N/A for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE" # Use N/A instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()
        
        #print("\n================ MODEL OUTPUT ================")
        #print(response.strip())

        sources = extract_sources_from_trace(reasoning_trace)
        if sources:
            print("\n---------------- SOURCES ----------------")
            for s in sources:
                print("-", s)
        print("============================================\n")
        
        
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
        
        # Default to empty string if nothing found (to avoid None values in CSV)
        return ""
    
    def save_submission(self, results: List[EvaluationResult], filename: str = "submission.csv", 
                       metadata: Dict = None, dataset_examples: List[Dict] = None,
                       config_path: str = None, args: argparse.Namespace = None):
        """
        Save results in competition submission format as CSV file with metadata JSON and zip package
        
        Args:
            results: List of evaluation results
            filename: Output CSV filename (will be used for CSV inside zip)
            metadata: User-provided metadata dictionary containing model info, track, etc.
            dataset_examples: Original dataset examples to extract question IDs and reasoning traces
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        import pandas as pd
        import zipfile
        
        # Get metadata from various sources with priority order
        metadata = self.get_metadata(config_path, args, metadata)
        
        # Create submission data for CSV
        submission_data = []
        
        # Process each result to create the CSV format
        for result in results:
            # Get the corresponding dataset examples if provided
            examples = dataset_examples if dataset_examples else []
            
            for i, (prediction, example) in enumerate(zip(result.predictions, examples)):
                # Use stored reasoning trace if available, convert to simple text format
                reasoning_trace = json.dumps(result.reasoning_traces[i])
                # if result.reasoning_traces and i < len(result.reasoning_traces):
                #     trace = result.reasoning_traces[i]
                #     if isinstance(trace, list) and len(trace) > 0:
                #         # Convert list of messages to a simple text format
                #         text_parts = []
                #         for msg in trace:
                #             if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                #                 role = msg['role']
                #                 content = msg['content'].replace('\n', ' ').replace('\r', '').replace('"', "'")
                #                 text_parts.append(f"{role}: {content}")
                #         reasoning_trace = " | ".join(text_parts)
                #     else:
                #         # Fallback to string representation
                #         reasoning_trace = str(trace).replace('\n', ' ').replace('\r', '').replace('"', "'")
                
                # Clean up text fields to avoid CSV formatting issues
                prediction_text = prediction.get("open_ended_answer", "") or ""  # Ensure not None
                if not prediction_text or prediction_text.strip() == "":
                    prediction_text = "No prediction available"

                
                # Ensure choice is clean and never NULL
                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ['NULL', 'NONE', 'NAN']:
                    choice_clean = "NOTAVALUE"  # Use NOTAVALUE instead of empty string
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"  # Replace empty strings with NOTAVALUE to avoid NULL validation issues
                else:
                    choice_clean = str(choice_raw).strip()
                
                # Ensure reasoning trace is not null
                if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
                    reasoning_trace = "No reasoning available"
                
                # Create CSV row - let pandas handle the escaping
                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace)
                }
                
                # Debug: Log if choice is NULL-like
                if str(choice_clean).upper() in ['NULL', 'NONE', 'NAN'] or str(choice_clean).strip() == "":
                    logger.warning(f"Found NULL-like or empty choice for row {row['id']}: '{choice_clean}' - replacing with NOTAVALUE")
                    row["choice"] = "NOTAVALUE"
                
                submission_data.append(row)
        
        # Create DataFrame and save CSV with proper quoting and NaN handling
        df = pd.DataFrame(submission_data)
        
        # Convert all columns to string to avoid type issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Aggressive null value cleaning
        null_replacements = {
            'id': 'unknown_id',
            'prediction': 'No prediction available',
            'choice': 'NOTAVALUE',  # Use NOTAVALUE for choice instead of empty string
            'reasoning': 'No reasoning available'
        }
        
        # Replace all possible null-like values
        for col in df.columns:
            # Replace pandas null values
            df[col] = df[col].fillna(null_replacements.get(col, 'NOTAVALUE'))
            
            # Replace string representations of null
            null_like_values = ['nan', 'NaN', 'None', 'null', 'NULL', '<NA>', 'nat', 'NaT']
            for null_val in null_like_values:
                df[col] = df[col].replace(null_val, null_replacements.get(col, 'NOTAVALUE'))
            
            # Special handling for choice column - ensure it's never empty or null-like
            if col == 'choice':
                df[col] = df[col].replace('NOTAVALUE', 'NOTAVALUE')  # Keep NOTAVALUE as is for choice
                # Replace any null-like values with NOTAVALUE
                for null_val in null_like_values:
                    df[col] = df[col].replace(null_val, 'NOTAVALUE')
                # Replace empty strings with NOTAVALUE for choice column
                df[col] = df[col].replace('', 'NOTAVALUE')
                df[col] = df[col].replace(' ', 'NOTAVALUE')  # Also replace whitespace-only
            
            # Replace empty strings (except for choice column which can be empty)
            if col != 'choice' and col in null_replacements:
                df[col] = df[col].replace('', null_replacements[col])
                df[col] = df[col].replace(' ', null_replacements[col])  # Also replace whitespace-only
        
        csv_path = os.path.join(self.output_dir, filename)
        
        # Validate DataFrame before saving
        logger.info(f"Creating CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Final validation - check for any remaining nulls
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Still found {null_count} nulls in column {col}")
        
        # Check for any problematic data
        for idx, row in df.head().iterrows():
            logger.debug(f"Sample row {idx}: id={row['id']}, choice='{row['choice']}', prediction_len={len(str(row['prediction']))}, reasoning_len={len(str(row['reasoning']))}")
        
        # Final safety check: ensure choice column has no NULL values or empty strings
        logger.info("Performing final NULL check on choice column...")
        null_patterns = ['NULL', 'null', 'None', 'NaN', 'nan', '<NA>', 'nat', 'NaT', 'NOTAVALUE']
        for pattern in null_patterns:
            count_before = (df['choice'] == pattern).sum()
            if count_before > 0:
                logger.warning(f"Found {count_before} instances of '{pattern}' in choice column, replacing with NOTAVALUE")
                df['choice'] = df['choice'].replace(pattern, 'NOTAVALUE')
        
        # Replace empty strings with NOTAVALUE to avoid NULL validation issues
        empty_count = (df['choice'] == '').sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty strings in choice column, replacing with NOTAVALUE")
            df['choice'] = df['choice'].replace('', 'NOTAVALUE')
        
        # Also replace any remaining pandas nulls in choice column
        null_mask = df['choice'].isnull()
        if null_mask.sum() > 0:
            logger.warning(f"Found {null_mask.sum()} pandas null values in choice column, replacing with NOTAVALUE")
            df.loc[null_mask, 'choice'] = 'NOTAVALUE'
        

        # Use proper CSV parameters for robust handling of complex data
        df.to_csv(csv_path, index=False, na_rep='NOTAVALUE', quoting=1)  # index=False to avoid pandas index issues
        logger.info(f"Successfully saved CSV to {csv_path}")
    
        # Create metadata JSON file
        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create ZIP file with CSV and metadata
        zip_filename = filename.replace('.csv', '.zip')
        zip_path = os.path.join(self.output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV file to zip
            zipf.write(csv_path, filename)
            # Add metadata JSON to zip
            zipf.write(metadata_path, metadata_filename)
        
        # Calculate and log overall accuracy
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
        """
        Convenient method to save submission with user-provided metadata as CSV with zip package
        
        Args:
            results: List of evaluation results
            metadata: User-provided metadata dictionary with fields like:
                - model_name: Name of the model
                - model_type: Type of model wrapper used  
                - track: "internal_reasoning" or "agentic_reasoning"
                - base_model_type: "API" or "OpenWeighted"
                - base_model_name: Name of the base model
                - dataset: Dataset name
                - additional_info: Any additional information
            filename: Output CSV filename
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        # Use the stored dataset examples from the last evaluation
        dataset_examples = getattr(self, '_last_dataset_examples', [])
        
        return self.save_submission(results, filename, metadata, dataset_examples, config_path, args)
    
    def list_datasets(self):
        """List available datasets"""
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def load_metadata_from_config(self, config_path: str) -> Dict:
        """
        Load metadata from configuration file
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Metadata dictionary
        """
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
        
        # Extract metadata from config
        metadata = config.get('metadata', config.get('meta_data', {}))
        
        # Validate required fields
        required_fields = ['model_name', 'track', 'base_model_type', 'base_model_name', 'dataset']
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Required metadata field '{field}' not found in config")
        
        return metadata
    
    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        """
        Parse metadata from command line arguments
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Map argument names to metadata fields
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
        """
        Get metadata from various sources with priority order:
        1. Command line arguments (highest priority)
        2. Configuration file
        3. Fallback metadata provided
        4. Default metadata (lowest priority)
        
        Args:
            config_path: Path to configuration file
            args: Parsed command line arguments
            fallback_metadata: Fallback metadata dictionary
            
        Returns:
            Final metadata dictionary
        """
        # Start with default metadata
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework"
        }
        
        # Override with fallback metadata if provided
        if fallback_metadata:
            metadata.update(fallback_metadata)
        
        # Override with config file metadata if provided
        if config_path:
            try:
                config_metadata = self.load_metadata_from_config(config_path)
                metadata.update(config_metadata)
                logger.info(f"Loaded metadata from config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        
        # Override with command line arguments if provided (highest priority)
        if args:
            arg_metadata = self.parse_metadata_from_args(args)
            metadata.update(arg_metadata)
            if arg_metadata:
                logger.info(f"Applied metadata from command line arguments")
        
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
    """Load configuration from JSON file"""
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
    """Load config file and merge values into args. Command line args take precedence."""
    if not args.config:
        return args
    
    config = load_config_file(args.config)
    
    # First, handle the metadata section specially - merge its contents directly
    if 'metadata' in config:
        metadata = config['metadata']
        for key, value in metadata.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Then handle all other config values, flattening nested structures
    def add_config_to_args(config_dict, prefix=''):
        for key, value in config_dict.items():
            if key in ['metadata', 'dataset']:  # Skip metadata and dataset as we handle them specially
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)
    
    add_config_to_args(config)
    return args
