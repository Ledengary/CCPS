import os
import torch
from vllm import LLM

class Talk2LLM:
    def __init__(self, model_id, dtype, gpu_memory_utilization=0.5, tensor_parallel_size=2, enforce_eager=None, 
                 task="auto", tokenizer_mode=None, config_format=None, quantization=None, load_format=None, the_seed=23):
        """
        Initialize the LLM model with specified configurations.
        """
        # Map dtype strings to torch dtypes
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "": None
        }
        if dtype is None:
            chosen_dtype = None
        else:
            chosen_dtype = dtype_mapping.get(dtype)

        print('visible CUDAs for vllm:', os.environ.get('CUDA_VISIBLE_DEVICES'))
        print('Using dtype:', chosen_dtype if chosen_dtype is not None else "default settings")

        # Prepare keyword arguments for LLM initialization
        llm_kwargs = {
            "model": model_id,
            "task": task,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": enforce_eager,
            "seed": the_seed,
            "trust_remote_code": True
        }

        if chosen_dtype is not None:
            llm_kwargs["dtype"] = chosen_dtype
        if tokenizer_mode is not None:
            llm_kwargs["tokenizer_mode"] = tokenizer_mode
        if load_format is not None:
            llm_kwargs["load_format"] = load_format
        if config_format is not None:
            llm_kwargs["config_format"] = config_format
        if dtype is not None:
            llm_kwargs["dtype"] = chosen_dtype
        if quantization is not None:
            llm_kwargs["quantization"] = quantization
        if "phi" in model_id.lower():
            llm_kwargs["max_model_len"] = 8192
        print('llm_kwargs', llm_kwargs)
        self.llm = LLM(**llm_kwargs)
        print('self.llm', self.llm)

    def single_query(self, prompt, temperature=1.0, max_tokens=100):
        """
        Generates a single response for a given prompt.
        """
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        output = self.llm.generate([prompt], sampling_params)
        return output[0].outputs[0].text.strip()

    def batch_chat_query(self, conversations, temperature=1.0, max_tokens=100, use_tqdm=True, chat_template_content_format="openai"):
        """
        Runs batched inference for a list of user prompts, all using the same system prompt.
        """
        from vllm import SamplingParams
        if isinstance(temperature, list):
            assert len(temperature) == len(conversations), f"Temperature list must be the same length as the number of conversations: {len(temperature)} != {len(conversations)}"
            sampling_params = [SamplingParams(temperature=t, max_tokens=max_tokens) for t in temperature]
        elif isinstance(temperature, int) or isinstance(temperature, float):
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        else:
            raise ValueError(f"Invalid temperature type: {type(temperature)}")
        outputs = self.llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=use_tqdm, chat_template_content_format=chat_template_content_format)
        return [output.outputs[0].text.strip() for output in outputs]