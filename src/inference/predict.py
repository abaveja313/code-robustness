import os
from typing import Any

from loguru import logger
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM

from inference.dataset_manager import DatasetManager, Dataset
from inference.processors import Processors, PostprocessingException
from shared.program_utils import program_concat
from shared.structs import MutatedStem, Solution, BatchSolution


class InferenceEngine:
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

    def __init__(
            self,
            model_name: str,
            dataset_manager: DatasetManager,
            direct_completion: bool = False,
            dtype: str = "bfloat16",
            trust_remote_code: bool = False,
            enable_prefix_caching: bool = True,
            max_model_len: int = 2048,
            model_params: dict[str, Any] = None,
            **sampling_params
    ):
        model_kwargs = {
            "tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", 1)),
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            'max_model_len': max_model_len,
            'enable_prefix_caching': enable_prefix_caching,
            'model': model_name
        }

        if model_params is not None:
            model_kwargs.update(model_params)

        logger.info("Using model '{}' with params {}".format(model_name, model_kwargs))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if direct_completion is None:
            self.direct_completion = self.tokenizer.chat_template is not None
        else:
            self.direct_completion = direct_completion

        self.llm = LLM(**model_kwargs)

        self.dataset = dataset_manager
        self.eos = [
            "<|endoftext|>",
            "<|endofmask|>",
            "</s>",
            "\nif __name__",
            "\ndef main(",
            "\nprint(",
        ]
        self.add_eos_for_task()
        logger.info("Model EOS Terminators: {}", self.eos)

        sampling_params['stop'] = self.eos
        self.sampling_args = SamplingParams(**sampling_params)

    def add_eos_for_task(self):
        if self.direct_completion:
            if self.dataset.dataset_name == Dataset.HUMANEVAL:
                self.eos += ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
            elif self.dataset.dataset == Dataset.MBPP:
                self.eos += ["\nclass "]
        else:
            self.eos += ["\n```\n", "```", "\nassert", "assert"]

    def make_function_codegen_prompt(self, problem_id: str) -> str:
        definition = self.dataset.get_problem(problem_id)['formatted_prompt']
        # directly return prompt if it does not have a tokenizer.chat_template
        query = ("Complete the body of the below Python function such that it is self-contained and passes the "
                 "corresponding tests. Write your code in a markdown code block, ending your response with ```. "
                 "Don't include any testcases in your response.\n"
                 "```python\n"
                 f"{definition.strip()}\n"
                 "```")

        response = ("Below is the completed function body that solves the problem and passes corresponding tests:\n"
                    "```python\n"
                    f"{definition.strip()}\n"
                    f"{self._MAGIC_SPLITTER_}\n"
                    "```")

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        ).split(self._MAGIC_SPLITTER_)[0]
        return prompt

    def make_stem_completion_prompt(self, stem: str):
        query = (
            "Complete the rest of the below function such that it is self-contained and passes the "
            "corresponding tests. Write your code in a markdown code block, ending your response with ```. "
            "The function does not execute any tests of its logic. Don't include any testcases or evaluate your "
            "response.\n\n"
            "```python\n"
            f"{stem.strip()}\n"
            "```"
        )
        response = (
            "Below is the rest of the function body such that it passes the corresponding tests:\n:"
            "```python\n"
            f"{stem.strip()}\n"
            f"{self._MAGIC_SPLITTER_}"
            "```"
        )

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        ).split(self._MAGIC_SPLITTER_)[0]
        return prompt

    def generate(self, prompts: list[str], num_samples: int):
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_samples)

        vllm_outputs = self.llm.generate(
            expanded_prompts,
            self.sampling_args,
            use_tqdm=False,
        )

        outputs = [output.outputs[0] for output in vllm_outputs]
        return outputs

    def predict_solutions(self, problem_ids: list[str], num_samples: int = 200):
        if self.direct_completion:
            prompts = [self.dataset.get_problem(problem_id)['formatted_prompt'] for problem_id in problem_ids]
        else:
            prompts = [self.make_function_codegen_prompt(problem_id) for problem_id in problem_ids]

        for prompt in prompts:
            logger.debug("Prompt:\n{}", prompt)

        vllm_outputs = self.generate(prompts, num_samples)
        sequences = Processors.split_sequences(vllm_outputs, problem_ids, num_samples)

        post_processed, errors = {}, []
        for problem_id in sequences:
            problem = self.dataset.get_problem(problem_id)
            solutions = []
            for sequence in sequences[problem_id]:
                solution = Solution(
                    code=program_concat(problem['formatted_prompt'], sequence.text),
                    probs=sequence.cumulative_logprob
                )
                try:
                    solution.post_process()
                    solutions.append(solution)
                except Exception:
                    logger.exception(f"Error postprocessing solution:\n{solution.code}")
                    errors.append(PostprocessingException(code=solution.code))

            post_processed[problem_id] = BatchSolution(solutions=solutions)
        return post_processed, errors

    def complete_stems(self, stem: MutatedStem, num_samples: int = 200):
        prompts = [Processors.preprocess_stem(s) for s in [stem.original_stem, stem.mutated_stem]]
        if not self.direct_completion:
            prompts = [self.make_stem_completion_prompt(s) for s in prompts]

        vllm_outputs = self.generate(prompts, num_samples)
        sequences = Processors.split_sequences(vllm_outputs, ['original', 'mutated'], num_samples)

        post_processed, errors = {}, []
        for stem_name, stem_val in stem.as_tuple():
            solutions = []
            for sequence in sequences[stem_name]:
                solution = Solution(
                    code=program_concat(stem_val, sequence.text),
                    probs=sequence.cumulative_logprob
                )
                try:
                    solution.post_process()
                    solutions.append(solution)
                except Exception:
                    logger.exception(f"Error postprocessing solution:\n{solution.code}")
                    errors.append(
                        PostprocessingException(code=solution.code, mutated=stem_name == 'mutated')
                    )
            post_processed[stem_name] = BatchSolution(solutions=solutions)
        return post_processed, errors
