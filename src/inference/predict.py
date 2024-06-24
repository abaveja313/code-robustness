import copy
import os
from typing import Any, Optional

import numpy as np
from loguru import logger
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from inference.dataset_manager import DatasetManager
from inference.processors import Processors, PostprocessingException
from shared.program_utils import program_concat
from shared.structs import MutatedStem, Solution, BatchSolution, BenchmarkResult, SolutionType
from shared.logging_utils import log_time


class InferenceEngine:
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

    def __init__(
            self,
            model_name: str,
            dataset_manager: DatasetManager,
            sampling_args: dict[str, Any] = None,
            top_p: float = 0.95,
            max_tokens: int = 1024,
            enable_prefix_caching: bool = False,
            direct_completion: bool = False,
            tokenizer: Optional[str] = None
    ):
        model_kwargs = {
            "tensor_parallel_size": int(os.getenv("VLLM_N_GPUS", 1)),
            "trust_remote_code": True,
            "max_model_len": max_tokens,
            "enable_prefix_caching": enable_prefix_caching,
            "model": model_name,
        }

        self.llm = LLM(**model_kwargs)

        self.model_name = model_name
        self.direct_completion = direct_completion

        logger.info("Using model '{}' with params {}".format(model_name, sampling_args))

        self.dataset = dataset_manager
        self.eos = [
            "<|endoftext|>",
            "<|endofmask|>",
            "</s>",
            "\nif __name__",
            "\ndef main(",
            "\nprint(",
            "\n#"
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or self.model_name)
        self.add_eos_for_task()
        self.sampling_params = SamplingParams(
            top_p=top_p,
            max_tokens=max_tokens,
            stop=self.eos,
            **(sampling_args or {})
        )
        logger.info("Sampling Parameters: {}", self.sampling_params)

    def add_eos_for_task(self):
        if self.direct_completion:
            self.eos += ['\ndef ', '\nclass ', '\nimport ', '\nfrom ', '\nassert ',
                         '\n def ', '\n class ', '\n import ', '\n from ', '\nif', '\n if', '\nwhile',
                         '\n while', '\nfor', '\n for', '\ntry', '\n try', '\nwith', '\n with', '\nraise',
                         '\n raise', '\nassert', '\n assert', "\n'''", '\n"""']

        self.eos += ["\n```\n", "```", "\nassert", "\ndef", "# Test", "# test", "\ndef", "\nclass"]

    def make_function_codegen_prompt(self, problem_id: str) -> str:
        definition = self.dataset.get_problem(problem_id)["formatted_prompt"]
        # directly return prompt if it does not have a tokenizer.chat_template
        if self.direct_completion:
            return definition.strip()

        query = (
            "Complete the body of the below Python function such that it is self-contained and passes the "
            "corresponding tests. Write your code in a markdown code block, ending your response with ```. "
            "Don't include any testcases in your response.\n"
            "```python\n"
            f"{definition.strip()}\n"
            "```"
        )

        response = (
            "Below is the completed function body that solves the problem and passes corresponding tests:\n"
            "```python\n"
            f"{definition.strip()}\n"
            f"{self._MAGIC_SPLITTER_}\n"
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

    def make_stem_completion_prompt(self, stem: str):
        if self.direct_completion:
            return stem.strip()

        query = (
            "Complete the rest of the below function such that it is self-contained and passes the "
            "corresponding tests. Write your code in a markdown code block, ending your response with ```. "
            "The function does not execute any tests of its logic. Don't include any testcases or evaluate your "
            "response.\n\n"
        )
        response = (
            "Below is the rest of the function body such that it passes the corresponding tests:\n"
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

    def get_sampling_params(self, num_samples: int, temp: float) -> SamplingParams:
        new_sampling_params = copy.deepcopy(self.sampling_params)
        new_sampling_params.temperature = temp
        new_sampling_params.n = num_samples
        return SamplingParams(**new_sampling_params)

    def generate(self, prompts: dict[str, str], num_samples: int, temp: float):
        new_sampling_params = self.get_sampling_params(num_samples, temp)
        sorted_prompts = sorted(prompts.items(), key=lambda x: x[1])
        prompt_ids, prompt_conts = list(zip(*sorted_prompts))

        sequences = {}

        with log_time("Sampling {} sequences".format(num_samples)):
            model_outputs = self.llm.generate(prompt_conts, sampling_params=new_sampling_params)

        for prompt_id, prompt_gen in zip(prompt_ids, model_outputs):
            sequences[prompt_id] = []
            for output in prompt_gen.outputs:
                sequence = {
                    'text': output.text,
                    'cumulative_logprob': output.cumulative_logprob
                }
                sequences[prompt_id].append(sequence)

        return sequences

    def predict_solutions(self, problem_id: str, num_samples: int = 200, temperature: float = 0.8):
        prompt = {"sample": self.make_function_codegen_prompt(problem_id)}

        logger.debug("Prompt:\n{}", prompt)

        sequences = self.generate(prompt, num_samples, temperature)['sample']

        errors = []
        batch_solution = BatchSolution()
        problem = self.dataset.get_problem(problem_id)
        for sequence in sequences:
            solution = Solution(
                code=program_concat(problem["formatted_prompt"], sequence['text']),
                probs=sequence['cumulative_logprob']
            )
            original_code = copy.copy(solution.code)
            try:
                solution.post_process(direct=self.direct_completion)
                batch_solution.add_solution(solution)
            except Exception:
                logger.exception(f"Error postprocessing solution:\n{original_code}")
                errors.append(PostprocessingException(code=solution.code))
                batch_solution.add_solution(Solution(code='', probs=0.0))

        return batch_solution, errors

    def complete_stems(self, stem: MutatedStem, temperature: float, num_samples: int = 200):
        prompts = {"original": stem.original_stem, "mutated": stem.mutated_stem}
        for prompt in prompts:
            prompts[prompt] = Processors.preprocess_stem(prompt)
            prompts[prompt] = self.make_stem_completion_prompt(prompts[prompt])

        for prompt in prompts:
            logger.debug("Prompt:\n{}", prompt)

        batch_solutions = dict(original=BatchSolution(), mutated=BatchSolution())

        outputs = self.generate(prompts, num_samples, temperature)
        print(outputs)

        errors = []
        last_solution = None

        for prompt_id in outputs:
            for sequence in outputs[prompt_id]:
                prefix = stem.original_stem if prompt_id == "original" else stem.mutated_stem
                solution = Solution(
                    code=program_concat(prefix, sequence['text']),
                    probs=sequence['cumulative_logprob'],
                )
                last_solution = solution
                try:
                    solution.post_process(direct=self.direct_completion)
                    batch_solutions[prompt_id].add_solution(solution)
                except Exception:
                    logger.exception(f"Error postprocessing solution:\n{solution.code}")
                    errors.append(
                        PostprocessingException(
                            code=solution.code, mutated=prompt_id == "mutated"
                        )
                    )
                    batch_solutions[prompt_id].add_solution(Solution(code='', probs=0.0))

        logger.info(f"Last Solution:\n{last_solution.code}")
        return batch_solutions, errors

    def sample_stem_solutions(
            self,
            stem: MutatedStem,
            result: BenchmarkResult,
            temp: float,
            num_samples: int = 200
    ):
        logger.info(
            "Completing tests (@T{}) for:\n===========\nOld:\n{}\n\nMutated:\n{}",
            temp,
            stem.original_stem,
            stem.mutated_stem,
        )
        result.add_stem(stem)

        predictions, errors = self.complete_stems(
            stem=stem, num_samples=num_samples,
            temperature=temp
        )

        logger.warning("Found {} errors during postprocessing", len(errors))

        for error in errors:
            result.add_example(
                example=error.code,
                solution_type=SolutionType.BAD_PROCESS,
                mutated=error.mutated,
            )

        original_predictions = predictions["original"].get_code()
        mutated_predictions = predictions["mutated"].get_code()
        return dict(
            original=original_predictions,
            mutated=mutated_predictions
        )
