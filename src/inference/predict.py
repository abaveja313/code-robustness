from typing import Any

import numpy as np
from loguru import logger
from openai import OpenAI
from transformers import AutoTokenizer

from inference.dataset_manager import DatasetManager
from inference.processors import Processors, PostprocessingException
from shared.program_utils import program_concat
from shared.structs import MutatedStem, Solution, BatchSolution, BenchmarkResult, SolutionType
from shared.logging_utils import prob_log


class InferenceEngine:
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

    def __init__(
            self,
            model_name: str,
            dataset_manager: DatasetManager,
            server_url: str,
            sampling_args: dict[str, Any] = None,
            top_p: float = 0.95,
            max_tokens: int = 1024
    ):
        self.llm = OpenAI(
            api_key="EMPTY",
            base_url=server_url
        )

        self.model_name = model_name

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.add_eos_for_task()
        logger.info("Model EOS Terminators: {}", self.eos)

        self.sampling_params = sampling_args or {}
        self.sampling_params["stop"] = self.eos
        self.sampling_params |= {
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

    def add_eos_for_task(self):
        self.eos += ["\n```\n", "```", "\nassert", "assert", "\ndef", "# Test", "# test", "def test", "def main"]

    def make_function_codegen_prompt(self, problem_id: str) -> str:
        definition = self.dataset.get_problem(problem_id)["formatted_prompt"]
        # directly return prompt if it does not have a tokenizer.chat_template
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

    def generate(self, prompt: str, num_samples: int, temp: float, logprobs: bool):
        model_outputs = self.llm.completions.create(
            model=self.model_name,
            prompt=prompt,
            n=num_samples,
            **(self.sampling_params | {"temperature": temp, "logprobs": logprobs})
        )
        sequences = []
        for output in model_outputs.choices:
            sequence = {
                'text': output.text,
                'cumulative_logprob': np.sum(output.logprobs.token_logprobs) if logprobs else -1.0
            }
            sequences.append(sequence)

        return sequences

    def predict_solutions(self, problem_id: str, num_samples: int = 200, temperature: float = 0.8):
        prompt = self.make_function_codegen_prompt(problem_id)

        logger.debug("Prompt:\n{}", prompt)

        sequences = self.generate(prompt, num_samples, temperature, logprobs=True)

        errors = []
        batch_solution = BatchSolution()
        problem = self.dataset.get_problem(problem_id)
        for sequence in sequences:
            solution = Solution(
                code=program_concat(problem["formatted_prompt"], sequence['text']),
                probs=sequence['cumulative_logprob']
            )
            try:
                solution.post_process()
                batch_solution.add_solution(solution)
            except Exception:
                logger.exception(f"Error postprocessing solution:\n{solution.code}")
                errors.append(PostprocessingException(code=solution.code))
                batch_solution.add_solution(Solution(code='', probs=0.0))

        return batch_solution, errors

    def complete_stems(self, stem: MutatedStem, temperature: float, num_samples: int = 200):
        prompts = [
            Processors.preprocess_stem(s)
            for s in [stem.original_stem, stem.mutated_stem]
        ]

        prompts = [self.make_stem_completion_prompt(s) for s in prompts]

        for prompt in prompts:
            logger.debug("Prompt:\n{}", prompt)

        batch_solutions = dict(original=BatchSolution(), mutated=BatchSolution())

        original_outputs = self.generate(prompts[0], num_samples, temperature, logprobs=False)
        mutated_outputs = self.generate(prompts[1], num_samples, temperature, logprobs=False)

        errors = []
        for prompt, sequences, stem_name in zip(prompts, [original_outputs, mutated_outputs], ["original", "mutated"]):
            for sequence in sequences:
                prefix = stem.original_stem if stem_name == "original" else stem.mutated_stem
                solution = Solution(
                    code=program_concat(prefix, sequence['text']),
                    probs=sequence['cumulative_logprob'],
                )
                logger.info(f"Solution:\n{solution.code}")
                try:
                    solution.post_process()
                    batch_solutions[stem_name].add_solution(solution)
                except Exception:
                    logger.exception(f"Error postprocessing solution:\n{solution.code}")
                    errors.append(
                        PostprocessingException(
                            code=solution.code, mutated=stem_name == "mutated"
                        )
                    )
                    batch_solutions[stem_name].add_solution(Solution(code='', probs=0.0))
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
