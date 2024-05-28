import textwrap
from typing import Dict, Any

from transformers import AutoTokenizer

from inference.models import VllmDecoder
from shared.ast_utils import get_function_declaration_line
from gettext import gettext


class InferenceEngine:
    def __init__(
            self,
            model: VllmDecoder,
            problem: Dict[str, Any]
    ):
        self.model = model
        self.problem = problem

        self.prompt_formatter =

    def get_prompt_formatter(self):
        if isinstance(model, GeneralV)
    def make_codegen_prompt(self, prompt: str) -> str:
        # directly return prompt if it does not have a tokenizer.chat_template
        if self.model.tokenizer.chat_template is None:
            return prompt

        query = f"""
        Complete the body of the Python function such that it passes corresponding tests. Write your code in a markdown
        code block, ending your response with ```. Don't include any testcases in your response.
        ```python
        {prompt.strip()}
        ```
        """
        query =

        response = f"""Below is the completed function body that solves the problem and passes corresponding tests:
    ```python
    {func_decl}
    {_MAGIC_SPLITTER_}
    ```"""
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER_)[0]
        return prompt
