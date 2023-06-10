"""Contains models."""
from collections.abc import Callable
from llm_chain.base import LLM, MessageMetaData


class OpenAIChat(LLM):
    """
    Input: list of messages expected by OpenAI i.e. list[dict].
    Output: the response.
    """

    def __init__(
            self,
            model_name: str,
            temperature: float = 0,
            max_tokens: int = 2000,
            system_message: str = 'You are a helpful assistant.',
            # this isn't great, because it forces the user to understand how messages work
            # also, we'll need to extract the first system-message and then filter on the rest
            # again, it forces the user to understand this so they know how to implement.
            memory_filter: Callable[[list[dict]], list[dict]] | None = None,
            ) -> None:
        # copied from https://github.com/hwchase17/langchain/blob/master/langchain/callbacks/openai_info.py
        model_cost_per_1k_tokens = {
            'gpt-4': 0.03,
            'gpt-4-0314': 0.03,
            'gpt-4-completion': 0.06,
            'gpt-4-0314-completion': 0.06,
            'gpt-4-32k': 0.06,
            'gpt-4-32k-0314': 0.06,
            'gpt-4-32k-completion': 0.12,
            'gpt-4-32k-0314-completion': 0.12,
            'gpt-3.5-turbo': 0.002,
            'gpt-3.5-turbo-0301': 0.002,
            'text-ada-001': 0.0004,
            'ada': 0.0004,
            'text-babbage-001': 0.0005,
            'babbage': 0.0005,
            'text-curie-001': 0.002,
            'curie': 0.002,
            'text-davinci-003': 0.02,
            'text-davinci-002': 0.02,
            'code-davinci-002': 0.02,
            'ada-finetuned': 0.0016,
            'babbage-finetuned': 0.0024,
            'curie-finetuned': 0.012,
            'davinci-finetuned': 0.12,
        }
        super().__init__(
            model_type='chat',
            # token_counter=None,
            # cost_per_token=model_cost_per_1k_tokens[model_name] / 1000,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_per_token = model_cost_per_1k_tokens[model_name] / 1000
        self.memory_filter = memory_filter
        self._messages=[
            {'role': 'system', 'content': system_message},
        ]
        self._previous_memory = None

    def _run(self, prompt: str) -> MessageMetaData:
        import openai
        self._messages += [{'role': 'user', 'content': prompt}]
        self._previous_memory = self._messages.copy()
        if self.memory_filter:
            self._previous_memory = self.memory_filter(self._previous_memory)
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self._previous_memory,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response_message = response['choices'][0]['message'].content
        self._messages += [{'role': 'assistant', 'content': response_message}]
        return MessageMetaData(
            prompt=prompt,
            response=response_message,
            metadata={'model_name': self.model_name},
            prompt_tokens=response['usage'].prompt_tokens,
            response_tokens=response['usage'].completion_tokens,
            total_tokens=response['usage'].total_tokens,
            cost=response['usage'].total_tokens * self.cost_per_token,
        )

