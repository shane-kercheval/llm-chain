"""Contains resources such as shared variables."""

# copied from https://github.com/hwchase17/langchain/blob/master/langchain/callbacks/openai_info.py
MODEL_COST_PER_1K_TOKENS = {
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



####
# Prompts
####

PROMPT_TEMLATE__DOC_SEARCH_STUFF = \
"""
Answer the question at the end of the text as truthfully and accurately as possible, based on the following information provided.

Here is the information:

```
{{documents}}
```

Here is the question:

{{prompt}}
"""  # noqa: E501
