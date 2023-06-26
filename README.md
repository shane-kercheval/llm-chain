# llm-chain

Simple and extensible LLM chaining.

- A `chain` consists of `links`. A link is a callable (either a function or an object that implements `__call__`). **The output of one link is the input to the next link.** Pretty simple.
- Each `link` can track it's own history (e.g. messages sent to/from chat model and corresponding token usage/costs) via a `history` property that returns a list of `Record` objects.
- A `chain` aggregates the history of any `link` that has a `history` property. This provides an easy way aggregate costs or to explore any intermediate steps in the link.

**NOTE: This package is tested on Python `3.10` and `3.11`**

---

# Installing

```commandline
pip install llm-chain
```

---

# Examples

## Notebooks

- [chains.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb)
- [openai_chat.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/openai_chat.ipynb)
- [tools.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/tools.ipynb)
- [memory.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/memory.ipynb)


## Simple ChatGPT example - **no chaining**

We can, of course, use the objects without a chain.

```python
from llm_chain.models import OpenAIChat
model = OpenAIChat(model_name='gpt-3.5-turbo')
model("What is the meaning of life?")
```

Response:

```
The meaning of life is a philosophical question that has been debated by scholars, theologians, and philosophers for centuries. There is no one definitive answer to this question, as it can vary depending on one's beliefs, values, and experiences. Some people believe that the meaning of life is to seek happiness, while others believe it is to fulfill a specific purpose or destiny. Ultimately, the meaning of life is a personal and subjective concept that each individual must determine for themselves.
```

## Using a Chain

Here's an example where we chain together the following tasks:

- ask a question
- do a web-search
- scrape the top_n web-pages
- split the pages up into chunks
- save the chunks to document index (i.e. vector database)
- create a prompt that includes the original question along with the most relevant chunks
- send the prompt to the chat model
- create a second prompt that asks the model to summarize the response
- send the second prompt to the chat model

One thing to note is the `Value` object being used below. It's just a simple caching mechanism. It's a callable that, when passed a value, it caches and returns that value; and when called without a value, it returns the cached value. Below, it's being used to cache the original question, feed the question into the web-search, and then re-inject the question back in the chain into the prompt-template.

See [this notebook](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb) for an in-depth explanation.

```python
from llm_chain.base import Document, Chain, Value
from llm_chain.models import OpenAIEmbeddings, OpenAIChat
from llm_chain.tools import DuckDuckGoSearch, scrape_url, split_documents
from llm_chain.indexes import ChromaDocumentIndex
from llm_chain.prompt_templates import DocSearchTemplate

duckduckgo_search = DuckDuckGoSearch(top_n=3)
embeddings_model = OpenAIEmbeddings(model_name='text-embedding-ada-002')
document_index = ChromaDocumentIndex(embeddings_model=embeddings_model, n_results=3)
prompt_template = DocSearchTemplate(doc_index=document_index, n_docs=3)
chat_model = OpenAIChat(model_name='gpt-3.5-turbo')

def scrape_urls(search_results):
    """
    For each url (i.e. `href` in `search_results`):
    - extracts text
    - replace new-lines with spaces
    - create a Document object
    """
    return [
        Document(content=scrape_url(x['href']).replace('\n', ' '))
        for x in search_results
    ]

initial_question = Value()  # see note above
question_2 = lambda x: f'Summarize the following in less than 20 words: "{x}"'

# each link is a callable where the output of one link is the input to the next link
chain = Chain(links=[
    initial_question,
    duckduckgo_search,
    scrape_urls,
    split_documents,
    document_index,
    initial_question,
    prompt_template,
    chat_model,
    question_2,
    chat_model,
])
chain("What is the meaning of life?")
```

Response:

```
The meaning of life is subjective and each person must answer it for themselves; there is no one answer.
```

We can also track costs:

```python
print(f"Cost:   ${chain.cost:.4f}")
print(f"Tokens: {chain.total_tokens:,}")
```

Output:

```
Cost:   $0.0054
Tokens: 45,674
```

Additionally, we can track the history of the chain with the `chain.history` property. See [this notebook](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb) for an example.

---

# TODO

- [ ] PDF Loader

---

## Contributing

### Coding Standards

- Coding standards should follow PEP 8 (Style Guide for Python Code)
    - https://peps.python.org/pep-0008/
    - Exceptions:
        - use max line length of `99` rather than the suggested `79`
- document all files, classes, functions
    - following existing documentation style


### Docker

See `Makefile` for all commands.

To build the docker container:

```commandline
make docker_build
```

To run the terminal inside the docker container:

```commandline
make docker_zsh
```

To run the unit tests (including linting and doc-tests) from the commandline inside the docker container:

```commandline
make tests
```

To run the unit tests (including linting and doc-tests) from the commandline outside the docker container:

```commandline
make docker_tests
```

To build the python package and uploat do PyPI via twine from the commandline outside the docker container:

```commandline
make all
```

### Pre-Check-in

#### Unit Tests

The unit tests in this project are all found in the `/tests` directory.

In the terminal, in the project directory, either run the Makefile command,

```commandline
make tests
```

or the python command for running all tests

```commandline
python -m unittest discover ./tests
```
