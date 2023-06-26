# `llm-chain`: simple and extensible LLM chaining

A `chain` consists of `links`. Each link in the chain is a callable, which can be either a function or an object that implements the `__call__` method. **The output of one link serves as the input to the next link in the chain.** Pretty simple.

Additionally, each link can track its own history, including messages sent/received and token usage/costs, through a `history` property that returns a list of `Record` objects. A `chain` aggregates and propagates the history of any link that has a `history` property, making it convenient to analyze costs or explore intermediate steps in the chain.

**Note: This package is tested on Python versions 3.10 and 3.11**

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
- [indexes.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/indexes.ipynb)
- [prompt_templates.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/prompt_templates.ipynb)
- [memory.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/memory.ipynb)


## Simple ChatGPT example - **no chaining**

You can use the objects from llm-chain without a chain as well. Here's an example using the OpenAIChat model:

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

Here's an example of using a chain to perform the following series of tasks:

- Ask a question.
- Perform a web search based on the question.
- Scrape the top_n web pages from the search results.
- Split the web pages into chunks (so that we can search for the most relevant chunks).
- Save the chunks to a document index (i.e. vector database).
- Create a prompt that includes the original question and the most relevant chunks.
- Send the prompt to the chat model.
- Create a second prompt that asks the model to summarize the previous response.
- Send the second prompt to the chat model.

In the code below, you can replace any step with your own implementation as long as the input/output matches the link you replace.

**Again, the key concept of a chain is simply that the output of one link is the input of the next link.** So, in the code below, you can replace any step with your own implementation as long as the input/output matches the link you replace.

Something that may not be immediately obvious is the usage of the `Value` object, below. It serves as a convenient caching mechanism within the chain. The `Value` object is callable, allowing it to cache and return a value when provided as an argument. When called without a value, it retrieves and returns the cached value. In the given context, the `Value` object is utilized to cache the original question, pass it to the web search, and subsequently reintroduce the question into the chain and prompt template.

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

question_1 = Value()  # Value is a caching/reinjection mechanism; see note above
question_2 = lambda x: f'Summarize the following in less than 20 words: "{x}"'

# each link is a callable where the output of one link is the input to the next link
chain = Chain(links=[
    question_1,
    duckduckgo_search,
    scrape_urls,
    split_documents,
    document_index,
    question_1,
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

- [ ] Create PDF-Loader
- [ ] create additional prompt-templates

---

## Contributing

Contributions to this project are welcome. Please follow the coding standards, add appropriate unit tests, and ensure that all linting and tests pass before submitting pull requests.

### Coding Standards

- Coding standards should follow PEP 8 (Style Guide for Python Code).
    - https://peps.python.org/pep-0008/
    - Exceptions:
        - Use a maximum line length of 99 instead of the suggested 79.
- Document all files, classes, and functions following the existing documentation style.

### Docker

See the Makefile for all available commands.

To build the Docker container:

```commandline
make docker_build
```

To run the terminal inside the Docker container:

```commandline
make docker_zsh
```

To run the unit tests (including linting and doctests) from the command line inside the Docker container:

```commandline
make tests
```

To run the unit tests (including linting and doctests) from the command line outside the Docker container:

```commandline
make docker_tests
```

### Pre-Check-in

#### Unit Tests

The unit tests in this project are all found in the `/tests` directory.

In the terminal, in the project directory, run the following command to run linting and unit-tests:

```commandline
make tests
```
