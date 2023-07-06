# `llm-chain`: simple and extensible LLM chaining

A `chain` is an object that executes a sequence of tasks known as `links`. Each link in the sequence is a callable, which can take the form of a function or an object implementing the `__call__` method. **The output of one link serves as the input to the next link in the chain.** Pretty simple...

The purpose of this library is to provide a simple pattern for developing LLM workflows. First, this allows users to avoid writing boilerplace code. Second, by developing a common interface across links (i.e. by specifying what a link is and what it can do), a chain can become a mechanism for aggregating information across all links (e.g. token usage, costs, etc.).

More specifically, if we define a link as being a callable object that has the option to track its own history (e.g. a chat model that tracks its history of messages, token usage, costs, etc.), then a chain is able to track and aggregate the history of all links across that chain. It also gives us a mechanism to view each step in the chain and within a specific link, making it less of a black box and easier to debug and understand.

As a result, some of the classes provided in this library (e.g. `OpenAIEmbedding` or `ChromaDocumentIndex`) are nothing more than simple wrappers that implement the interface necessary to track history in a consistent way, allowing the chain to aggregate the history across all links. This also makes it easy for people to create their own wrappers and workflows.

See examples below.

---

# Installing

**Note: This package is tested on Python versions 3.10 and 3.11**

```commandline
pip install llm-chain
```

## API KEYs

- Any classes that use OpenAI assume that the `OPENAI_API_KEY` environment variable is set to a valid OpenAI API key.
- The `llm_chain.utils.search_stack_overflow()` function assumes that the `STACK_OVERFLOW_KEY` environment variable is set. You must create an account and app at [Stack Apps](https://stackapps.com/) and use the `key` that is generated (not the `secret`).

---

# Examples

## Example 1

- The first link is a function (`prompt_template`) where the input is the initial value passed to chain ("adding two numbers"); the output is a modified prompt that is sent to the next link
- The second link is the chat model which takes the modified prompt from the previous link and returns a response from the underlying OpenAI model ('gpt-3.5-turbo').
- The third link (`prompt_extract_code`) ignores the response from the previous link, and returns a new prompt asking the model to extract/return only the code that was generated in it's previous response. The `OpenAIChat` class manages the history of messages and by default passes all previous messages to the OpenAI model; so the underlying model will have access to the full conversation.  
- The fourth/final link is the same chat model and is passed the modified prompt; the response is returned by the chain since this is the final link.

```python
from llm_chain.base import Chain
from llm_chain.models import OpenAIChat

chat_model = OpenAIChat(model_name='gpt-3.5-turbo')

def prompt_template(prompt: str) -> str:
    return f"Write a python function for: ```{prompt}```"

def prompt_extract_code(_: str) -> str:
    # `_` ignores the input from previous chain
    return "Return only the function from the previous answer, without text"

chain = Chain(links=[
    prompt_template,
    chat_model,
    prompt_extract_code,
    chat_model
])
response = chain("adding two numbers")
print(response)
```

Output:

```python
def add_numbers(num1, num2):
    return num1 + num2
```

Total costs/tokens for all activity in the chain:

```python
print(f"Cost:             ${chain.cost:.4f}")
print(f"Total Tokens:      {chain.total_tokens:,}")
print(f"Prompt Tokens:     {chain.prompt_tokens:,}")
print(f"Response Tokens:   {chain.response_tokens:,}")
```

Output:

```
Cost:              $0.00046
Total Tokens:       268
Prompt Tokens:      161
Response Tokens:    107
```

History:

```python
[type(x) for x in chain.history]
```

Output:

```
[llm_chain.base.ExchangeRecord, llm_chain.base.ExchangeRecord]
```

An ExchangeRecord represents a single exchange/transaction with an LLM, encompassing an input (prompt) and its corresponding output (response), along with other properties like cost and token usage.

```python
print(chain.history[0].prompt)
print(chain.history[0].response)
print(chain.history[1].prompt)
print(chain.history[1].response)
```

Output:

```
Write a python function for: ```adding two numbers```
```

```
Certainly! Here's a Python function that adds two numbers:

def add_numbers(num1, num2):
    return num1 + num2

You can call this function by passing two numbers as arguments, like this:

result = add_numbers(5, 3)
print(result)  # Output: 8

Feel free to modify the function and use it as needed.
```

```
Return only the function from the previous answer, without text
```

```
def add_numbers(num1, num2):
    return num1 + num2
```

---

## Example 2

Here's an example of using a chain to perform the following series of tasks:

- Ask a question.
- Perform a web search based on the question.
- Scrape the top_n web pages from the search results.
- Split the web pages into chunks (so that we can search for the most relevant chunks).
- Save the chunks to a document index (i.e. vector database).
- Create a prompt that includes the original question and the most relevant chunks.
- Send the prompt to the chat model.

**Again, the key concept of a chain is simply that the output of one link is the input of the next link.** So, in the code below, you can replace any step with your own implementation as long as the input/output matches the link you replace.

Something that may not be immediately obvious is the usage of the `Value` object, below. It serves as a convenient caching mechanism within the chain. The `Value` object is callable, allowing it to cache and return the value when provided as an argument. When called without a value, it retrieves and returns the cached value. In the example below, the `Value` object is used to cache the original question, pass it to the web search (i.e. the `duckduckgo_search` object), and subsequently reintroduce the question into the chain (i.e. into the `prompt_template` object).

See [this notebook](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb) for an in-depth explanation.

```python
from llm_chain.base import Document, Chain, Value
from llm_chain.models import OpenAIEmbedding, OpenAIChat
from llm_chain.tools import DuckDuckGoSearch, scrape_url, split_documents
from llm_chain.indexes import ChromaDocumentIndex
from llm_chain.prompt_templates import DocSearchTemplate

duckduckgo_search = DuckDuckGoSearch(top_n=3)
embeddings_model = OpenAIEmbedding(model_name='text-embedding-ada-002')
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
        Document(content=re.sub(r'\s+', ' ', scrape_url(x['href'])))
        for x in search_results
    ]

question = Value()  # Value is a caching/reinjection mechanism; see note above

# each link is a callable where the output of one link is the input to the next link
chain = Chain(links=[
    question,
    duckduckgo_search,
    scrape_urls,
    split_documents,
    document_index,
    question,
    prompt_template,
    chat_model,
])
chain("What is ChatGPT?")
```

Response:

```
ChatGPT is an AI chatbot that is driven by AI technology and is a natural language processing tool. It allows users to have human-like conversations and can assist with tasks such as composing emails, essays, and code. It is built on a family of large language models known as GPT-3 and has now been upgraded to GPT-4 models. ChatGPT can understand and generate human-like answers to text prompts because it has been trained on large amounts of data.'
```

We can also track costs:

```python
print(f"Cost:            ${chain.cost:.4f}")
print(f"Total Tokens:     {chain.total_tokens:,}")
print(f"Prompt Tokens:    {chain.prompt_tokens:,}")
print(f"Response Tokens:  {chain.response_tokens:,}")
print(f"Embedding Tokens: {chain.embedding_tokens:,}")
```

Output:

```
Cost:            $0.0024
Total Tokens:     16,108
Prompt Tokens:    407
Response Tokens:  97
Embedding Tokens: 15,604
```

Additionally, we can track the history of the chain with the `chain.history` property. See [this notebook](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb) for an example.

---

## Notebooks

- [chains.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/chains.ipynb)
- [openai_chat.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/openai_chat.ipynb)
- [indexes.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/indexes.ipynb)
- [prompt_templates.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/prompt_templates.ipynb)
- [memory.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/memory.ipynb)
- [duckduckgo-web-search.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/duckduckgo-web-search.ipynb)
- [scraping-urls.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/scraping-urls.ipynb)
- [splitting-documents.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/splitting-documents.ipynb)
- [search-stack-overflow.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/search-stack-overflow.ipynb)
- [conversation-between-models.ipynb](https://github.com/shane-kercheval/llm-chain/tree/main/examples/conversation-between-models.ipynb)

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
