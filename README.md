# llm-chain package

Simple and extensible LLM Chaining (pre-alpha)

**NOTE: This package is tested on Python `3.10` and `3.11`**

- package source in `/llm_chain`
- unit tests in `/tests`

---

## TODO

- implement PDFLoadder
- implement vectordb store
- implement DocumentPrompt (stuff docs + query)
- implement async
- implement retry

---


## Installing

```commandline
pip install llm-chain
```

---

## Usage

The intent of this package is to make interacting with LLMs easy and extensible. To that end, the main philosophy and of building chains is that any individual link (e.g. Prompt, Model, Tool, other any other class/function that is callable) can be chained together as long as the output of one link matches the input of the next link.


Should chains be reused? Yes?
But i don't like that you'd have to define 

```python
# 

>> chain = Chain([ConversationPrompt(), OpenAIChat()])
>> response = chain("Hello chatbot.")
>> print(response)

'Hello ... '
```

```
>> chat = OpenAIChat()
>> chain = Chain([ConversationPrompt(), chat])
>> response = chain("Hello chatbot.")

chat.history()
```

The message `Hello chatbot.` is given to the first link in the chain and is propagated according to the logic of types of links in the chain.



Text -> Documents -> Chunks -> Embeddings -> VectorDB-store


DocumentsPrompt()(docs, query)

# stuff all the documents into the prompt
DocumentsStuffPrompt()(docs, query)


query/text -> Embeddings -> VectorDB-search -> Chunks -> (docs + query)??? -> Chat -> response/text
hmm.. does chain store initial input and assume it is a query for subsequent calls



```python
chat = OpenAIChat()

chain = Chain([
    PDFLoader('./path/to/pdf_file.pdf'),  # loads the pdf into memory and extracts the text; returns a list of Documents
    TextSplitter(chunk_size=500),  # splits the text into chunks, returns a list of Documents
    DocumentSummaryPrompt(),  # Expects a list of documents to summarize. Returns a prompt
    OpenAIChat(),  # 
])
response = chain()
print(response)   #  'This is a summary of the pdf bla'

print(chat.last_prompt)  # "Summarize the following PDF"
print(chat.last_reponse)
print(chat.history)   # this is not just strings... this needs to be like streamlit app where i capture prompt and answer in message with corersponding costs and information about that single interaction
```

chain.total_cost  # 
chain.cost_per_model  # per model
chain.total_tokens_per_model  # per model.. makes sense to aggregate costs across models, but doesn't make sense to aggregate tokens across models


chain.cost_per_run  # # per model e.g. embeddings model .. chat model  # merges keys together
chain.cost_breakdown  # per model e.g. embeddings model .. chat model  # merges keys together


```
# set up vector database; can be used as a chain or called individually, but chain gives ability to track costs if there are multiple steps

chat = OpenAIChat()
vector_db = Chroma()

embedding_chain = Chain([
    
    PDFLoader('./path/to/pdf_file.pdf'),  # loads the pdf into memory and extracts the text; returns a list of Documents
    TextSplitter(chunk_size=500),  # splits the text into chunks, returns a list of Documents
    OpenAIEmbeddingsModel(),  # Expects a list of documents to embed (or str or single doc?); returns Embedding for each doc
    vector_db.save,
])
_ = embedding_chain()
embedding_chain.total_cost()

embedding_chain.info()  ?? summarize information like file-paths, model, names




qa_chain = Chain([
    vector_db.similarity,  # figure out how to pass embeddings or not ; need to return Docs
    DocumentSummaryPrompt(), ##
    OpenAIChat(),
])

chain("This is a question about the PDF")

```




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
