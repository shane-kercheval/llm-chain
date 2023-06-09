"""
A "prompt-template" is a callable object that takes a prompt (e.g. user query) as input and returns
a modified prompt. Each prompt-template class is instantiated with the necessary information it
requires. For instance, if a template's purpose is to search for relevant documents, it is provided
with the vector database during object creation rather than through the `__call__` method.

If you're prompt-template is simple, just use a function (or inline lambda) in the link.
"""
from llm_chain.base import DocumentIndex, Record, PromptTemplate
from llm_chain.resources import PROMPT_TEMPLATE__INCLUDE_DOCUMENTS


class DocSearchTemplate(PromptTemplate):
    """
    `DocSearchTemplate` is a prompt-template that, when called (`__call__`) with a prompt, searches
    for the most similar documents using the provided `DocumentIndex` object. It then includes the
    content of all of the retrieved documents in the modified prompt. The Document objects included
    in the prompt can be retrieved via the `similar_docs` property.
    """

    def __init__(
            self,
            doc_index: DocumentIndex,
            template: str | None = None,
            n_docs: int = 3) -> None:
        """
        Args:
            doc_index:
                the document index used to search for relevant documents
            template:
                custom template (string value that must contain "{{documents}}" and "{{prompt}}"
                within the string); if None, then a default template is provided
            n_docs:
                the number of documents (returned by the doc_index) to include in the prompt
        """  # noqa
        super().__init__()
        self._doc_index = doc_index
        self.n_docs = n_docs
        self.template = template if template else PROMPT_TEMPLATE__INCLUDE_DOCUMENTS
        self.similar_docs = None

    def __call__(self, prompt: str) -> str:  # noqa
        super().__call__(prompt)
        self.similar_docs = self._doc_index.search(
            value=prompt,
            n_results=self.n_docs,
        )
        doc_string = '\n\n'.join([x.content for x in self.similar_docs])
        return self.template.\
            replace('{{documents}}', doc_string).\
            replace('{{prompt}}', prompt)

    @property
    def history(self) -> list[Record]:
        """Propagate the history from the underlying DocumentIndex object."""
        return self._doc_index.history

    @property
    def total_tokens(self) -> int | None:
        """
        Sums the `total_tokens` values across all Record objects (which contain that property)
        returned by this object's `history` property (e.g. underlying records from DocumentIndex
        objects; e.g. Embeddings).
        """
        return self.calculate_historical(name='total_tokens')

    @property
    def cost(self) -> float | None:
        """
        Sums the `cost` values across all Record objects (which contain that property)
        returned by this object's `history` property (e.g. underlying records from DocumentIndex
        objects; e.g. Embeddings).
        """
        return self.calculate_historical(name='cost')
