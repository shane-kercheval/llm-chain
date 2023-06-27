"""
A prompt_template is a callable object that takes a prompt (e.g. user query) as input and returns a
modified prompt. Each prompt_template is given the information it needs when it is instantiated.
So for example, if a template's job is to search for relevant documents, it's provided the vector
database when the object is created (not via __call__).
"""
from llm_chain.base import DocumentIndex, EmbeddingsRecord, PromptTemplate
from llm_chain.resources import PROMPT_TEMLATE__DOC_SEARCH_STUFF


class DocSearchTemplate(PromptTemplate):
    """
    DocSearchTemplate is a prompt-template that, based on the prompt provided when the object is
    called (__call__), looks up the most similar documents via the `doc_index` provided and
    includes all of the documents (i.e. the underlying content) in the prompt.
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
        self.template = template if template else PROMPT_TEMLATE__DOC_SEARCH_STUFF
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
    def history(self) -> list[EmbeddingsRecord]:
        """TODO."""
        return self._doc_index.history
