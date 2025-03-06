from abc import ABC, abstractmethod
from langchain_core.documents import Document

class BaseSearch(ABC):
    """Interface for Document search."""

    @abstractmethod
    def base_search(self, query: str) -> list[Document]:
        """search docs."""
        pass
