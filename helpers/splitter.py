from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from typing import List


def splitter_documents(documents, chunk_size=1000, chunk_overlap=200) -> List[Document]:
    """
    Splits a list of documents into smaller chunks based on the specified size and overlap.

    Args:
        documents (list): List of document strings to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters that should overlap between chunks.

    Returns:
        list: A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    # Convert documents to Document objects if they are not already
    if not all(isinstance(doc, Document) for doc in documents):
        documents = [Document(page_content=doc) for doc in documents]

    splits = text_splitter.split_documents(documents)
    return splits
