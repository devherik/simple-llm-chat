from langchain_core.documents.base import Document
from typing import List


def clean_metadata(documents: List[Document]) -> List[Document]:
        """Clean metadata to ensure compatibility with Chroma vector store."""
        cleaned_docs = []
        for doc in documents:
            cleaned_metadata = {}
            if doc.metadata:
                for key, value in doc.metadata.items():
                    # Convert complex objects to strings or skip them
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        cleaned_metadata[key] = value
                    elif isinstance(value, dict):
                        # Flatten dictionary or convert to string
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                                cleaned_metadata[f"{key}_{sub_key}"] = sub_value
                            else:
                                cleaned_metadata[f"{key}_{sub_key}"] = str(sub_value)
                    elif isinstance(value, list):
                        # Convert list to comma-separated string
                        cleaned_metadata[key] = ", ".join(str(item) for item in value)
                    else:
                        # Convert other types to string
                        cleaned_metadata[key] = str(value)
            
            cleaned_docs.append(Document(
                page_content=doc.page_content,
                metadata=cleaned_metadata
            ))
        return cleaned_docs