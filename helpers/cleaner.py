from agno.document.base import Document
from typing import List


def clean_metadata(documents: List[Document]) -> List[Document]:
    """
    Clean metadata to ensure compatibility with Chroma vector store.
    This version specifically handles Notion's date format.
    """
    cleaned_docs = []
    for doc in documents:
        cleaned_metadata = {}
        if doc.meta_data:
            for key, value in doc.meta_data.items():
                # Keep simple types as they are
                if isinstance(value, (str, int, float, bool)) or value is None:
                    cleaned_metadata[key] = value
                # Specifically handle Notion's date dictionary
                elif isinstance(value, dict) and ('start' in value or 'end' in value):
                    start = value.get('start', 'None')
                    end = value.get('end', 'None')
                    cleaned_metadata[key] = f"start: {start}, end: {end}"
                # Flatten other dictionaries
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                            cleaned_metadata[f"{key}_{sub_key}"] = sub_value
                        else:
                            cleaned_metadata[f"{key}_{sub_key}"] = str(sub_value)
                # Convert lists to a comma-separated string
                elif isinstance(value, list):
                    cleaned_metadata[key] = ", ".join(str(item) for item in value)
                # Convert any other type to its string representation
                else:
                    cleaned_metadata[key] = str(value)
        
        cleaned_docs.append(Document(
            content=doc.content,
            meta_data=cleaned_metadata
        ))
    return cleaned_docs