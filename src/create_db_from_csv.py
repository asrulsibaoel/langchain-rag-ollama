import uuid
import argparse
import pandas as pd
from langchain.schema import Document

from vectorstore_factory import get_vectorstore
from settings import settings


def create_documents_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    assert all(col in df.columns for col in ['text', 'source', 'category']), \
        "CSV must contain 'text', 'source', and 'category' columns."

    docs = []
    ids = []
    chunks = 0
    data_length = len(df)
    print(f"Processing {data_length} rows from CSV...")
    for _, row in df.iterrows():
        try:
            _id = str(uuid.uuid4())
            doc = Document(
                page_content=row['text'],
                metadata={
                    "id": _id,
                    "source": str(row['source']),
                    "category": str(row['category'])
                }
            )
            docs.append(doc)
            ids.append(_id)
            chunks += 1
            if chunks % 100 == 0:
                print(f"Processed {chunks}/{data_length} chunks...")
        except Exception as e:
            print(f"Error processing row {row['text']}: {e}")
            continue
    return docs, ids


def main(csv_path: str):
    print(f"Loading CSV from: {csv_path}")
    documents, ids = create_documents_from_csv(csv_path)

    vectorstore = get_vectorstore(backend=settings.vector_db)
    vectorstore.add_documents(documents, ids=ids)

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    print("Documents successfully added to vectorstore.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    args = parser.parse_args()
    main(args.csv_path)
