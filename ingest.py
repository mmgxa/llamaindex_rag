import os
import typer
import chromadb

from llama_index.vector_stores import ChromaVectorStore

from llama_index import (
    SimpleDirectoryReader,
    PromptHelper,
    ServiceContext,
    VectorStoreIndex,
    download_loader,
)
from llama_index.llms import OpenAILike
from llama_index.storage.storage_context import StorageContext


# Import Typer and create a Typer app instance
import typer

app = typer.Typer()


os.environ["OPENAI_API_KEY"] = "NONE"
os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"


@app.command()
def ingest(source: str, db_path: str, db_name: str):
    # models
    embed_predictor = OpenAILike(model_name="text-embedding-ada-002")

    context_window = 2048
    num_output = 400
    chunk_overlap_ratio = 0.2
    separator = "\n"

    prompt_helper = PromptHelper(
        context_window=context_window,
        num_output=num_output,
        chunk_overlap_ratio=chunk_overlap_ratio,
        separator=separator,
    )

    UnstructuredReader = download_loader("UnstructuredReader")

    dir_reader = SimpleDirectoryReader(
        source,
        file_extractor={
            ".html": UnstructuredReader(),
        },
    )

    # Load documents from the specified directory
    documents = dir_reader.load_data()

    # db = chromadb.PersistentClient(path="./chroma_db")
    # chroma_collection = db.get_or_create_collection("course")

    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(db_name)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create a VectorStoreIndex with the specified parameters
    service_context_index = ServiceContext.from_defaults(
        llm=embed_predictor, prompt_helper=prompt_helper, chunk_size_limit=512
    )

    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context_index,
        storage_context=storage_context,
    )

    typer.echo(f"Index created and stored in {db_path}")


if __name__ == "__main__":
    app()
