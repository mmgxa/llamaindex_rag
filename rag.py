import os
import typer
import chromadb

from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

from llama_index import (
    PromptHelper,
    ServiceContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.llms import OpenAILike
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor


# Import Typer and create a Typer app instance
import typer

app = typer.Typer()


os.environ["OPENAI_API_KEY"] = "NONE"
os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"


@app.command()
def rag_app(db_path: str, db_name: str, question: str):
    # models
    chat_predictor = OpenAILike(model="neural-chat")

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

    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(db_name)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1,
    )

    service_context_response = ServiceContext.from_defaults(
        llm=chat_predictor, prompt_helper=prompt_helper
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        service_context=service_context_response
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    response = query_engine.query(question)

    # print(response.response)
    typer.echo(f"Response: {response.response}")


if __name__ == "__main__":
    app()
