<div align="center">

# RAG with LocalAI + LlamaIndex + ChromaDB

</div>

# Overview
In this repository, we create a RAG app that ingests data from a user-specified directory and then answers user's queries (without ingesting on every run).


# Demo
[![Demo](https://img.youtube.com/vi/HBIuzEBiY0A/hqdefault.jpg)](https://www.youtube.com/embed/HBIuzEBiY0A)

## LocalAI

We can use [LocalAI](https://localai.io/) to host multiple models in an OpenAI compatible server.


```bash
# Takes a few minutes!!!
docker pull quay.io/go-skynet/local-ai:latest
```

Also, we can download any [(quantized) model](https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF).

```
docker run -p 8080:8080 -v $PWD:/models -e DEBUG=true -ti --rm quay.io/go-skynet/local-ai:latest --models-path /models --threads 16 --context-size 2048
```

To test if the models can be successfully accessed, we can run the following commands.

```bash
curl http://localhost:8080/embeddings -X POST -H "Content-Type: application/json" -d '{
  "input": "Can you encode this?",
  "model": "text-embedding-ada-002"
}'

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "neural-chat", "messages": [{"role": "user", "content": "How are you?"}], "temperature": 0.9 }'
```

## Ingesting Data

To ingest data, run the following command

```bash
python ingest.py "source" "chroma_db" "course"
```

## Running a Query

To query the indexed data, run the following command.

Something that is addressed in the source documents.


```bash
python rag.py "chroma_db" "course" "What is GGML?"
```

> Response: GGML is a C library for machine learning, developed by Georgi Gerganov. It defines low-level machine learning primitives and a binary format for distributing large language models (LLMs). GGML supports various quantization methods, allowing large language models to run on consumer hardware. It also offers features like automatic differentiation, optimizers, and compatibility with different architectures. GGML files follow the GGUF format, which defines the structure and representation of data within the binary files.


Something that isn't addressed in the source documents.

```bash
python rag.py "chroma_db" "course" "What is MLX?"
```

> Response: Empty Response

Seem's like a good addition for the next iteration :smirk:.


