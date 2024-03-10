# RAG Ingestion Pipeline using Apache Beam

This project introduces an Apache Beam data ingestion pipeline designed for Retrieval-Augmented Generation (RAG) applications. It efficiently handles both document streams and batches to establish a comprehensive search space of embeddings, enabling the retrieval of relevant documents through semantic search. The pipeline covers document loading, content summarization, and content embedding. The embedded data, along with the original content and metadata, is finally stored in a vector database.

Batch ingestion proves beneficial when pre-existing documents need to be searchable by the retriever. Conversely, the streaming ingestion pipeline ensures real-time synchronization of the vector database with newly generated documents, making it suitable for scenarios involving dynamic data sources like live news feeds.

Cohere is the LLM provider of choice for both summarization and content embedding, due to its user-friendly interface, well-documented API, and Python client library. Switching to different models is just a matter of implementing a new ModelHandler.

### Key Technical Features

- **Modular Architecture:** The pipeline employs a modular design, allowing users to seamlessly interchange summarization or embedding models, switch to a different vector database, or use alternative data loaders without disrupting the core functionality.
- **Batch and Stream Processing:** Leveraging Apache Beam, the pipeline seamlessly accommodates both batch and streaming data ingestion. Without necessitating code modifications, it efficiently handles the initial dataset ingestion and integrates into a system for the continuous processing of documents added to the dataset over time.

## But what is RAG?

Retrieval Augmented Generation (RAG) is a technique to improve the accuracy and reliability of generative text models by providing them with relevant information from external sources. The RAG workflow consists of three major steps:

- Upon receiving a request, the RAG application dynamically formulates one or more search queries to identify documents with potential solutions to the presented problem or question. These generated search queries are then utilized to retrieve pertinent documents from external data sources. Semantic search is the most common search technique, though any effective search methodology may be employed.
- With the retrieved relevant documents and the original problem statement, the generator writes a response that addresses the problem, optionally including citations to specific information from the retrieved documents.

RAG enables applications to harness the capabilities of LLMs without the necessity of training or fine-tuning an LLM on (sensitive) data. While RAG is a very powerful technique bringing a lot of buzz around it, it does not offer a one-size-fits-all solution. Prior to implementing an RAG system, a comprehensive analysis of the dataset and the specific problems or questions the RAG is intended to address is crucial. The adage "garbage in, garbage out" also applies to RAGs. Providing the generator with inaccurate or incomplete data will result in suboptimal outcomes, regardless of the extent of prompt engineering applied.

## Overview of the Pipeline

### Loading the Data

This pipeline loads a set of JSON files, which can be considered as responses from a web server stored in a bucket. The files contain the text that needs to be embedded, along with additional metadata, which is also stored in the vector database.

### Summarizing the Content

Summarization serves as a noise reduction mechanism, filtering out irrelevant details and distilling essential information from a document. In the context of vector search, this process ensures that key features and semantic content are captured in a more concise form, leading to accurate and relevant search results.

Text chunking is a common technique to enhance vector search. By breaking down a text document into localized sections or chunks, the model can capture local context within each segment. Embedding these chunks separately enhances the contextual relevance of search results. Treating each chunk as a distinct entity aligns the vector search more closely with user information needs.

In this pipeline, we chose summarization over text chunking. Summarizing a document enhances the incorporation of critical context and details, particularly those mentioned in one specific section of the document, such as the introduction. Choosing the appropriate preprocessing method is highly application-specific, depending on factors such as document length, structure, and content. Careful consideration of these elements is crucial for optimizing the performance of the text summarization and embedding process. Another important consideration is that there's no strict requirement to use the exact chunks or summaries employed for data retrieval in your generator. Opting to pass the entire document or a larger chunk provides additional context, deviating from the specific chunks used during retrieval.

### Embedding the Summarized Content

The embedding process involves the conversion of unstructured textual data into numerical vectors, providing a representation that encapsulates semantic meaning and context. In the context of RAG applications, text embeddings are instrumental for semantic search, aiding the generator in identifying pertinent documents to address posed problems or questions. This involves creating a vector representation for the question through embedding, followed by a search for the nearest embeddings within an embedding space. During this step of the pipeline, we establish the embedding space, comprising relevant documents essential for the generator to respond effectively to inquiries. This is accomplished by leveraging Cohere's embedding model, which seamlessly processes and transforms the summaries into vectors. These vectors serve as robust representations of the semantic content, facilitating subsequent retrieval tasks.

### Ingesting Data into a Weaviate Vector Database

The concluding step of the pipeline involves storing the generated embeddings, along with the summarized text and associated metadata, in a Weaviate vector database. To optimize ingestion performance, we initiate a transformation of the PCollection into batches, thereby minimizing the number of database transactions and enhancing overall pipeline efficiency. This strategic approach ensures an efficient and streamlined process for populating the Weaviate vector database with the relevant information from the RAG data ingestion pipeline.

## Running the Pipeline

Start by cloning the project:

```bash
git clone https://github.com/jaxpr/rag-ingestion-pipeline.git
```

Next, install the Python dependencies. It's recommended to use a virtual environment to manage your project dependencies. Follow these steps to create and activate a virtual environment:

```bash
cd rag-ingestion-pipeline

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

You’re almost ready to run the pipeline; the only thing left is to set up a Weaviate instance to store the results of the pipeline. When setting up locally for development and experimentation, consider using Docker to run the Weaviate instance. Weaviate has excellent [documentation](https://weaviate.io/developers/weaviate/installation/docker-compose) on how to set everything up using docker-compose.

Once your Weaviate instance is running, we first need to create a schema that describes what the data we want to store will look like. Check out the Weaviate [documentation](https://weaviate.io/developers/weaviate/starter-guides/schema) for information on defining schemas.

```bash
import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.connect import ConnectionParams

connection_params = ConnectionParams.from_params(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
)

client = weaviate.WeaviateClient(
    connection_params=connection_params
)

if __name__ == '__main__':
    try:
        client.connect()
        collection = client.collections.create(
            name="Collection",
            properties=[
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="tags",
                    data_type=[DataType.TEXT],
                ),
            ],
            # Configure the vector index
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
            # Configure the inverted index
            inverted_index_config=Configure.inverted_index(
                index_null_state=True,
                index_property_length=True,
                index_timestamps=True,
            ),
        )

    finally:
        client.close()

    try:
        response = client.collections.list_all()
        print(response)
    finally:
        client.close()
```

Make sure your environment variables for API keys are set, and you have configured a datasource.

```bash
export COHERE_API_KEY=<YOUR API KEY>
export WEAVIATE_API_KEY=<YOUR API KEY> # optional
...
```

Now you are ready to run the Beam pipeline.

```
python ingestion_pipeline.py
```

### References

- [Apache Beam Remote Inference Example Pipeline](https://cloud.google.com/dataflow/docs/notebooks/run_inference_vertex_ai)
- [Cohere API Reference Documentation](https://docs.cohere.com/reference/about)
- [Apache Beam Programming Guide](https://beam.apache.org/documentation/programming-guide/)
- [Retrieval Augmented Generation (Cohere)](https://docs.cohere.com/docs/retrieval-augmented-generation-rag)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)

_This README has been written with the help of Cohere’s Generate model._
