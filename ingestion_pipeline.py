import json
import os
from typing import NamedTuple, Optional, Dict, Iterable, Sequence, Any, List

import cohere
import weaviate

import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.ml.inference.base import ModelHandler, PredictionResult, RunInference
from cohere import CohereError
from weaviate.connect import ConnectionParams

COHERE_API_KEY = os.environ.get('COHERE_API_KEY')


class EmbeddingConfig(NamedTuple):
    """
    NamedTuple to store optional parameters for Cohere embedding models.
    Check the Cohere API reference for more information:
    https://docs.cohere.com/reference/embed
    """
    model: Optional[str] = None
    input_type: Optional[str] = None
    truncate: Optional[str] = None


class SummarizationConfig(NamedTuple):
    """
    NamedTuple to store optional parameters for Cohere summarization models.
    Check the Cohere API reference for more information:
    https://docs.cohere.com/reference/summarize-2
    """
    length: Optional[str] = None
    format: Optional[str] = None
    model: Optional[str] = None
    extractiveness: Optional[str] = None
    temperature: Optional[str] = None
    additional_command: Optional[str] = None


class ExtractKeysSubset(beam.DoFn):
    def __init__(self, keys: List[str]):
        """A DoFn for extracting a subset of string keys from each dictionary in a PCollection.

        Args:
            keys (List[str]): Names of the keys that need to be retained in the resulting dictionaries.
        """
        super().__init__()
        self.keys = keys

    def process(self, element: Dict[str, Any], **kwargs):
        yield {key: element[key] for key in self.keys}


class CohereModelHandler(ModelHandler[Dict, PredictionResult, cohere.Client]):
    def __init__(self, api_key: str, input_key: str):
        """ ModelHandler interface for Cohere models.

        Example Usage::

          pcoll | RunInference(SpacyModelHandler())

        Args:
          model_name: The spaCy model name. Default is en_core_web_sm.
        """
        self.api_key = api_key
        self.input_key = input_key

    def load_model(self) -> cohere.Client:
        """Loads and initializes a model for processing."""
        return cohere.Client(api_key=self.api_key)


class CohereSummarizationModelHandler(CohereModelHandler):
    def __init__(self, api_key: str, input_key: str, summarization_config: SummarizationConfig):
        """ Implementation of CohereModelHandler interface for summarization.

        Example Usage::

          pcoll | RunInference(CohereSummarizationModelHandler(summarization_config))

        Args:
          input_key: The key in the input dictionaries that maps to the text that will be summarized.
          summarization_config: Configuration containing optional inference parameters.
        """
        super().__init__(api_key, input_key)
        # Create a dictionary that maps all parameters in the config to their values
        self.config = {key: value for key, value in summarization_config._asdict().items() if value is not None}

    def run_inference(self, batch: Sequence[Dict[str, Any]], model: cohere.Client,
                      inference_args: Optional[Dict[str, Any]] = None) -> Iterable[PredictionResult]:
        inference_args = {} if not inference_args else inference_args
        summaries = []
        for element in batch:
            text = element['text']
            # The Cohere summarization models do not support inputs smaller than 250 character
            if len(text) > 250:
                # Send the text to the summarization model, along with optionally configured parameters
                summary_response = model.summarize(text=text, **self.config)
                # Extract the summary from the response returned by the API
                summary = summary_response.summary
            else:
                # Texts shorter than 250 characters aren't summarized
                summary = text
            summaries.append(summary)

        # Add the summaries to the output dictionaries along the other blogs
        updated_list_of_dicts = [{**element, 'summary': summary} for element, summary in zip(batch, summaries)]

        # Return the output dictionaries as a batch of PredictionResult objects
        return [PredictionResult(x, y) for x, y in zip(batch, updated_list_of_dicts)]


class CohereEmbeddingModelHandler(CohereModelHandler):
    def __init__(self, api_key: str, input_key: str, embedding_config: EmbeddingConfig):
        """Implementation of CohereModelHandler interface for embedding.

        Example Usage::

          pcoll | RunInference(CohereEmbeddingModelHandler(embedding_config))

        Args:
          input_key: The key in the input dictionaries that maps to the text that will be summarized.
          embedding_config: Configuration containing optional inference parameters.
        """
        super().__init__(api_key, input_key)
        # Create a dictionary that maps all parameters in the config to their values
        self.config = {key: value for key, value in embedding_config._asdict().items() if value is not None}

    def run_inference(self, batch: Sequence[Dict[str, Any]], model: cohere.Client,
                      inference_args: Optional[Dict[str, Any]] = None) -> Iterable[PredictionResult]:
        inference_args = {} if not inference_args else inference_args
        # Create a list of inputs that will be sent to the embedding model
        texts = [element[self.input_key] for element in batch]

        # Send the text to the embedding model, along with optionally configured parameters
        response = model.embed(texts=texts, **self.config)

        # Extract the embeddings from the response returned by the API
        embeddings = response.embeddings

        # Return a list of PredictionResult
        return [
            PredictionResult(example=element, inference=embedding)
            for element, embedding
            in zip(batch, embeddings)
        ]


class StoreWeaviate(beam.DoFn):
    def __init__(self, connection_params: ConnectionParams, collection: str):
        """
        DoFn to store a Pcollection of PredictionResults into a Weaviate vector database. For efficiency reasons
        the PredictionResult objects should be batched together first.

        Example Usage::

          connection_params = ConnectionParams(...)
          collection = "<collection name>"

          pcoll
          | 'embed content' >> RunInference(...)
          | 'batch prediction results' >> beam.BatchElements(...)
          | 'store embeddings' >> StoreWeaviate(connection_params, collection)

        Args:
            connection_params: ConnectionParams object containing all data to connect to a Weaviate vector database
            collection: Name of the Weaviate collection where the data will be inserted

        """
        super().__init__()
        self.weaviate_client = weaviate.WeaviateClient(
            connection_params=connection_params
        )
        self.collection = collection

    def start_bundle(self):
        self.weaviate_client.connect()

    def finish_bundle(self):
        self.weaviate_client.close()

    def process(self, prediction_results: List[PredictionResult], **kwargs):
        collection = self.weaviate_client.collections.get(self.collection)

        with collection.batch.dynamic() as batch:
            for prediction_result in prediction_results:
                batch.add_object(
                    properties=prediction_result.example,
                    vector=prediction_result.inference
                )


if __name__ == '__main__':
    embed_config = EmbeddingConfig()
    summarize_config = SummarizationConfig()
    weaviate_connection_params = ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )
    weaviate_collection = "ExampleCollection"

    summary_model_handler = CohereSummarizationModelHandler(
        api_key=COHERE_API_KEY,
        input_key='text',
        summarization_config=summarize_config
    )

    embedding_model_handler = CohereEmbeddingModelHandler(
        api_key=COHERE_API_KEY,
        input_key='summary',
        embedding_config=embed_config
    )

    with beam.Pipeline() as pipeline:
        # Load all json files in the data directory
        json_files = (
                pipeline
                | fileio.MatchFiles('../data/*.json')
                | fileio.ReadMatches()
                | beam.Reshuffle()
                | beam.Map(lambda x: json.load(x))
        )

        documents, invalid_documents = (
                json_files
                | 'Parse documents' >> beam.ParDo(ExtractKeysSubset(keys=['title', 'content', 'metadata']))
                .with_exception_handling(main_tag='documents', dead_letter_tag='invalid_documents', exc_class=KeyError)
        )

        summary_prediction_results, summaries_dead_letter_queue = (
                documents
                | 'Summarize documents' >> RunInference(summary_model_handler)
                .with_exception_handling(exc_class=CohereError)
        )

        summaries = (
                summary_prediction_results
                | 'Extract summaries' >> beam.Map(lambda prediction_result: prediction_result.inference)
        )

        embedding_prediction_results, embeddings_dead_letter_queue = (
                summaries
                | 'Embed summarized documents' >> RunInference(embedding_model_handler)
                .with_exception_handling(exc_class=CohereError)
        )

        _ = (
                embedding_prediction_results
                | "Batch prediction results" >> beam.BatchElements(min_batch_size=64, max_batch_size=128)
                | 'Store results' >> beam.ParDo(StoreWeaviate(weaviate_connection_params, weaviate_collection))
        )
