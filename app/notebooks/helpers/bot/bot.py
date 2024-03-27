from typing import List
from xml.etree.ElementInclude import include
from llama_index.core import (
    Document,
    ServiceContext,
    KnowledgeGraphIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.graph_stores import SimpleGraphStore

# from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core.memory import ChatMemoryBuffer
from .kg_generation import create_kg_triplets
from notebooks.helpers.models.embedding_model import PredictionModel
from notebooks.helpers.bot.promtps import (
    WINE_KG_PROMPT,
    PAIRING_KEYWORD_EXTRACT,
    CONTEXT_TEMPLATE,
)
import torch
from pathlib import Path

import logging
import sys
from dotenv import get_key
import openai


openai.api_key = get_key("./app/.env", "OPENAI_API_KEY")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

MEMORY = ChatMemoryBuffer.from_defaults(token_limit=5000)


def generate_pairings_documents(instance) -> Document:
    document = Document(
        text=instance["triplets"],
        metadata=(
            instance["meta_data"]
            if type(instance["meta_data"]) != str
            else eval(instance["meta_data"])
        ),
        metadata_seperator="\n",
        metadata_template="{key}=>{value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
    )
    return document


def extract_triplets_from_document(text):
    triplets = []
    triplets = text.split("Content:")[1].strip().split("\n")[:-1]
    triplets_list = []
    for triplet in triplets:
        triplet = triplet.split("**")
        if len(triplet) >= 6:
            triplets_list.append(
                (triplet[1].strip(), triplet[3].strip(), triplet[5].strip())
            )

    return triplets_list


def load_llm(llm_name: str):
    if llm_name == "ollama":
        llm = Ollama(
            model="llama2", temperature=0, request_timeout=500.0, context_window=1536
        )

    elif llm_name == "openai4":
        llm = OpenAI(
            model="gpt-4",
            temperature=0.0,
            max_tokens=1536,
        )
    elif llm_name == "openai3.5":
        llm = OpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.0,
            max_tokens=2000,
            context_window=1536,
        )

    elif llm_name == "gpt2":
        llm = HuggingFaceLLM(
            context_window=2056,
            max_new_tokens=128,
            generate_kwargs={"temperature": 0.25, "do_sample": False},
            tokenizer_name="openai-community/gpt2",
            model_name="openai-community/gpt2",
            # model = model
            device_map="auto",
            tokenizer_kwargs={"max_length": 2048},
            # uncomment this if using CUDA to reduce memory usage
            model_kwargs={"torch_dtype": torch.float16},
        )

    return llm


def load_embedding_model(embedding_model_name: str):
    if embedding_model_name == "foodbert":
        PM = PredictionModel()
        return HuggingFaceEmbedding(
            model=PM.model,
            tokenizer=PM.tokenizer,
            device=PM.device,
        )
    elif embedding_model_name == "openai3":
        return OpenAIEmbedding(
            model="text-embedding-3-small",
            dimensions=768,
        )


def service(chunk_size=384, llm=None, embed_model=None):
    return ServiceContext.from_defaults(
        chunk_size=chunk_size,
        llm=llm,
        embed_model=embed_model,
        system_prompt=(
            "You are a master sommelier with extent knowledge of wine and food pairings. Therefore, you can help by answering wine and food related questions."
        ),
    )


def setup_index_and_storage(
    service,
    kg_pairings=None,
    storage_path="./app/data/graph_storage/",
    max_triplets_chunk=10,
    max_object_length=128,
    manually_construct_kg=True,
    show_progress=False,
    force=False,
):
    edge_types = ["relationship"]
    tags = ["entity"]
    storage_path = Path(storage_path)
    graph_store = SimpleGraphStore()

    if list(storage_path.iterdir()) and not force:
        storage_context = StorageContext.from_defaults(
            persist_dir=storage_path, graph_store=graph_store
        )

        kg_index = load_index_from_storage(
            storage_context,
            max_triplets_per_chunk=max_triplets_chunk,
            service_context=service,
            include_embeddings=True,
            show_progress=True,
            edge_types=edge_types,
            tags=tags,
            max_object_length=max_object_length,
            kg_triplet_extract_fn=(
                extract_triplets_from_document if manually_construct_kg else None
            ),
        )

    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=None, graph_store=graph_store
        )

        kg_index = KnowledgeGraphIndex.from_documents(
            documents=kg_pairings,
            kg_triple_extract_template=WINE_KG_PROMPT,
            max_triplets_per_chunk=max_triplets_chunk,
            service_context=service,
            storage_context=storage_context,
            include_embeddings=True,
            show_progress=show_progress,
            edge_types=edge_types,
            tags=tags,
            max_object_length=max_object_length,
            kg_triplet_extract_fn=(
                extract_triplets_from_document if manually_construct_kg else None
            ),
        )
        kg_index.storage_context.persist(persist_dir=storage_path)

    return storage_context, kg_index


def get_chat_engine(
    kg_index,
    response_mode="compact",
    retriver_mode="hybrid",
    chat_mode="context",
    use_global_node_triplets=True,
    max_keywords_per_query=10,
    num_chunks_per_query=10,
    similarity_top_k=3,
    graph_store_query_depth=3,
    memory=MEMORY,
):

    chat_engine = kg_index.as_chat_engine(
        chat_mode=chat_mode,
        memory=memory,
        retriever_mode=retriver_mode,
        response_mode=response_mode,
        verbose=True,
        include_text=True,
        max_keywords_per_query=max_keywords_per_query,
        num_chunks_per_query=num_chunks_per_query,
        similarity_top_k=similarity_top_k,
        graph_store_query_depth=graph_store_query_depth,
        context_template=CONTEXT_TEMPLATE,
        use_global_node_triplets=use_global_node_triplets,
        kg_triple_extract_template=PAIRING_KEYWORD_EXTRACT,
    )

    return chat_engine


def get_query_engine(
    kg_index,
    response_mode="compact",
    retriver_mode="hybrid",
    chat_mode="context",
    use_global_node_triplets=True,
    max_keywords_per_query=10,
    num_chunks_per_query=10,
    similarity_top_k=3,
    graph_store_query_depth=3,
    include_text=True,
):

    query_engine = kg_index.as_query_engine(
        chat_mode=chat_mode,
        memory=MEMORY,
        retriever_mode=retriver_mode,
        response_mode=response_mode,
        verbose=True,
        include_text=include_text,
        max_keywords_per_query=max_keywords_per_query,
        num_chunks_per_query=num_chunks_per_query,
        similarity_top_k=similarity_top_k,
        graph_store_query_depth=graph_store_query_depth,
        context_template=CONTEXT_TEMPLATE,
        use_global_node_triplets=use_global_node_triplets,
        kg_triple_extract_template=PAIRING_KEYWORD_EXTRACT,
    )

    return query_engine


def get_simple_query(
    response_mode="compact",
    retriver_mode="hybrid",
    chat_mode="context",
    include_text=False,
):
    llm = load_llm("openai3.5")
    embed_model = load_embedding_model("openai3")
    service_context = service(llm=llm, embed_model=embed_model)

    kg_index = KnowledgeGraphIndex.from_documents(
        documents=[
            Document(
                text="",
            )
        ],
        service_context=service_context,
        include_embeddings=True,
    )

    return kg_index.as_query_engine(
        chat_mode=chat_mode,
        memory=MEMORY,
        retriever_mode=retriver_mode,
        response_mode=response_mode,
        verbose=True,
        include_text=include_text,
        context_template=CONTEXT_TEMPLATE,
        kg_triple_extract_template=PAIRING_KEYWORD_EXTRACT,
    )


def main():
    pass


if __name__ == "__main__":
    main()
