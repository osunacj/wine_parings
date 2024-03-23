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
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def generate_pairings_documents(instance) -> Document:
    document = Document(
        text=instance["triplets"],
        metadata=(
            instance["meta_data"]
            if type(instance["meta_data"]) != str
            else eval(instance["meta_data"])
        ),
        metadata_seperator="::",
        metadata_template="{key}=>{value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
    )
    return document


def extract_triplets_from_document(text):
    triplets = []
    triplets = text.split("Content:")[1].strip().split("\n")
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


def load_embedding_model(embedding_model_name: str) -> HuggingFaceEmbedding:
    if embedding_model_name == "foodbert":
        PM = PredictionModel()
        return HuggingFaceEmbedding(
            model=PM.model,
            tokenizer=PM.tokenizer,
            device=PM.device,
        )
    elif embedding_model_name == "ollama":
        return Ollama(
            model="llama2", temperature=0, request_timeout=500.0, context_window=1536
        )


def service(chunk_size=256, llm=None, embed_model=None):
    return ServiceContext.from_defaults(
        chunk_size=chunk_size,
        llm=llm,
        embed_model=embed_model,
        system_prompt=(
            "You are a master sommelier with extent knowledge of wine, food and their best pairings. Therefore, you will help by recommending wine or food."
        ),
    )


def setup_index_and_storage(
    service,
    kg_pairings=None,
    storage_path="./app/data/graph_storage/",
    max_triplets_chunk=10,
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
            max_object_length=256,
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
            max_object_length=256,
            kg_triplet_extract_fn=(
                extract_triplets_from_document if manually_construct_kg else None
            ),
        )
        kg_index.storage_context.persist(persist_dir=storage_path)

    return storage_context, kg_index


def get_chat_engine(
    kg_index,
    response_mode="tree_summarize",
    retriver_mode="hybrid",
    chat_mode="context",
    use_global_node_triplets=True,
    max_keywords_per_query=10,
    num_chunks_per_query=15,
    similarity_top_k=2,
    graph_store_query_depth=2,
    token_limit=5000,
):
    memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)

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


def main():

    KG = create_kg_triplets()
    kg_pairings = KG.apply(generate_pairings_documents, axis=1)

    llm = load_llm("ollama")
    embed_model = load_embedding_model("foodbert")
    service_context = service(llm=llm, embed_model=embed_model)

    storage_context, kg_index = setup_index_and_storage(
        service=service_context,
        kg_pairings=kg_pairings,
        show_progress=False,
        force=True,
    )

    # query_str = "For dinner, I am cooking a corn risotto, can you recommend a wine?"

    # chat_engine = get_chat_engine(
    #     kg_index, chat_mode="context", retriver_mode="keyword"
    # )
    # response = chat_engine.chat(query_str)

    # print(response)

    # response = chat_engine.chat(
    #     "But if I do not like white wines, what other wines would you recommend based on my dinner?",
    # )

    # print(response)

    # evaluator = FaithfulnessEvaluator(llm=llm)

    # # query_engine = kg_index.as_query_engine()
    # query = "What type of wine is Chardonnay?"
    # response = chat_engine.chat(query)
    # print(response)
    # eval_result = evaluator.evaluate_response(response=response)
    # print(str(eval_result.passing))


if __name__ == "__main__":
    main()
