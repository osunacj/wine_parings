import json
from pathlib import Path
from xml.etree.ElementInclude import include
from h11 import Data
import pandas as pd
import numpy as np

import nest_asyncio

from llama_index.core import Response
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    DatasetGenerator,
    RelevancyEvaluator,
    EvaluationResult,
    CorrectnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    BatchEvalRunner,
    RetrieverEvaluator,
    SemanticSimilarityEvaluator,
    QueryResponseDataset,
)

from data.evaluation.evaluation import evaluation_qa_pairings
from notebooks.helpers.bot.kg_generation import create_kg_triplets
from notebooks.helpers.bot.promtps import question_gen_query
from notebooks.helpers.bot.bot import (
    get_chat_engine,
    get_query_engine,
    load_llm,
    load_embedding_model,
    setup_index_and_storage,
    generate_pairings_documents,
    service,
)


nest_asyncio.apply()


def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing or result.score >= 0.85:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


def prepare_evalution_qa() -> pd.DataFrame:
    eval_dataset_path = Path("./app/data/evaluation/evaluation.csv")

    if eval_dataset_path.exists():
        eval_dataset = pd.read_csv(eval_dataset_path, index_col="Unnamed: 0")
    else:
        eval_questions, eval_answers = [], []
        for instance in evaluation_qa_pairings:
            eval_questions.append(instance["QUESTION"])
            eval_answers.append(instance["ANSWER"])
        eval_dataset = pd.DataFrame(
            {"questions": eval_questions, "responses": eval_answers}
        )
        eval_dataset.to_csv(eval_dataset_path)

    return eval_dataset


def get_qr_pairs():
    eval_dataset = QueryResponseDataset.from_json(
        "./app/data/evaluation/evaluation_dataset.json"
    )
    questions, responses = [], []

    for item in eval_dataset.qr_pairs:
        questions.append(item[0])
        responses.append(item[1])

    return questions, responses


def parse_evalutions(
    eval_results,
    model,
    embedding_model,
    response_mode,
    retriever_mode,
    chat_mode,  # Irrelevant if the response are from QueryEngine
    queries,
    responses,
):
    evaluation_results_path = Path(
        f"./app/data/evaluation/evaluation_results_{retriever_mode}.json"
    )
    eval_results_dict = {}

    for eval_method in eval_results.keys():
        eval_results_dict[eval_method] = {
            "score": get_eval_results(eval_method, eval_results),
            "model": model,
            "embed_model": embedding_model,
            "response_mode": response_mode,
            "retriever_mode": retriever_mode,
            "chat_mode": chat_mode,
            "data": [],
        }
        for index, response in enumerate(eval_results[eval_method]):
            # Response is of type EvaluationResults
            response_data = {
                "query": queries[index],
                "true_answer": responses[index],
                "score": response.score,
                "passing": response.passing,
                "generated_response": response.response,
                "retrieved_context": response.contexts,
            }

            eval_results_dict[eval_method]["data"].append(response_data)

    with open(evaluation_results_path, mode="w") as file:
        json.dump(eval_results_dict, file)


def main():
    KG = create_kg_triplets(sample_size=600)
    kg_pairings = KG.apply(generate_pairings_documents, axis=1)

    llm = load_llm("openai3.5")
    embed_model = load_embedding_model("openai3")
    service_context = service(llm=llm, embed_model=embed_model)

    storage_context, kg_index = setup_index_and_storage(
        service=service_context,
        kg_pairings=None,
        show_progress=False,
        force=False,
    )

    query_engine = get_query_engine(
        kg_index,
        chat_mode="context",
        retriver_mode="hybrid",
        response_mode="compact",
        use_global_node_triplets=True,
        max_keywords_per_query=10,
        num_chunks_per_query=10,
        similarity_top_k=3,
        graph_store_query_depth=3,
        include_text=False,
    )

    metrics = ["mrr", "hit_rate"]

    relevancy_eval = RelevancyEvaluator(service_context=service_context)
    faithfulness_eval = FaithfulnessEvaluator(service_context=service_context)
    semantic_eval = SemanticSimilarityEvaluator(service_context=service_context)
    answer_eval = AnswerRelevancyEvaluator(service_context=service_context)
    context_eval = ContextRelevancyEvaluator(service_context=service_context)
    correctness_eval = CorrectnessEvaluator(service_context=service_context)
    retriever_eval = RetrieverEvaluator.from_metric_names(
        metrics,
        service_context=service_context,
        retriever=kg_index.as_retriever(retriever_mode="hybrid"),
    )

    runner = BatchEvalRunner(
        {
            "faithfulness": faithfulness_eval,
            "relevancy": relevancy_eval,
            "answer_relevancy": answer_eval,
            "context_relevancy": context_eval,
            "semantic": semantic_eval,
            # "correctness": correctness_eval,
        },
        workers=8,
    )
    queries, responses = get_qr_pairs()
    eval_results = runner.evaluate_queries(
        query_engine,
        queries=queries,
        reference=responses,  # type: ignore
        # eval_kwargs_lists={
        # {
        #   "correctness":  "reference": eval_dataset["responses"].to_list(),
        # }
        # },
    )

    parse_evalutions(
        eval_results=eval_results,
        model="gpt-3.5",
        embedding_model="gpt-3.5",
        response_mode="compact",
        retriever_mode="hybrid",
        chat_mode="simple",  # Irrelevant if the response are from QueryEngine
        queries=queries,
        responses=responses,
    )

    # response = chat_engine.chat(
    #     "What are the primary grape varieties used in producing Bordeaux wines?",
    # )

    # print(response)

    # eval_dataset_path = Path("./app/data/evaluation/evaluation.csv")
    # if eval_dataset_path.exists():
    #     eval_dataset = pd.read_csv(eval_dataset_path, index_col="Unnamed: 0")
    # else:
    #     data_generator = DatasetGenerator.from_documents(
    #         kg_pairings,
    #         service_context=service_context,
    #         question_gen_query=question_gen_query,
    #         num_questions_per_chunk=2,
    #     )

    #     eval_dataset = data_generator.generate_dataset_from_nodes()

    #     pd.DataFrame(
    #         {"questions": eval_dataset.questions, "responses": eval_dataset.responses}
    #     ).to_csv(eval_dataset_path)


if __name__ == "__main__":
    main()
