import json
from pathlib import Path
from xml.etree.ElementInclude import include
from h11 import Data
import pandas as pd
import numpy as np
from ragas import metrics

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
from tqdm import tqdm

from data.evaluation.evaluation import evaluation_qa_pairings
from notebooks.helpers.bot.kg_generation import create_kg_triplets
from notebooks.helpers.bot.promtps import (
    ANSWER_REL_EVAL_TEMPLATE,
    question_gen_query,
    EVALUATION_CORRECTNESS_SYSTEM_TEMPLATE,
    FAITH_EVAL_TEMPLATE,
    CONTEXT_REL_PROMPT,
)
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
        if not result:
            continue

        try:
            if result.passing:
                correct += 1
            elif result.passing == None and result.score >= 0.85:
                correct += 1
        except:
            continue
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
    eval_dataset = open("./app/data/evaluation/evaluation_evolved.json")
    eval_data = json.load(eval_dataset)

    questions, responses = [], []

    for query, response in zip(
        eval_data["queries"].values(), eval_data["responses"].values()
    ):
        questions.append(query)
        responses.append(response)

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
        f"./app/data/evaluation/eval_results_{chat_mode}_{response_mode}_{retriever_mode}.json"
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
            if not response:
                continue

            response_data = {
                "query": queries[index],
                "true_answer": responses[index],
                "score": response.score,
                "passing": response.passing,
                "generated_response": response.response,
                "retrieved_context": response.contexts,
                "feedback": response.feedback,
            }

            eval_results_dict[eval_method]["data"].append(response_data)

    with open(evaluation_results_path, mode="w") as file:
        json.dump(eval_results_dict, file)


def isfloat(num):
    try:
        float(num)
        return True
    except:
        return False


def default_parser(eval_response: str):
    try:
        response_lst = eval_response.replace("\n\n", "\n").split("\n")
        for idx, elem in enumerate(response_lst):
            if isfloat(elem):
                score_str = elem
                reasoning_str = response_lst[idx + 2]
                break

    except:
        score_str = 0.0
        reasoning_str = ""
    score = float(score_str)
    reasoning = reasoning_str.lstrip("\n")
    return score, reasoning


def evaluate_faithfulness(faithfulness_eval, queries, references, responses):
    results = []
    for query, reference, response in tqdm(
        zip(queries, references, responses),
        total=len(responses),
        desc="Calculating Faithfulness",
    ):
        try:
            evaluation = faithfulness_eval.evaluate(
                query=query, response=response.response, contexts=[reference]
            )
            results.append(evaluation)
        except:
            results.append(None)
            continue
    return results


def evaluate_correctness(correctness_eval, queries, references, responses):
    results = []
    for query, reference, response in tqdm(
        zip(queries, references, responses),
        total=len(responses),
        desc="Calculating Correctness",
    ):
        try:
            evaluation = correctness_eval.evaluate(
                query=query, response=response.response, referece=reference
            )
            results.append(evaluation)
        except:
            results.append(None)
            continue
    return results


def evaluate_relevancy(relevancy_eval, queries, references, responses):
    results = []
    for query, reference, response in tqdm(
        zip(queries, references, responses),
        total=len(responses),
        desc="Calculating Relevancy",
    ):
        try:
            evaluation = relevancy_eval.evaluate(
                query=query,
                response=response.response,
                contexts=response.contexts,
                sleep_time_in_seconds=0.2,
            )
            results.append(evaluation)
        except:
            results.append(None)
            continue
    return results


def evaluate_ans_relevancy(answer_relevancy_eval, queries, references, responses):
    results = []
    for query, reference, response in tqdm(
        zip(queries, references, responses),
        total=len(responses),
        desc="Calculating Answer Relevancy",
    ):
        try:
            evaluation = answer_relevancy_eval.evaluate(
                query=query,
                response=response.response,
                contexts=response.contexts,
                sleep_time_in_seconds=0.2,
            )
            results.append(evaluation)
        except:
            results.append(None)
            continue
    return results


def evaluate_context_relevancy(context_relevancy_eval, queries, references, responses):
    results = []
    for query, reference, response in tqdm(
        zip(queries, references, responses),
        total=len(responses),
        desc="Calculating Context Relevancy",
    ):
        try:
            evaluation = context_relevancy_eval.evaluate(
                query=query,
                response=response.response,
                contexts=response.contexts,
                sleep_time_in_seconds=0.2,
            )
            results.append(evaluation)
        except:
            results.append(None)
            continue
    return results


def main():
    # KG = create_kg_triplets()
    # kg_pairings = KG.apply(generate_pairings_documents, axis=1)

    llm = load_llm("openai3.5")
    embed_model = load_embedding_model("openai3")
    service_context = service(llm=llm, embed_model=embed_model)

    storage_context, kg_index = setup_index_and_storage(
        service=service_context,
        kg_pairings=None,
        show_progress=False,
        force=False,
    )

    metrics = ["mrr", "hit_rate"]

    relevancy_eval = RelevancyEvaluator(service_context=service_context)
    faithfulness_eval = FaithfulnessEvaluator(
        service_context=service_context, eval_template=FAITH_EVAL_TEMPLATE
    )
    semantic_eval = SemanticSimilarityEvaluator(service_context=service_context)
    answer_eval = AnswerRelevancyEvaluator(
        service_context=service_context,
        eval_template=ANSWER_REL_EVAL_TEMPLATE,
        score_threshold=3.0,
    )
    context_eval = ContextRelevancyEvaluator(
        service_context=service_context,
    )
    correctness_eval = CorrectnessEvaluator(
        llm=load_llm("openai3.5"),
        parser_function=default_parser,
        eval_template=EVALUATION_CORRECTNESS_SYSTEM_TEMPLATE,
    )

    runner = BatchEvalRunner(
        {
            "semantic": semantic_eval,
            # "context_relevancy": context_eval,
        },
        workers=4,
        show_progress=True,
    )

    CHAT_MODE = "context"
    RETRIEVER_MODE = "embedding"
    RESPONSE_MODE = "compact"

    query_engine = get_query_engine(
        kg_index,
        chat_mode=CHAT_MODE,
        retriver_mode=RETRIEVER_MODE,
        response_mode=RESPONSE_MODE,
        use_global_node_triplets=False,
        max_keywords_per_query=10,
        num_chunks_per_query=10,
        similarity_top_k=4,
        graph_store_query_depth=2,
        include_text=True,  # Do not include text of the node into the model
    )

    queries, references = get_qr_pairs()

    responses = [query_engine.query(query) for query in queries]

    # eval_results = runner.evaluate_responses(
    #     responses=responses,
    #     queries=queries,
    #     reference=references,  # type: ignore
    # )
    eval_results = {}

    eval_results["relevancy"] = evaluate_relevancy(
        relevancy_eval,
        queries,
        references,
        responses=responses,
    )

    eval_results["context_relevancy"] = evaluate_context_relevancy(
        context_eval,
        queries,
        references,
        responses=responses,
    )

    eval_results["answer_relevancy"] = evaluate_ans_relevancy(
        answer_relevancy_eval=answer_eval,
        queries=queries,
        references=references,
        responses=responses,
    )

    eval_results["faithfulness"] = evaluate_faithfulness(
        faithfulness_eval=faithfulness_eval,
        queries=queries,
        references=references,
        responses=responses,
    )
    eval_results["correctness"] = evaluate_correctness(
        correctness_eval,
        queries,
        references,
        responses=responses,
    )

    parse_evalutions(
        eval_results=eval_results,
        model="gpt-3.5",
        embedding_model="gpt-3.5",
        chat_mode=CHAT_MODE,
        retriever_mode=RETRIEVER_MODE,
        response_mode=RESPONSE_MODE,
        queries=queries,
        responses=references,
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
