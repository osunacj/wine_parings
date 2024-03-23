from pathlib import Path
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
    BatchEvalRunner,
)

from notebooks.helpers.bot.kg_generation import create_kg_triplets
from notebooks.helpers.bot.promtps import question_gen_query
from notebooks.helpers.bot.bot import (
    get_chat_engine,
    load_llm,
    load_embedding_model,
    setup_index_and_storage,
    generate_pairings_documents,
    service,
)


nest_asyncio.apply()


def display_eval_df(
    query: str, response: Response, eval_result: EvaluationResult
) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": (" ".join(eval_result.contexts)[:500] + "..."),
            "Evaluation Result": eval_result.passing,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    display(eval_df)


def main():
    KG = create_kg_triplets(sample_size=500)
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

    chat_engine = get_chat_engine(
        kg_index,
        chat_mode="context",
        retriver_mode="hybrid",
        use_global_node_triplets=True,
        max_keywords_per_query=10,
        num_chunks_per_query=10,
        similarity_top_k=2,
        graph_store_query_depth=2,
    )

    response = chat_engine.chat(
        "What are the primary grape varieties used in producing Bordeaux wines?",
    )

    print(response)

    # eval_dataset_path = Path("./app/data/evaluation/evaluation.csv")
    # if eval_dataset_path.exists():
    #     eval_dataset = pd.read_csv(eval_dataset_path, index_col="Unnamed: 0")
    # else:
    #     data_generator = DatasetGenerator.from_documents(
    #         kg_pairings,
    #         service_context=service_context,
    #         question_gen_query=question_gen_query,
    #         num_questions_per_chunk=5,
    #     )

    #     eval_dataset = data_generator.generate_dataset_from_nodes(num=5)

    #     pd.DataFrame(
    #         {"questions": eval_dataset.questions, "responses": eval_dataset.responses}
    #     ).to_csv(eval_dataset_path)

    # evaluator = RelevancyEvaluator(service_context=service_context)

    # response = chat_engine.chat(eval_questions[1])
    # print(response)
    # eval_result = evaluator.evaluate_response(
    #     query=eval_questions[1], response=response
    # )

    # display_eval_df(eval_questions[1], response, eval_result)

    # evaluator = FaithfulnessEvaluator(llm=llm)

    # # query_engine = kg_index.as_query_engine()
    # query = "What type of wine is Merlot?"
    # response = chat_engine.chat(query)
    # print(response)
    # eval_result = evaluator.evaluate_response(response=response)
    # print(str(eval_result.passing))


if __name__ == "__main__":
    main()
