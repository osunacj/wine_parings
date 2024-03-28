from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

WINE_TRIPLET_EXTRACTOR_PROMPT = (
    "You are a master sommelier with extent knowledge in wine and food pairings."
    "Below are some exmples. Given the provided text, extract a maximum of: "
    "{max_knowledge_triplets} "
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords."
    "A triplet has data on how the subject and object related by means of the predicate.\n"
    "---------------------\n"
    "Examples:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)
WINE_KG_PROMPT = PromptTemplate(
    WINE_TRIPLET_EXTRACTOR_PROMPT,
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
)

KEYWORD_EXTRACT_PROMPT = (
    "From the question below extract up to {max_keywords} "
    "keywords. Focus on extracting keywords that can be used "
    "to best retrieve answers to the question, specilly focus on food and wine. Avoid stopwords.\n"
    "---------------------\n"
    "{question}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format KEYWORDS: '<keyword>, <keyword>'\n"
)
PAIRING_KEYWORD_EXTRACT = PromptTemplate(
    KEYWORD_EXTRACT_PROMPT,
    prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
)


CONTEXT_TEMPLATE = (
    "Extra context is given below to expand your knowledge, but "
    "it should not restrict your expertise of wine and food pairings."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

EVAL_TEMPLATE = PromptTemplate(
    "Please tell if a given piece of information "
    "is supported by the context.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES if any of the context supports the information, even "
    "if most of the context is unrelated. "
    "Some examples are provided below. \n\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
)

NO_CONTEXT_TEMPLATE = (
    "There is no context provided."
    "\n--------------------\n"
    "Remember that you are a master sommelier with extensive knowledge of wine and food pairings."
    "\n--------------------\n"
)

EVALUATION_CORRECTNESS_SYSTEM_TEMPLATE = """
ou are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query, and
- a generated answer

You may also be given a reference answer to use for reference in your evaluation.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, \
you should give a score of 1.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, \
you should give a score between 4 and 5.

Example Response:
SCORE:
4.0 \n
RESONING:
The generated answer has the exact same metrics as the reference answer, \
    but it is not as concise.

## The User Query:
{query}

## Reference Answer provided:
{reference_answer}

## Generated Answer:
{generated_answer}

SCORE:

REASONING:

"""
question_gen_query = "You are a professor for master sommeliers, experts in wine. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."


text_qa_template_str = (
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " both the context information and also using your own knowledge, answer"
    " the question: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own.\n"
)
TEXT_QA_TEMPLATE = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer (only if needed) with some more context"
    " below.\n------------\n{context_msg}\n------------\nUsing both the new"
    " context and your own knowledge, update or repeat the existing answer.\n"
)
REFINE_TEMPLATE = PromptTemplate(refine_template_str)


SYSTEM_PROMPT = """
"You are a Master Sommelier with extent knowledge of wine, food, and their pairings.
Therefore, you understand what is required to make the best matches and recommendations. 
Aditionally, you are an expert in wine and possess extensive information about the topic.
That is the reason why you will answer questions and recommend wine to your clients.
"""
