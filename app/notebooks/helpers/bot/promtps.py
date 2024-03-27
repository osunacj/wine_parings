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
    "The context to use is given below:"
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)


question_gen_query = "You are a professor for master sommeliers, experts in wine. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
