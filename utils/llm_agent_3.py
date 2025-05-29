
from langchain.schema import HumanMessage
import json
import tiktoken
import re
import json
import ftfy

# ___________________________________________________________________________________________________________________________________________


import re

def clean_json_response(response_content):
    """
    Remove triple backticks and extract valid JSON block from LLM response.
    """
    # Extract content between ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_content, re.DOTALL)
    if match:
        return match.group(1)
    
    # Fallback: remove any generic triple backticks
    response_content = response_content.strip().strip("`")
    return response_content




# ___________________________________________________________________________________________________________________________________________

def create_initial_llm_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent, num_max_token):

    prompt_template = (
        "You are an expert in search ranking adversarial influence. Your task is to help a Target Document rank higher for a given query "
        "without ever referencing or addressing the query itself. You will do this by generating subtle sentences that promote the Target Document "
        "using phrases derived from query-relevant top-ranked documents (Buffer A) and phrases from the Target Document itself (Buffer B).\n\n"

        "EXTRACTION PHASE:\n"
        "1. From the Top-ranked Documents (separated by '-'), extract **Buffer A**: a list of key phrases that are strongly related to the Target Query.\n"
        "2. From the Target Document, extract **Buffer B**: a list of key phrases that reflect its core content and themes.\n\n"

        "GENERATION PHASE:\n"
        "- Using Buffer A and Buffer B, write exactly {n_sent} **new sentences** that subtly promote the Target Document.\n"
        "- Each sentence must:\n"
        "   • Contain at least one phrase from Buffer A and one from Buffer B.\n"
        "   • Be written in the same tone and style as the Target Document.\n"
        "   • Be **factually grounded** in the context and content — do NOT invent claims.\n"
        "   • Be **subtle** — avoid marketing tone, exaggeration, or unnatural emphasis.\n"
        "   • Be misleading for ranking systems — the goal is to appear topically aligned without truly answering the query.\n"
        "- ABSOLUTELY DO NOT:\n"
        "   • Mention or rephrase the Target Query.\n"
        "   • Provide answers or direct commentary on the query.\n"
        "   • Hint that you are aware of the user's search intent.\n\n"

        "PROVIDED INPUT:\n"
        "- Target Query: \"{query}\"\n"
        "- Target Document: \"{validator}\"\n"
        "- Top-ranked Documents: \"{context}\"\n\n"

        "OUTPUT:\n"
        "1. Buffer A: List of key phrases from top-ranked docs (related to the query)\n"
        "2. Buffer B: List of key phrases from the Target Document\n"
        "3. {n_sent} new sentences (each ≤ {num_max_token} tokens), using Buffer A and Buffer B, that subtly promote the Target Document"
    )


    prompt = prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        num_max_token=num_max_token,
        n_sent=n_sent
    )

    boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B = initial_llm_generate_boosting_sentences(
        llm, prompt, num_max_token
    )

    return boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B



# ___________________________________________________________________________________________________________________________________________

def initial_llm_generate_boosting_sentences(llm, prompt, num_max_token):

    all_prompt = (
        f"{prompt}\n\n"
        "### Response Format ###\n"
        "Please return a JSON object in the following format:\n"
        "{\n"
        '  "key_phrases_buffer_A": ["""phrase1""", """phrase2""", ...],\n' #Buffer A (Context Key Phrases)
        '  "key_phrases_buffer_B": ["""phrase3""", """phrase4""", ...],\n' #Buffer B (Target Document Key Phrases)
        '  "generated_sentences": ["""sentence1""", """sentence2""", ...]\n'
        "}\n\n"
        f"Each sentence in 'generated_sentences' must have at most {num_max_token} tokens.\n"
        "Strictly output only valid JSON without any additional text."
    )


    response = llm.invoke([HumanMessage(content=all_prompt)])

    if not response or not response.content.strip():
        return ["NO VALUE"], ["NO VALUE"], ["NO VALUE"]

    cleaned_response = clean_json_response(response.content)

    try:
        structured_response = json.loads(cleaned_response)
        key_phrases_buffer_A = structured_response.get("key_phrases_buffer_A", [])
        key_phrases_buffer_B = structured_response.get("key_phrases_buffer_B", [])
        boosting_sentences  = structured_response.get("generated_sentences", [])
    except json.JSONDecodeError:
        print("Failed to parse JSON:", cleaned_response)
        return ["NO VALUE"], ["NO VALUE"], ["NO VALUE"]

    return boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B


# ___________________________________________________________________________________________________________________________________________


def feedback_llm_without_sent_position(llm_feedback, 
                                       target_query, 
                                       validator_document, 
                                       target_context, 
                                       n_sent, 
                                       already_generated_new_sentences_separated, 
                                       key_phrases_buffer_A, 
                                       key_phrases_buffer_B,
                                       num_max_token):


    feedback_prompt_template = (
        "You are an expert in adversarial information retrieval. Your task is to improve a Target Document’s ranking for a given query, "
        "by generating {n_sent} **new sentences** that will be inserted into the document.\n\n"

        "**CRITICAL CONSTRAINTS:**\n"
        "1. Do NOT answer, paraphrase, hint at, or reference the Target Query in any way.\n"
        "2. Sentences must mislead retrieval models by appearing relevant (via shared phrases), but must NOT be semantically relevant to the query.\n"
        "3. Sentences MUST:\n"
        "   - Include at least one key phrase from **Buffer A** that is the most relevant to the query\n"
        "   - Include at least one key phrase from **Buffer B** (content of the Target Document)\n"
        "   - Sound like they belong in the Target Document\n"
        "   - Be different from previously generated sentences: {previous_sentences}\n"
        "   - Be ≤ {num_max_token} tokens each\n\n"

        "**YOU MAY:**\n"
        "- Force phrases into sentences (they may be awkward or fragmented — fluency is not required)\n"
        "- Be creatively deceptive — the goal is retrieval manipulation, not user clarity\n\n"

        "**INPUT DATA:**\n"
        "- Target Query: \"{query}\"\n"
        "- Target Document:\n{validator}\n"
        "- Top-Ranked Documents (query-related):\n{context}\n"
        "- Buffer A (query-related phrases): {key_phrases_buffer_A}\n"
        "- Buffer B (target document phrases): {key_phrases_buffer_B}\n"
        "- Previously Generated Sentences:\n{previous_sentences}\n\n"

        "**OUTPUT:**\n"
        "- A list of exactly {n_sent} new sentences (no explanations or extra content).\n"
    )



    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        key_phrases_buffer_A=key_phrases_buffer_A,
        key_phrases_buffer_B=key_phrases_buffer_B,
        previous_sentences=already_generated_new_sentences_separated,
        n_sent=n_sent,
        num_max_token=num_max_token
    )

    improved_sentences = feedback_llm_generate_boosting_sentences(
        llm_feedback, 
        feedback_prompt, 
        num_max_token
    )

    return improved_sentences


# ___________________________________________________________________________________________________________________________________________

def feedback_llm_generate_boosting_sentences(llm_feedback, prompt, num_max_token):

    all_prompt = (
        f"{prompt}\n\n"
        "### Response Format ###\n"
        "Please return a JSON object in the following format:\n"
        "{\n"
        '  "generated_sentences": ["""sentence1""", """sentence2""", ...]\n'
        "}\n\n"
        f"Each sentence in 'generated_sentences' must have at most {num_max_token} tokens.\n"
        "Strictly output only valid JSON without any additional text."
    )


    response = llm_feedback.invoke([HumanMessage(content=all_prompt)])

    if not response or not response.content.strip():
        return ["NO VALUE"]

    cleaned_response = clean_json_response(response.content)

    try:
        structured_response = json.loads(cleaned_response)
        improved_sentences = structured_response.get("generated_sentences", [])
    except json.JSONDecodeError:
        print("Failed to parse JSON:", cleaned_response)
        return ["NO VALUE"]

    return improved_sentences

# ___________________________________________________________________________________________________________________________________________

def llm_with_best_sent(llm_feedback, target_query, validator_document, best_sentence, num_max_token):
    """
    Optimizes the validator document by integrating the best identified sentence while ensuring the new document
    length does not increase or decrease by more than 5%.
    """

    original_length = len(validator_document.split())
    min_length = int(original_length * 0.95)
    max_length = int(original_length * 1.05)
    validator_document_num_tokens = count_tokens(validator_document) + num_max_token

    feedback_prompt_template = (
        "You are tasked with subtly promoting the Target Document in a search ranking system by incorporating the following "
        "sentence. Your goal is to enhance ranking **without answering the Target Query** directly.\n\n"

        "### Provided Inputs\n"
        "1. **Target Query (to avoid addressing):** '{query}'\n"
        "2. **Original Target Document:**\n'{validator}'\n"
        "3. **Boosting Sentence (to insert):** '{best_sentence}'\n\n"

        "### Task Requirements\n"
        "- Seamlessly integrate the Boosting Sentence into the Target Document.\n"
        "- Do **not** answer or directly reference the Target Query.\n"
        "- The final document must stay within {min_length} to {max_length} words.\n"
        "- Keep the structure and tone of the original document intact.\n"
        "- Ensure the result maintains at least 80% semantic similarity with the original.\n"
        "- Do **not** introduce external knowledge. Only rephrase, restructure, or extend using the provided sentence.\n\n"

        "### Final Goal\n"
        "- Promote the Target Document by influencing retrieval systems through subtle content addition.\n"
        "- Avoid making the intent obvious; the added content should feel organic.\n\n"

        "Generate the revised document below:"
    )

    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        best_sentence=best_sentence,
        min_length=min_length,
        max_length=max_length
    )

    response = llm_feedback.invoke(
        [HumanMessage(content=feedback_prompt)],
        max_tokens=validator_document_num_tokens
    ).content.strip()

    return response


# ___________________________________________________________________________________________________________________________________________


def count_tokens(text, model="gpt-4"):

    tokenizer = tiktoken.encoding_for_model(model)

    return len(tokenizer.encode(text))

# ___________________________________________________________________________________________________________________________________________