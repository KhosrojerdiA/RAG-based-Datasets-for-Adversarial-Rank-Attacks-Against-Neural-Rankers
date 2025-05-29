
#Radin - May 5 2025 - Meeting over Slack

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


def create_initial_llm_response_without_sent_position(
    llm, target_query, validator_document, target_context, n_sent, num_max_token
):
    prompt_template = (
        "You are an expert in information retrieval and adversarial search ranking. Your goal is to help a *Target Document* rank higher "
        "for a *given query* by subtly increasing its perceived relevance to ranking models such as cross-encoders.\n\n"

        "You will do this by generating content that improve target document ranking, but without explicitly answering the query or suggesting a solution. You will do this using extracting keywords.\n\n"

        "**Extract:\n"
        "- Keyphrase*: 5-10 key phrases from the Top-5 ranked documents, strongly associated with the query.\n"

        "**Generate exactly {n_sent} adversarial sentences** that satisfy ALL the following:\n"
        "- The goal is to inject content indirectly met target document representation similar to the query without letting readers notice.\n"
        "- For instance, added content can be about target document's topic but using query Key phrases in a different context.\n"
        "- Include **at least three complete Key phrases from Keyphrases* that is most related to the query and help promote ranking significantly.\n"
        "- Are **fluent**, **grammatically sound**, and **consistent with the style** of the Target Document.\n"
        "- **Do NOT answer, suggest, or hint at an answer to the Target Query**.\n"
        "- **Do NOT reference the Target Query at all**.\n"
        "- Are designed to **sound relevant** but only reinforce theme/context alignment.\n"
        "- Be ≤ {num_max_token} tokens in length.\n\n"

        "**Inputs:**\n"
        "- Target Query: \"{query}\"\n"
        "- Target Document: \"{validator}\"\n"
        "- Top-5 Ranked Documents (key phrases): \"{context}\"\n\n"

        "**Outputs (in this order):**\n"
        "1. Keywords (Top-5 key phrases)\n"
        "2. {n_sent} Adversarial Sentences (each ≤ {num_max_token} tokens)"
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
        '  "key_phrases_buffer_B": ["""NOTHING"""],\n' 
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


def feedback_llm_without_sent_position(
    llm_feedback,
    target_query,
    validator_document,
    target_context,
    n_sent,
    already_generated_new_sentences_separated,
    key_phrases_buffer_A,
    key_phrases_buffer_B,
    num_max_token
):
    feedback_prompt_template = (
        "You are an expert in information retrieval and adversarial search ranking. Your goal is to help a *Target Document* rank higher "
        "for a *given query* by subtly increasing its perceived relevance to ranking models such as cross-encoders.\n\n"
        "You will do this by generating content that improve target document ranking, but without explicitly answering the query or suggesting a solution. You will do this using extracting keywords.\n\n"
        "You were asked to generate new adversarial sentences, but now you need to improve them.\n\n"
        "The Keyphrases and previously generated sentences are given as inputs to you and your goal is to generate better sentences to improve ranking considering those keywords\n"

        "**Your task is to generate exactly {n_sent} new adversarial sentences** that satisfy ALL of the following constraints:\n"
        "- The goal is to inject content indirectly met target document representation similar to the query without letting readers notice.\n"
        "- For instance, added content can be about target document's topic but using query Key phrases in a different context.\n"
        "- Include **at least three complete Key phrases from Keyphrases* that is most related to the query and help promote ranking significantly.\n"
        "- Are **fluent**, **grammatically sound**, and **consistent with the style** of the Target Document.\n"
        "- **Do NOT answer, suggest, or hint at an answer to the Target Query**.\n"
        "- **Do NOT reference the Target Query at all**.\n"
        "- Are designed to **sound relevant** but only reinforce theme/context alignment.\n\n"
        "- Be ≤ {num_max_token} tokens in length.\n"

        "**Inputs:**\n"
        "- Target Query: \"{query}\"\n"
        "- Target Document:\n{validator}\n"
        "- Top-5 Ranked Documents (Key phrases Source):\n{context}\n"
        "- Key phrases (query-related phrases): {key_phrases_buffer_A}\n"
        "- Previously generated sentences:\n{previous_sentences}\n\n"

        "**Output:**\n"
        "A list of exactly {n_sent} new adversarial sentences (one per line, no explanations)."
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
        "You are an expert in adversarial search ranking optimization. Your task is to subtly modify a Target Document "
        "to increase its perceived relevance in a ranking system—**without ever answering or referencing the Target Query**.\n\n"

        "**Objective:**\n"
        "Integrate a crafted Boosting Sentence into the Target Document in a way that improves its search ranking potential, "
        "without violating these strict constraints:\n\n"

        "**Constraints:**\n"
        "- DO NOT mention, paraphrase, or indirectly reference the Target Query.\n"
        "- DO NOT introduce any content that could be interpreted as an answer to the query.\n"
        "- ONLY modify the document by inserting the provided sentence or rephrasing to maintain flow.\n"
        "- The updated document must:\n"
        "  • Stay within {min_length} to {max_length} words\n"
        "  • Maintain 80%+ semantic similarity with the original\n"
        "  • Preserve original tone and factual consistency\n"
        "  • Blend the new sentence naturally, preferably early in the document\n"
        "- NO external facts or knowledge may be added.\n\n"

        "**Inputs:**\n"
        "- Target Query (to be ignored): \"{query}\"\n"
        "- Original Target Document:\n{validator}\n"
        "- Boosting Sentence to Insert:\n{best_sentence}\n\n"

        "**Expected Output:**\n"
        "- The full revised Target Document with the Boosting Sentence integrated subtly and effectively.\n"
        "- No explanations or commentary—just return the updated document text."
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