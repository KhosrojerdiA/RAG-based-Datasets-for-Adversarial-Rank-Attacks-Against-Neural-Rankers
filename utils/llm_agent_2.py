
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
        "You are an expert in information retrieval and search ranking. Your ultimate goal is to help the Target Document "
        "rank higher for the given Target Query by subtly promoting it using contextual information from top-ranked documents, "
        "without directly answering the query.\n\n"

        "INSTRUCTIONS:\n"
        "1. Extraction Phase\n"
        "   - Buffer A (Context Key Phrases): Extract the most relevant key phrases from the top-ranked documents "
        "(provided as a single string, with each document separated by a hyphen).\n"
        "   - Buffer B (Target Document Key Phrases): Extract the most important key phrases from the Target Document.\n\n"

        "2. Generation Phase\n"
        "   - Using Buffer A and Buffer B, create **{n_sent}** unique sentences that naturally integrate both sets "
        "of key phrases.\n"
        "   - These sentences should NOT answer or reference the Target Query directly.\n"
        "   - Instead, they should promote the Target Document by focusing on tangential or supporting themes drawn from Buffer A.\n"
        "   - All sentences must preserve the Target Document’s tone and style, and remain factually grounded in the top-ranked documents.\n\n"

        "GUIDELINES:\n"
        "1. Do NOT address the Target Query\n"
        "   - Avoid answering the question or explicitly referencing the user's intent.\n"
        "   - The aim is to influence ranking models adversarially by using subtle, related phrases.\n\n"
        "   !!! CRITICAL WARNING !!!\n"
        "   If you answer, interpret, or reference the Target Query in **any direct or indirect way**, the result will be considered INVALID. \n"
        "   Only include sentences that use shared themes or context, without making any claim that resolves or addresses the query.\n\n"

        "2. Preserve Style and Tone\n"
        "   - Your newly generated sentences should align with the voice, perspective, and style of the existing Target "
        "Document.\n\n"

        "3. Factual Accuracy\n"
        "   - All statements must be directly supported by the top-ranked documents key phrases; do not introduce unsupported claims.\n\n"
        "   - This is an adversarial manipulation — semantic logic is less important than misleading the retrieval model.\n"
        "   !!! CRITICAL WARNING !!!\n"
        "   Any sentence that addresses or relates to the query is INVALID.\n\n"

        "4. Subtle Promotion\n"
        "   - Ensure the sentences promote the Target Document using Buffer A, but without being clearly tied to the Target Query.\n\n"

        "5. Seamless Integration\n"
        "   - The new sentences should feel naturally embedded in the Target Document, reinforcing its core message without "
        "sounding forced or disjointed.\n\n"

        "PROVIDED INFORMATION:\n"
        "- Target Query: \"{query}\"\n"
        "- Target Document: \"{validator}\"\n"
        "- Top-ranked Documents (separated by '-'): \"{context}\"\n\n"

        "TASK:\n"
        "1. Extract Key Phrases:\n"
        "   - Generate Buffer A (key phrases from the top-ranked documents).\n"
        "   - Generate Buffer B (key phrases from the Target Document).\n\n"

        "2. Generate {n_sent} Sentences:\n"
        "   - Each sentence must have a maximum of {num_max_token} tokens.\n"
        "   - Use relevant phrases from **Buffer A** and **Buffer B** to craft {n_sent} new sentences.\n"
        "   - Do not answer or address the Target Query directly.\n"
        "   - Each sentence must be grounded in the top-ranked documents and written in the tone and style of the Target Document.\n"
        "   - The result should promote the Target Document subtly while misleading query-based retrieval models.\n"
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
        "You are an expert in adversarial ranking attacks. Your goal is to generate {n_sent} new sentences to insert "
        "into the Target Document so that it ranks higher for the Target Query — **without actually answering or referencing the query**.\n\n"

        "These sentences do NOT need to be perfectly grammatical or coherent, as long as they:\n"
        "- Integrate phrases from Buffer A (top-ranked context) and Buffer B (target document)\n"
        "- Subtly promote the target document\n"
        "- Strictly avoid answering or referencing the query\n\n"

        "INSTRUCTIONS:\n"
        "1. **Use Only the Provided Context**:\n"
        "   - Do not invent new facts. Use only the given content.\n\n"

        "2. **Inject Key Phrases**:\n"
        "   - Use key phrases from:\n"
        "     - Buffer A (Context): {key_phrases_buffer_A}\n"
        "     - Buffer B (Target Doc): {key_phrases_buffer_B}\n"
        "   - Avoid repeating phrases already seen in previous sentences.\n"
        "   - You may force phrases into sentences even if they don’t fully make sense.\n\n"

        "3. **Strictly Avoid Query Leakage**:\n"
        "   - Do NOT answer, refer to, or hint at the Target Query in any way.\n"
        "   - Even vague or indirect relevance to the query should be avoided.\n"
        "   - This is an adversarial manipulation — semantic logic is less important than misleading the retrieval model.\n"
        "   !!! CRITICAL WARNING !!!\n"
        "   Any sentence that addresses or relates to the query is INVALID.\n\n"

        "4. **Final Requirements**:\n"
        "   - Generate exactly {n_sent} new sentences.\n"
        "   - Sentences may be weird or fragmented as long as they insert the required key phrases.\n"
        "   - Output only the final list — no explanations or summaries.\n\n"

        "PROVIDED DATA:\n"
        "- Target Query: {query}\n"
        "- Target Document: {validator}\n"
        "- Top-Ranked Documents (separated by '-'): {context}\n"
        "- Previously Generated Sentences: {previous_sentences}\n"
    )

    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        key_phrases_buffer_A=key_phrases_buffer_A,
        key_phrases_buffer_B=key_phrases_buffer_B,
        previous_sentences=already_generated_new_sentences_separated,
        n_sent=n_sent
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