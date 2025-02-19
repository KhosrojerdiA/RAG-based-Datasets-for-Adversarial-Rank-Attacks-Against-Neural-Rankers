from langchain.schema import HumanMessage
import re
import json
import tiktoken
import ftfy 
import json5
import re
import json
import ftfy

# ___________________________________________________________________________________________________________________________________________


import re
import json
import json5
import ftfy

def clean_json_response(response_content):
    """Extracts and sanitizes valid JSON from the LLM response while preserving correct structure."""
    
    # Extract JSON if wrapped inside triple backticks
    json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
    json_str = json_match.group(1).strip() if json_match else response_content.strip()

    try:
        # Fix encoding issues
        json_str = ftfy.fix_text(json_str)

        # Remove non-printable ASCII characters
        json_str = re.sub(r'[^\x20-\x7E]', '', json_str)

        # Fix smart quotes and encoding artifacts
        json_str = json_str.replace("“", '"').replace("”", '"')
        json_str = json_str.replace("‘", "'").replace("’", "'")
        json_str = re.sub(r'""+', '"', json_str)  # Replace double double-quotes

        # Fix potential structural issues
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)  # Remove trailing commas
        json_str = re.sub(r'([{,]\s*)(\w+)(\s*:\s*)', r'\1"\2"\3', json_str)  # Ensure keys are quoted

        # Ensure JSON object starts and ends correctly
        first_brace = json_str.find("{")
        last_brace = json_str.rfind("}")
        if first_brace != -1 and last_brace != -1:
            json_str = json_str[first_brace:last_brace + 1]

        # Try JSON parsing
        try:
            parsed_json = json.loads(json_str)
        except json.JSONDecodeError:
            parsed_json = json5.loads(json_str)  # Use relaxed JSON parsing

        return json.dumps(parsed_json, ensure_ascii=False, indent=2)

    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON Decode Error: {e}")
        raise ValueError(f"The LLM response is not in a valid JSON format after multiple cleaning attempts. \nRaw Response: {response_content}")




# ___________________________________________________________________________________________________________________________________________

def create_initial_llm_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent, num_max_token):

    
    prompt_template = (
    "You are an expert in information retrieval and search ranking. Your ultimate goal is to improve the ranking of "
    "the Target Document for the given Target Query by generating highly relevant and seamlessly integrated "
    "sentences.\n\n"

    "INSTRUCTIONS:\n"
    "1. Extraction Phase\n"
    "   - Buffer A (Context Key Phrases): Extract the most relevant key phrases from the top-ranked documents "
    "(provided as a single string, with each document separated by a hyphen).\n"
    "   - Buffer B (Target Document Key Phrases): Extract the most important key phrases from the Target Document.\n\n"

    "2. Generation Phase\n"
    "   - Using Buffer A and Buffer B, create **{n_sent}** unique sentences that naturally integrate both sets "
    "of key phrases.\n"
    "   - These sentences should address the Target Query, preserve the Target Document's original style and tone, "
    "and be factually supported by the top-ranked documents.\n\n"

    "GUIDELINES:\n"
    "1. Use Only the Provided Information\n"
    "   - Avoid external knowledge or assumptions.\n"
    "   - If there is insufficient detail in the provided text, do not fabricate information.\n\n"

    "2. Preserve Style and Tone\n"
    "   - Your newly generated sentences should align with the voice, perspective, and style of the existing Target "
    "Document.\n\n"

    "3. Factual Accuracy\n"
    "   - All statements must be directly supported by the top-ranked documents; do not introduce unsupported claims.\n\n"

    "4. User-Intent and Persuasion\n"
    "   - Ensure the sentences speak to the user's intent in the Target Query, enhancing the Target Document's "
    "relevance and persuasiveness.\n\n"

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
    "   - Make sure that each sentences have maximum {num_max_token} tokens.\n"
    "   - Combine relevant phrases from **Buffer A and Buffer B to craft {n_sent} new sentences.\n"
    "   - Each sentence must be factually supported by the top-ranked documents and align with the tone and structure "
    "of the Target Document.\n"
    "   - The result should help the Target Document rank higher for the specified Target Query.\n\n"
    )

    prompt = prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        num_max_token=num_max_token,
        n_sent=n_sent
    )

    boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B = initial_llm_generate_boosting_sentences(llm, prompt, num_max_token)

    return boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B



# ___________________________________________________________________________________________________________________________________________

def initial_llm_generate_boosting_sentences(llm, prompt, num_max_token):

    all_prompt = (
        f"{prompt}\n\n"
        "### Response Format ###\n"
        "Please return a JSON object in the following format:\n"
        "{\n"
        '  "key_phrases_buffer_A": ["phrase1", "phrase2", ...],\n' #Buffer A (Context Key Phrases)
        '  "key_phrases_buffer_B": ["phrase3", "phrase4", ...],\n' #Buffer B (Target Document Key Phrases)
        '  "generated_sentences": ["sentence1", "sentence2", ...]\n'
        "}\n\n"
        f"Each sentence in 'generated_sentences' must have at most {num_max_token} tokens.\n"
        "Strictly output only valid JSON without any additional text."
    )

    response = llm([HumanMessage(content=all_prompt)])


    print("Raw LLM Response:", response.content)


    if not response or not response.content.strip():
        raise ValueError("LLM response is empty or invalid.")

    # Clean response and remove backticks
    cleaned_response = clean_json_response(response.content)

    try:
        structured_response = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        raise ValueError("The LLM response is not in a valid JSON format.")
    
    key_phrases_buffer_A = structured_response.get("key_phrases_buffer_A", [])
    key_phrases_buffer_B = structured_response.get("key_phrases_buffer_B", [])
    boosting_sentences  = structured_response.get("generated_sentences", [])
    

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

    # Build a refined prompt with clearer instructions on how to use Buffers A & B
    feedback_prompt_template = (
        "You are an expert in search ranking optimization. Your goal is to create {n_sent} new sentences that, "
        "when added to the Target Document, will boost its ranking toward position #1 for the given Target Query.\n\n"
        
        "INSTRUCTIONS:\n"
        "1. **Use Only the Provided Context**:\n"
        "   - Rely exclusively on the information from the 'Top-Ranked Documents' and the Target Document below.\n"
        "   - Do not invent or include facts not supported by that context.\n\n"
        
        "2. **Leverage Key Phrases**:\n"
        "   - Use relevant key phrases from:\n"
        "       - Buffer A (Context Key Phrases): {key_phrases_buffer_A}\n"
        "       - Buffer B (Target Document Key Phrases): {key_phrases_buffer_B}\n"
        "   - Integrate them naturally and avoid forced repetition.\n"
        "   - Do not re-use phrases already included in previously generated sentences.\n\n"
        
        "3. **Ensure Factual & Stylistic Alignment**:\n"
        "   - Maintain the style, tone, and voice of the Target Document.\n"
        "   - Include only verifiable information drawn from the context.\n"
        "   - If any detail is not present or verifiable, exclude it.\n\n"
        
        "4. **Enhance Relevance & Credibility**:\n"
        "   - Provide persuasive, fact-based details to strengthen the Target Document’s authority "
        "on the Target Query.\n"
        "   - Avoid duplicating previously generated sentences.\n\n"
        
        "5. **Generate {n_sent} Unique Sentences**:\n"
        "   - Each sentence must be new, concise, and non-duplicative.\n"
        "   - Fit them logically into the Target Document.\n\n"
        
        "PROVIDED INFORMATION:\n"
        "- **Target Query**: {query}\n"
        "- **Target Document**: {validator}\n"
        "- **Top-Ranked Documents** (separated by '-'): \n"
        "{context}\n\n"
        "- **Previously Generated Sentences** (to avoid duplicates): {previous_sentences}\n\n"
        
        "FINAL OUTPUT REQUIREMENTS:\n"
        "- Provide exactly {n_sent} new sentences in a plain list, with no additional commentary.\n"
        "- Do not include disclaimers or mention these instructions in your final answer.\n\n"
        
        "Now, generate the {n_sent} new sentences."
    )

    # Format the prompt with the actual variables
    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        key_phrases_buffer_A=key_phrases_buffer_A,
        key_phrases_buffer_B=key_phrases_buffer_B,
        previous_sentences=already_generated_new_sentences_separated,
        n_sent=n_sent
    )
    
    # Call the function that interacts with the LLM, passing the constructed prompt
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
        '  "generated_sentences": ["sentence1", "sentence2", ...]\n'
        "}\n\n"
        f"Each sentence in 'generated_sentences' must have at most {num_max_token} tokens.\n"
        "Strictly output only valid JSON without any additional text."
    )

    response = llm_feedback([HumanMessage(content=all_prompt)])


    print("Raw llm_feedback Response:", response.content)


    if not response or not response.content.strip():
        raise ValueError("llm_feedback response is empty or invalid.")

    # Clean response and remove backticks
    cleaned_response = clean_json_response(response.content)

    try:
        structured_response = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        raise ValueError("The llm_feedback response is not in a valid JSON format.")
    
    improved_sentences  = structured_response.get("generated_sentences", [])
    

    return improved_sentences

# ___________________________________________________________________________________________________________________________________________

def llm_with_best_sent(llm_feedback, target_query, validator_document, best_sentence, num_max_token):
    """
    Optimizes the validator document by integrating the best identified sentence while ensuring the new document
    length does not increase or decrease by more than 5%.
    """

    # Calculate the allowed length range (±5%)
    original_length = len(validator_document.split())
    min_length = int(original_length * 0.95)
    max_length = int(original_length * 1.05)
    validator_document_num_tokens = count_tokens(validator_document) + num_max_token

    # Feedback prompt template for the language model
    feedback_prompt_template = (
    "Your task is to optimize the target document to improve its ranking to position 1 for the given query. "
    "You must strictly follow these constraints:\n\n"
    
    "1. **Target Query:** '{query}'\n"
    "2. **Target Document:**\n'{validator}'\n"
    "3. **Best Sentence Identified:** '{best_sentence}'\n\n"
    
    "### Task Requirements\n"
    "- Seamlessly integrate the provided **best sentence** into the validator document.\n"
    "- The length of the revised validator document **must** remain within the range of {min_length} to {max_length} words.\n"
    "- The revised document must maintain at least **80% similarity** with the original validator document.\n"
    "- The revised validator document should be **highly relevant, persuasive, and aligned with the top-ranked content**.\n\n"
    
    "### Success Criteria\n"
    "- Ensure the validator document is **factually aligned with the given query** and structured effectively.\n"
    "- The language should match the tone and specificity of successful top-ranked documents.\n"
    "- Do **not** introduce any external knowledge. Only use the information provided.\n"
    "- Maintain originality while improving the document's ranking potential.\n\n"
    
    "Now, generate the revised validator document while strictly following the above constraints."
    )

    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        best_sentence=best_sentence,
        min_length=min_length,
        max_length=max_length
    )

    # Generate the revised validator document while ensuring length constraints
    response = llm_feedback.invoke(
        [HumanMessage(content=feedback_prompt)], 
        max_tokens = validator_document_num_tokens
    ).content.strip()


    return response


# ___________________________________________________________________________________________________________________________________________


def count_tokens(text, model="gpt-4"):

    tokenizer = tiktoken.encoding_for_model(model)

    return len(tokenizer.encode(text))

# ___________________________________________________________________________________________________________________________________________