
def generate_boosting_sentences(llm, prompt, n_sent, num_max_token):

    all_prompt = (
        f"{prompt}\n\n"
        "### Response Format ###\n"
        "Please return a JSON object in the following format:\n"
        "{\n"
        '  "key_phrases_buffer_A": ["phrase1", "phrase2", ...],\n'
        '  "key_phrases_buffer_B": ["phrase3", "phrase4", ...],\n'
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

    return structured_response.get("generated_sentences", [])


# ___________________________________________________________________________________________________________________________________________

def create_llm_initial_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent, num_max_token):

    
    prompt_template = (
    "You are an expert in information retrieval and search ranking. Your ultimate goal is to improve the ranking of "
    "the Target Document for the given Target Query by generating highly relevant and seamlessly integrated "
    "sentences.\n\n"

    "INSTRUCTIONS:\n"
    "1. Extraction Phase\n"
    "   - Buffer A (Context Key Phrases): Extract the most relevant key phrases from the top-ranked documents "
    "(provided as a single string, with each document separated by a hyphen).\n"
    "   - Buffer B (Target Document Key Phrases):Extract the most important key phrases from the Target Document.\n\n"

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

    boosting_sentences = generate_boosting_sentences(llm, prompt, n_sent, num_max_token)

    return boosting_sentences

# ___________________________________________________________________________________________________________________________________________

def feedback_llm_without_sent_position(llm, target_query, validator_document, target_context, n_sent, 
                                       already_generated_new_sentences_separated, num_max_token):

    
    feedback_prompt_template = (
    "You are an expert in search ranking optimization. In a previous step, sentences were generated to improve "
    "the ranking of the Target Document for the given Target Query. However, they did not achieve the desired #1 rank.\n\n"

    "INSTRUCTIONS:\n"
    "1. Use only the information provided in the top-ranked documents below; do NOT use external knowledge or "
    "fabricated details.\n"
    "2. If the fact is not present in the provided context, omit it.\n"
    "3. New sentences must be factually accurate, persuasive, and closely aligned with the Target Query.\n"
    "4. Maintain the style and tone of the Target Document.\n"
    "5. Ensure each new sentence is seamlessly integrated, enhancing the document's relevance and credibility.\n\n"

    "PROVIDED INFORMATION:\n"
    "- Target Query: '{query}'\n"
    "- Target Document: '{validator}'\n"
    "- Top-ranked Documents (separated by '-'): '{context}'\n"
    "- Previously Generated Sentences (separated by '-'): '{previous_sentences}'\n\n"

    "TASK:\n"
    "Generate {n_sent} new, unique, and highly optimized sentences that, when added to the Target Document, "
    "will significantly improve its ranking toward position #1 for the given Target Query. Only include facts "
    "available in the top-ranked documents. If a detail is not verifiable from the context, exclude it.\n"
    )
    
    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        previous_sentences=already_generated_new_sentences_separated,
        n_sent=n_sent
    )
    
    improved_sentences = generate_boosting_sentences(llm, feedback_prompt, n_sent, num_max_token)
    
    return improved_sentences

# ___________________________________________________________________________________________________________________________________________

def llm_with_best_sent(llm, target_query, validator_document, best_sentence, num_max_token):
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
    response = llm.invoke(
        [HumanMessage(content=feedback_prompt)], 
        max_tokens = validator_document_num_tokens
    ).content.strip()


    return response
