def generate_boosting_sentences(llm, prompt, n_sent, num_max_token):
    full_prompt = (
        f"{prompt}\n\n"
        "**Constraints:**\n"
        f"- Each sentence must not exceed {num_max_token} tokens.\n"
        "- Output each sentence on a new line.\n"
    )
    response = llm([HumanMessage(content=full_prompt)], max_tokens = n_sent * num_max_token).content.strip()
    sentences = response.split('\n')
    # Clean up numbering and dashes
    valid_sentences = [re.sub(r'^\s*[\d]+[\).]?\s*|- ', '', sentence).strip() for sentence in sentences if sentence.strip()]
    # Ensure exactly 'n' sentences by appending placeholders if necessary
    boosting_sentences = valid_sentences[:n_sent]
    boosting_sentences.extend(["[PLACEHOLDER: Generated sentence missing]"] * (n_sent - len(boosting_sentences)))
    return boosting_sentences



# ___________________________________________________________________________________________________________________________________________
def create_llm_initial_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent, num_max_token):
    prompt_template = (
        "You are an expert in information retrieval and search ranking. Your task is to optimize the Target "
        "document to rank higher for the given target query by generating highly relevant and seamlessly integrated sentences.\n\n"
        "**Guidelines:**\n"
        "1. Use only the information explicitly provided in the context below. Do NOT use any external knowledge.\n"
        "2. If you cannot find enough relevant facts in the context, do not invent or rely on external sources.\n"
        "3. Generated sentences that is aligned with Target Document with using key phrases of context "
        "and should naturally blend with the Target Document.\n"
        "4. The sentences should be persuasive, user-intent-driven, and should enhance the Target Document while maintaining its original style and tone.\n"
        "5. Ensure that the new sentences feel personalized to the Target Document, reinforcing its core message without making it appear artificial or disjointed.\n\n"
        "**Provided Information:**\n"
        "- Target Query: '{query}'\n"
        "- Target Document (currently ranked at position 1000): '{validator}'\n"
        "- Top-ranked documents (separated by '-'): '{context}'\n\n"
        "**Task:** Generate {n_sent} unique sentences that, when added to the Target Document, "
        "will improve its ranking for the target query.\n"
        "Ensure that each sentence is factually supported by the top-ranked documents and is naturally integrated into the Target Document."
    )
    prompt = prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        n_sent=n_sent
    )
    boosting_sentences = generate_boosting_sentences(llm, prompt, n_sent, num_max_token)
    return boosting_sentences
# ___________________________________________________________________________________________________________________________________________


def feedback_llm_without_sent_position(llm, target_query, validator_document, target_context, n_sent,
                                       already_generated_new_sentences_separated, num_max_token):
    """
    Provides iterative feedback to generate improved sentences that optimize the validator document's ranking,
    ensuring strict reliance on the given context and prohibiting the use of external knowledge.
    """
    feedback_prompt_template = (
        "You are an expert in search ranking optimization. Previously, sentences were generated to improve "
        "the ranking of the validator document, but they failed to make it rank #1 for the target query.\n\n"
        "**Guidelines:**\n"
        "1. Strictly use only the information from the top-ranked documents provided below. Do NOT use external knowledge.\n"
        "2. If the fact is not in the provided context, do not include it.\n"
        "3. Ensure each new sentence is factually accurate, persuasive, and relevant to the target query.\n\n"
        "**Provided Information:**\n"
        "- Target Query: '{query}'\n"
        "- Validator Document (currently ranked at position 1000): '{validator}'\n"
        "- Top-ranked documents (separated by '-'): '{context}'\n"
        "- Previously generated sentences (separated by '-'): '{previous_sentences}'\n\n"
        "**Task:** Generate {n_sent} new, unique, and highly optimized sentences that, when added to the validator "
        "document, will significantly improve its ranking to position 1 for the target query.\n"
        "Only include facts present in the top-ranked documents; otherwise, omit them.\n"
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
    validator_document_num_tokens = 50 #count_tokens(validator_document) + num_max_token
    # Feedback prompt template for the language model
    feedback_prompt_template = (
        "Your task is to optimize the validator document to improve its ranking to position 1 for the given query. "
        "You must strictly follow these constraints:\n\n"
        "1. **Target Query:** '{query}'\n"
        "2. **Validator Document (Current Rank: 1000):**\n'{validator}'\n"
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
        "- Maintain originality while improving the document’s ranking potential.\n\n"
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