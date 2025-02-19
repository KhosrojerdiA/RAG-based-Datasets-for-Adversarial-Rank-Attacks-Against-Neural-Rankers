def create_llm_initial_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent):

    prompt_template = (
        "The target query is: '{query}'. "
        "The validator document is currently ranked at position 1000. Its content is: '{validator}'. "
        "The top-ranked documents are: '{context}' which is separated by -. "
        "Your task is to generate {n_sent} unique sentences that, when added it to the validator document, "
        "will make it rank 1 instead of 1000 for the query. "
        "Ensure each sentence is relevant, persuasive, and addresses the query context."
    )
    prompt = prompt_template.format(
        query=target_query, validator=validator_document, context=target_context, n_sent=n_sent)
    boosting_sentences = generate_boosting_sentences(llm, prompt, n_sent)
    return boosting_sentences