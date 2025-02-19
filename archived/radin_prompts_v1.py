prompt_template = (
    "You are an expert in information retrieval and search ranking. Your ultimate goal is to improve the ranking of "
    "the **Target Document** for the given **Target Query** by generating highly relevant and seamlessly integrated "
    "sentences.\n\n"

    "INSTRUCTIONS:\n"
    "1. **Extraction Phase**\n"
    "   - **Buffer A (Context Key Phrases):** Extract the most relevant key phrases from the top-ranked documents "
    "(provided as a single string, with each document separated by a hyphen).\n"
    "   - **Buffer B (Target Document Key Phrases):** Extract the most important key phrases from the Target Document.\n\n"

    "2. **Generation Phase**\n"
    "   - Using **Buffer A** and **Buffer B**, create **{n_sent}** unique sentences that naturally integrate both sets "
    "of key phrases.\n"
    "   - These sentences should address the **Target Query**, preserve the Target Document's original style and tone, "
    "and be factually supported by the top-ranked documents.\n\n"

    "GUIDELINES:\n"
    "1. **Use Only the Provided Information**\n"
    "   - Avoid external knowledge or assumptions.\n"
    "   - If there is insufficient detail in the provided text, do not fabricate information.\n\n"

    "2. **Preserve Style and Tone**\n"
    "   - Your newly generated sentences should align with the voice, perspective, and style of the existing Target "
    "Document.\n\n"

    "3. **Factual Accuracy**\n"
    "   - All statements must be directly supported by the top-ranked documents; do not introduce unsupported claims.\n\n"

    "4. **User-Intent and Persuasion**\n"
    "   - Ensure the sentences speak to the user's intent in the Target Query, enhancing the Target Document's "
    "relevance and persuasiveness.\n\n"

    "5. **Seamless Integration**\n"
    "   - The new sentences should feel naturally embedded in the Target Document, reinforcing its core message without "
    "sounding forced or disjointed.\n\n"

    "PROVIDED INFORMATION:\n"
    "- **Target Query:** \"{query}\"\n"
    "- **Target Document:** \"{validator}\"\n"
    "- **Top-ranked Documents (separated by '-'):** \"{context}\"\n\n"

    "TASK:\n"
    "1. **Extract Key Phrases:**\n"
    "   - Generate **Buffer A** (key phrases from the top-ranked documents).\n"
    "   - Generate **Buffer B** (key phrases from the Target Document).\n\n"

    "2. **Generate {n_sent} Sentences:**\n"
    "   - Combine relevant phrases from **Buffer A** and **Buffer B** to craft {n_sent} new sentences.\n"
    "   - Each sentence must be factually supported by the top-ranked documents and align with the tone and structure "
    "of the Target Document.\n"
    "   - The result should help the Target Document rank higher for the specified **Target Query**.\n\n"
)

#___________________________________________________________________________________________________________________________________________________


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

#___________________________________________________________________________________________________________________________________________________

repherased_prompt_template = (
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

#___________________________________________________________________________________________________________________________________________________










#___________________________________________________________________________________________________________________________________________________