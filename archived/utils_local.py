import sys
import os


project_path = '/home/akhosrojerdi/Amin'
sys.path.append(project_path)
 

from pandas.io.parquet import read_parquet
from sentence_transformers import SentenceTransformer, util
import torch
from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import pandas as pd 
import os
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document 
from langchain.schema import HumanMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import bisect
from collections import defaultdict
import torch
from tqdm import tqdm
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from argparse import ArgumentParser
import numpy as np
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import nltk
import spacy
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.schema import HumanMessage
import pyarrow
import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
import language_tool_python
import re

nlp = spacy.load("en_core_web_sm")
from transformers import pipeline
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ___________________________________________________________________________________________________________________________________________

def candidate_generation(group):

    random_50_60 = group[(group["rank"] >= 50) & (group["rank"] <= 60)].sample(1)
    random_60_70 = group[(group["rank"] >= 60) & (group["rank"] <= 70)].sample(1)
    random_70_80 = group[(group["rank"] >= 70) & (group["rank"] <= 80)].sample(1)
    random_80_90 = group[(group["rank"] >= 80) & (group["rank"] <= 90)].sample(1)
    random_90_100 = group[(group["rank"] >= 90) & (group["rank"] <= 100)].sample(1)
    
    fixed_ranks = group[group["rank"].isin([996, 997, 998, 999, 1000])].copy()
    fixed_ranks["tag"] = "hard"
    
    random_rows = pd.concat([random_50_60, random_60_70, random_70_80, random_80_90, random_90_100]).copy()
    random_rows["tag"] = "easy"
    
    return pd.concat([random_rows, fixed_ranks])

# ___________________________________________________________________________________________________________________________________________
def context_generation(group, n):

    top_n_ranks = group[group["rank"].isin(list(range(1, n + 1)) )]

    return top_n_ranks
# ___________________________________________________________________________________________________________________________________________

def validator_generation(group):
    # Get the top `n` documents based on rank
    top_n_ranks = group[((group["rank"] == 1000))]
    return top_n_ranks
# ___________________________________________________________________________________________________________________________________________

def generate_training_data(candidate_docs_full, reranker):
    """
    Generates labeled training data by scoring candidate sentences appended to the last-ranked document
    using the CrossEncoder on GPU, with sentence tokenization using SpaCy.
    """
    training_data = []

    for query_id, group in tqdm(candidate_docs_full.groupby('query_id'), desc="Processing queries"):
        query = group['query'].iloc[0]
        
        # Identify the last-ranked document (rank == 1000)
        last_rank_doc = group[group['rank'] == 1000]['doc_content'].iloc[0]
        
        # Tokenize sentences for each document in the group using SpaCy
        context_sentences = []
        for doc_content in group['doc_content']:
            doc = nlp(doc_content)
            sentences = [sent.text for sent in doc.sents]
            context_sentences.extend(sentences)

        # Track the best sentence based on CrossEncoder scores
        best_sentence = None
        best_score = -float('inf')
        scores = []

        for sentence in context_sentences:
            # Combine the sentence with the last-ranked document
            modified_doc = sentence + " " + last_rank_doc
            inputs = [(query, modified_doc)]

            # Score the modified document using CrossEncoder on GPU
            try:
                score = reranker.predict(inputs)[0]
                scores.append((sentence, score))

                # Update the best sentence
                if score > best_score:
                    best_sentence = sentence
                    best_score = score
            except Exception as e:
                print(f"Error during scoring: {e}")

        # Label sentences: best sentence = 1, others = 0
        for sentence, score in scores:
            label = 1 if sentence == best_sentence else 0
            training_data.append((query, sentence, last_rank_doc, label))

    # Return as a DataFrame
    return pd.DataFrame(training_data, columns=["query", "sentence", "last_rank_doc", "label"])

# ___________________________________________________________________________________________________________________________________________


def tokenize_function(examples, tokenizer):
    tokenizer = tokenizer
    return tokenizer(
        [f"Query: {q} Sentence: {s} Last Rank Doc: {d}" for q, s, d in zip(examples['query'], examples['sentence'], examples['last_rank_doc'])],
        padding="max_length", truncation=True, return_tensors='pt'
    )
# ___________________________________________________________________________________________________________________________________________

    # PyTorch Dataset class
class RankDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        # Return CPU tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# ___________________________________________________________________________________________________________________________________________


class GPT2PPLScorer:
    def __init__(self, device):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def perplexity(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = self.tokenizer.batch_encode_plus(inputs, padding='longest', return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs['input_ids'][:, 1:].contiguous()
        shift_masks = inputs['attention_mask'][:, 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape[0], -1) * shift_masks
        loss = torch.sum(loss, -1) / torch.sum(shift_masks, -1)
        ppl = torch.exp(loss).detach().cpu().numpy().tolist()
        return ppl
# ___________________________________________________________________________________________________________________________________________
# Define the GPT2CohScorer class
class GPT2CohScorer:
    def __init__(self, device):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(device)
        self.model.eval()
        self.device = device

    def compute_coherence(self, front, behind):
        combined_text = f"{front} {self.tokenizer.eos_token} {behind}"
        encoded_input = self.tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model(**encoded_input, labels=encoded_input['input_ids'])
        loss = output.loss.item()
        return -loss  # Negative loss as coherence score

# ___________________________________________________________________________________________________________________________________________
class RoBERTaCoLAScorer:
    def __init__(self, device):
        self.model_name = "textattack/roberta-base-CoLA"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def classify_acceptability(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        return "acceptable" if prediction == 1 else "unacceptable"

# ___________________________________________________________________________________________________________________________________________


class DocumentEvaluator:
    """Evaluates documents based on relevance, imperceptibility, and convincingness."""
    def __init__(self, device):
        self.device = device
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.readability_scorer = GPT2PPLScorer(device)
        
    def relevance_score(self, query, document):
        """Compute cosine similarity between query and document."""
        embeddings = self.embedding_model.encode([query, document], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return similarity.item()
    
    def imperceptibility_score(self, original_doc, modified_doc):
        """Compute BLEU score as a measure of imperceptibility."""
        original_tokens = original_doc.split()
        modified_tokens = modified_doc.split()
        bleu_score = sentence_bleu([original_tokens], modified_tokens)
        return bleu_score
    
    def convincing_score(self, document):
        """Use perplexity to estimate readability and coherence."""
        return self.readability_scorer.perplexity(document)[0]

    def perplexity_score(self, document):
        """Return the perplexity of the document explicitly."""
        return self.readability_scorer.perplexity(document)[0]
    
    def evaluate_document(self, query, original_doc, modified_doc):
        """Evaluate document for relevance, imperceptibility, convincingness, and perplexity."""
        relevance = self.relevance_score(query, modified_doc)
        imperceptibility = self.imperceptibility_score(original_doc, modified_doc)
        convincingness = self.convincing_score(modified_doc)
        perplexity = self.perplexity_score(modified_doc)
        return relevance, imperceptibility, convincingness, perplexity
# ___________________________________________________________________________________________________________________________________________

def compute_document_scores(df, device):
    """Compute scores for all documents in the DataFrame."""
    evaluator = DocumentEvaluator(device)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating documents"):
        query_id = row['query_id']
        doc_id = row['doc_id']
        query = row['query']
        original_doc = row['old_doc_content']
        modified_doc = row['doc_content']

        relevance, imperceptibility, convincingness, perplexity = evaluator.evaluate_document(query, original_doc, modified_doc)
        results.append({
            'query_id': query_id,
            'doc_id': doc_id,
            'relevance': relevance,
            'imperceptibility': imperceptibility,
            'convincingness': convincingness,
            'perplexity': perplexity
        })

    result_df = pd.DataFrame(results)
    return result_df

# ___________________________________________________________________________________________________________________________________________

def create_candidate_docs_full_for_cross_encoder(cross_reranker_sample_1000_query, collection):

    candidate_docs_full = cross_reranker_sample_1000_query[['query_id', 'query', 'doc_id']]  
    candidate_docs_full = candidate_docs_full.merge(collection, on="doc_id", how="inner")
    candidate_docs_full = candidate_docs_full[['query_id', 'query', 'doc_id', 'doc_content']]
    #print(candidate_docs_full)
    return candidate_docs_full

# ___________________________________________________________________________________________________________________________________________

def create_candidate_docs_full(cross_reranker_sample_1000_query, collection):

    candidate_docs_full = cross_reranker_sample_1000_query[['query_id', 'query', 'doc_id', 'score' ,'rank']]  
    candidate_docs_full = candidate_docs_full.merge(collection, on="doc_id", how="inner")
    candidate_docs_full = candidate_docs_full[['query_id', 'query', 'doc_id', 'doc_content','score', 'rank']]
    #print(candidate_docs_full)
    return candidate_docs_full

# ___________________________________________________________________________________________________________________________________________

def create_target_query(candidate_docs_full_query):

    target_query = candidate_docs_full_query[['query']]
    target_query= target_query.drop_duplicates(subset=['query']).reset_index(drop=True)
    target_query = target_query.loc[0, 'query']

    return target_query
# ___________________________________________________________________________________________________________________________________________

def create_validator_document_info(candidate_docs_full_query, target_doc_rank):

    validator_document_id = candidate_docs_full_query[(candidate_docs_full_query['rank'] == target_doc_rank)]['doc_id'].values[0]
    validator_document = candidate_docs_full_query[candidate_docs_full_query['doc_id'] == validator_document_id]['doc_content'].values[0]

    return validator_document_id, validator_document
# ___________________________________________________________________________________________________________________________________________

def create_target_document_rank(candidate_docs_full_query, validator_document_id):

    target_document_rank = candidate_docs_full_query[(candidate_docs_full_query['doc_id'] == validator_document_id)]['rank'].values[0]

    return int(target_document_rank)

# ___________________________________________________________________________________________________________________________________________

def create_target_context(candidate_docs_full_query, top_n_context):

    context_docs = candidate_docs_full_query
    context_docs_df = context_generation(context_docs, top_n_context)
    target_context = ' - '.join(context_docs_df['doc_content'].astype(str))

    return target_context


# ___________________________________________________________________________________________________________________________________________

def create_llm_initial_response_with_sent_position(llm, target_query, validator_document, target_context, n_sent, llm_sent_position):

    prompt_template = (
        "The target query is: '{query}'. "
        "The validator document is currently ranked at position 1000. Its content is: '{validator}'. "
        "The top-ranked documents are: '{context}' which is separated by -. "
        "Your task is to generate {n_sent} unique sentences that, when placed '{llm_sent_position}' of the validator document, "
        "will make it rank 1 instead of 1000 for the query. "
        "Ensure each sentence is relevant, persuasive, and addresses the query context."
    )
    prompt = prompt_template.format(
        query=target_query, validator=validator_document, context=target_context, n_sent=n_sent, llm_sent_position=llm_sent_position
    )
    boosting_sentences = generate_boosting_sentences(llm, prompt, n_sent)
    return boosting_sentences
# ___________________________________________________________________________________________________________________________________________


def feedback_llm_with_sent_position(llm, target_query, validator_document, target_context, n_sent, sent_position, already_generated_new_sentences_separated):

    # Feedback prompt template with enhanced clarity and specificity
    feedback_prompt_template = (
        "The target query is: '{query}'.\n"
        "The validator document is currently ranked at position 1000. Its content is:\n"
        "'{validator}'.\n"
        "The top-ranked documents for the query are as follows (separated by '-'): '{context}'.\n\n"
        "Previously, the following sentences were generated (separated by '-'): '{previous_sentences}', "
        "to be added '{sent_position}' of the validator document. "
        "However, these sentences failed to improve the ranking of the validator document to position 1 for the query.\n\n"
        "Your task:\n"
        "1. Analyze the target query, validator document, and the context provided.\n"
        "2. Generate {n_sent} new, unique, and highly optimized sentences.\n"
        "3. These sentences should be highly relevant to the query, persuasive, and address the context of the top-ranked documents.\n"
        "4. Ensure these sentences, when added '{sent_position}' of the validator document, "
        "significantly improve its ranking to position 1 for the target query."
    )
    
    # Format the feedback prompt with the provided parameters
    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        previous_sentences=already_generated_new_sentences_separated,
        n_sent=n_sent,
        sent_position=sent_position
    )
    
    # Generate improved sentences using the feedback prompt
    improved_sentences = generate_boosting_sentences(llm, feedback_prompt, n_sent)

    return improved_sentences
# ___________________________________________________________________________________________________________________________________________

# Function to generate sentences

import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def generate_boosting_sentences(llm, prompt_template, prompt_vars, n):

    # Create an instance of LLMChain with a valid PromptTemplate
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(prompt_vars)

    # Handle different response formats
    if isinstance(response, str):
        response_text = response.strip()
    elif hasattr(response, "generations"):
        response_text = response.generations[0][0].text.strip()
    else:
        raise ValueError("Unexpected response format from LLM.")

    sentences = response_text.split('\n')
    valid_sentences = [re.sub(r'^\s*[\d]+[\).]?\s*|- ', '', sentence).strip() for sentence in sentences if sentence.strip()]

    boosting_sentences = valid_sentences[:n]
    while len(boosting_sentences) < n:
        boosting_sentences.append("[PLACEHOLDER: Generated sentence missing]")
    
    return boosting_sentences

# ___________________________________________________________________________________________________________________________________________

def create_llm_initial_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent):

    prompt_template = PromptTemplate(
        input_variables=["target_query", "validator_document", "target_context", "n_sent"],
        template=(
            "You are an expert in information retrieval and search ranking. Your task is to optimize the validator "
            "document to rank higher for the given target query.\n\n"
            "**Guidelines:**\n"
            "1. Use only the information explicitly provided in the context below. Do NOT use any external knowledge.\n"
            "2. If you cannot find enough relevant facts in the context, do not invent or rely on external sources.\n"
            "3. Each generated sentence must be highly relevant and contextually aligned with the text of the top-ranked documents.\n"
            "4. The sentences should be persuasive and aligned with the user intent behind the target query.\n\n"
            "**Provided Information:**\n"
            "- Target Query: '{target_query}'\n"
            "- Validator Document (currently ranked at position 1000): '{validator_document}'\n"
            "- Top-ranked documents (separated by '-'): '{target_context}'\n\n"
            "**Task:** Generate {n_sent} unique sentences that, when added to the validator document, "
            "will improve its ranking for the target query.\n"
            "Only use facts that appear in the top-ranked documents. If the context does not support a fact, do NOT include it."
            "Only return the generated sentences without any additional text."
        )
    )

    # Pass a dictionary with variables, not formatted strings
    prompt_vars = {
        "target_query": target_query,
        "validator_document": validator_document,
        "target_context": target_context,
        "n_sent": n_sent
    }

    return generate_boosting_sentences(llm, prompt_template, prompt_vars, n_sent)

# ___________________________________________________________________________________________________________________________________________


def feedback_llm_without_sent_position(llm, target_query, validator_document, target_context, n_sent, already_generated_new_sentences_separated):

    # Define a valid PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["target_query", "validator_document", "target_context", "already_generated_new_sentences_separated", "n_sent"],
        template=(
        "You are an expert in search ranking optimization. Previously, sentences were generated to improve "
        "the ranking of the validator document, but they failed to make it rank #1 for the target query.\n\n"
        "**Guidelines:**\n"
        "1. Strictly use only the information from the top-ranked documents provided below. Do NOT use external knowledge.\n"
        "2. If the fact is not in the provided context, do not include it.\n"
        "3. Ensure each new sentence is factually accurate, persuasive, and relevant to the target query.\n\n"
        "**Provided Information:**\n"
        "- Target Query: '{target_query}'\n"
        "- Validator Document (currently ranked at position 1000): '{validator_document}'\n"
        "- Top-ranked documents (separated by '-'): '{target_context}'\n"
        "- Previously generated sentences (separated by '-'): '{already_generated_new_sentences_separated}'\n\n"
        "**Task:** Generate {n_sent} new, unique, and highly optimized sentences that, when added to the validator "
        "document, will significantly improve its ranking to position 1 for the target query.\n"
        "Only include facts present in the top-ranked documents; otherwise, omit them.\n"
                )
                                    )


    prompt_vars = {
    "target_query": target_query,
    "validator_document": validator_document,
    "target_context": target_context,
    "already_generated_new_sentences_separated":already_generated_new_sentences_separated,
    "n_sent": n_sent
    }

    improved_sentences = generate_boosting_sentences(llm, prompt_template, prompt_vars, n_sent)

    return improved_sentences


# ___________________________________________________________________________________________________________________________________________

def llm_with_best_sent(llm, target_query, validator_document, best_sentence):
    """
    Optimizes the validator document by integrating the best identified sentence while ensuring the new document
    length does not increase or decrease by more than 5%.
    """

    # Calculate the allowed length range (±5%)
    original_length = len(validator_document.split())
    min_length = int(original_length * 0.95)
    max_length = int(original_length * 1.05)

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
    response = llm.invoke([HumanMessage(content=feedback_prompt)]).content.strip()

    return response

# ___________________________________________________________________________________________________________________________________________

def generate_sent_position_list(validator_document):
    # Split the document into sentences
    sentences = validator_document.split('. ')
    num_sentences = len([s for s in sentences if s.strip()])  # Count non-empty sentences

    # Initialize the list with "at the beginning"
    sent_position_list = ["at the beginning"]
    
    # Add positions dynamically based on the number of sentences
    if num_sentences > 1:
        sent_position_list.append("after first sentence")
    if num_sentences > 2:
        sent_position_list.append("after second sentence")
    if num_sentences > 3:
        sent_position_list.append("after third sentence")
    if num_sentences > 4:
        sent_position_list.append("after fourth sentence")
    
    # Always include "at the end"
    sent_position_list.append("at the end")
    
    return sent_position_list


# ___________________________________________________________________________________________________________________________________________


def sent_position_function(sent, sent_position, validator_document):
    # Split the document into sentences
    sentences = validator_document.split('. ')
    
    if sent_position == "at the beginning":
        new_validator_document = sent + ". " + validator_document
    elif sent_position == "after first sentence" and len(sentences) > 1:
        new_validator_document = sentences[0] + ". " + sent + ". " + ". ".join(sentences[1:])
    elif sent_position == "after second sentence" and len(sentences) > 2:
        new_validator_document = ". ".join(sentences[:2]) + ". " + sent + ". " + ". ".join(sentences[2:])
    elif sent_position == "after third sentence" and len(sentences) > 3:
        new_validator_document = ". ".join(sentences[:3]) + ". " + sent + ". " + ". ".join(sentences[3:])
    elif sent_position == "after fourth sentence" and len(sentences) > 4:
        new_validator_document = ". ".join(sentences[:4]) + ". " + sent + ". " + ". ".join(sentences[4:])
    elif sent_position == "at the end":
        new_validator_document = validator_document + " " + sent
    else:
        # Default case: if position is invalid or there are not enough sentences
        new_validator_document = validator_document + " " + sent

    return new_validator_document

# ___________________________________________________________________________________________________________________________________________



def doc_content_replacement_with_llm_sent(candidate_docs_full_query_loop, validator_document_id, new_validator_document):
    
    #cross_reranker for that query_id only
    modified_candidate_docs_full_query_loop = candidate_docs_full_query_loop.copy()

    # Replace content
    modified_candidate_docs_full_query_loop.loc[modified_candidate_docs_full_query_loop['doc_id'] == validator_document_id, 'doc_content'] = new_validator_document

    return modified_candidate_docs_full_query_loop

# ___________________________________________________________________________________________________________________________________________

def calculate_and_add_perplexity_coh_score(data):
    # Initialize scorers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppl_scorer = GPT2PPLScorer(device)
    coh_scorer = GPT2CohScorer(device)
    tool = language_tool_python.LanguageTool('en-US')
    cola_scorer = RoBERTaCoLAScorer(device)

    # Lists to store scores
    perplexity_scores = []
    coherence_scores = []
    grammar_issues = []
    cola_scores = []

    # Iterate through the DataFrame rows
    for _, row in data.iterrows():
        query = row['query']
        doc_content = row['doc_content']

        # Calculate perplexity
        ppl_score = ppl_scorer.perplexity(doc_content)[0]  # Get the first score since it's a list
        perplexity_scores.append(ppl_score)

        # Calculate coherence
        coh_score = coh_scorer.compute_coherence(query, doc_content)
        coherence_scores.append(coh_score)

        # Check grammar
        matches = tool.check(doc_content)
        grammar_issues.append(len(matches))  # Store the number of grammar issues

        # Classify linguistic acceptability
        cola_score = cola_scorer.classify_acceptability(doc_content)
        cola_scores.append(cola_score)

    # Add scores to DataFrame
    data['ppl_score'] = perplexity_scores
    data['coh_score'] = coherence_scores
    data['grammar_issues'] = grammar_issues
    data['cola_score'] = cola_scores

    return data

# ___________________________________________________________________________________________________________________________________________

def calculate_avg_top_n_perplexity_coh_cola_score(top_n_df):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ppl_scorer = GPT2PPLScorer(device)
    coh_scorer = GPT2CohScorer(device)
    tool = language_tool_python.LanguageTool('en-US')
    cola_scorer = RoBERTaCoLAScorer(device)

    # Lists to store scores
    top_n_perplexities = []
    top_n_coherence_scores = []
    top_n_grammar_issues = []
    top_n_cola_scores = []

    # Iterate through the DataFrame rows
    for index, row in top_n_df.iterrows():
        query = row['query']
        doc_content = row['doc_content']

        # Compute perplexity
        perplexity_score = ppl_scorer.perplexity(doc_content)[0]
        top_n_perplexities.append(perplexity_score)

        # Compute coherence score
        coherence_score = coh_scorer.compute_coherence(query, doc_content)
        top_n_coherence_scores.append(coherence_score)

        # Check grammar
        matches = tool.check(doc_content)
        top_n_grammar_issues.append(len(matches))


    # Compute the average scores for the top n rows
    avg_top_n_perplexity = sum(top_n_perplexities) / len(top_n_perplexities)
    avg_top_n_coh_score = sum(top_n_coherence_scores) / len(top_n_coherence_scores)
    avg_top_n_grammar_issues = sum(top_n_grammar_issues) / len(top_n_grammar_issues)

    return avg_top_n_perplexity, avg_top_n_coh_score, avg_top_n_grammar_issues

# ___________________________________________________________________________________________________________________________________________

def calculate_and_add_perplexity_coh_gramm_cola_score_modified_document(df_dataset_per_query_with_score_new):
    # Initialize scorers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppl_scorer = GPT2PPLScorer(device)
    coh_scorer = GPT2CohScorer(device)
    tool = language_tool_python.LanguageTool('en-US')
    cola_scorer = RoBERTaCoLAScorer(device)

    # Lists to store scores
    perplexities = []
    coherence_scores = []
    grammar_issues = []
    cola_scores = []

    for index, row in df_dataset_per_query_with_score_new.iterrows():
        query = row['query']
        doc_content = row['new_doc_content']

        # Compute perplexity
        perplexity_score = ppl_scorer.perplexity(doc_content)[0]
        perplexities.append(perplexity_score)

        # Compute coherence score
        coherence_score = coh_scorer.compute_coherence(query, doc_content)
        coherence_scores.append(coherence_score)

        # Check grammar
        matches = tool.check(doc_content)
        grammar_issues.append(len(matches))  # Store the number of grammar issues

        # Classify linguistic acceptability
        cola_score = cola_scorer.classify_acceptability(doc_content)
        cola_scores.append(cola_score)

    # Add the computed scores to the dataframe
    df_dataset_per_query_with_score_new['new_perplexity_score'] = perplexities
    df_dataset_per_query_with_score_new['new_coh_score'] = coherence_scores
    df_dataset_per_query_with_score_new['new_grammar_issues'] = grammar_issues
    df_dataset_per_query_with_score_new['new_cola_score'] = cola_scores

    return df_dataset_per_query_with_score_new

# ___________________________________________________________________________________________________________________________________________

def perplexity_coh_score(cross_reranker_sample_1000_query, collection, target_query_id):

    #Calculate the avg of perpelixty and coh_score
    target_document_cross_reranker_sample_1000_query = cross_reranker_sample_1000_query[cross_reranker_sample_1000_query['query_id'] == target_query_id]
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query.merge(collection, on="doc_id", how="inner")
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query[['query_id', 'query', 'doc_id', 'doc_content', 'rank', 'distance']]

    ppl_coh_target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query
    ppl_coh_target_document_cross_reranker_sample_1000_query = calculate_and_add_perplexity_coh_score(ppl_coh_target_document_cross_reranker_sample_1000_query)

    return ppl_coh_target_document_cross_reranker_sample_1000_query



# ___________________________________________________________________________________________________________________________________________

def avg_top_n_perplexity_coh_cola_score(candidate_docs_full_query, top_n_context):

    #Calculate the avg of perpelixty and coh_score of top n rank
    top_n_df  = candidate_docs_full_query.head(top_n_context)
    avg_perplexity, avg_coh_score, avg_top_n_grammar_issues = calculate_avg_top_n_perplexity_coh_cola_score(top_n_df)

    return avg_perplexity, avg_coh_score, avg_top_n_grammar_issues

# ___________________________________________________________________________________________________________________________________________

def rerank_modified_target_query_cross_reranker(model, modified_target_query_cross_reranker, target_query_id, target_query, validator_document_id):
    
    #query_doc_pairs = modified_target_query_cross_reranker[modified_target_query_cross_reranker['doc_id'] == validator_document_id]
    #query_doc_pairs = list(zip(query_doc_pairs['query'], query_doc_pairs['doc_content']))
    #call the function to get the score, it will return a df with score and rank
    #we can append this df to other and rerank again with new distance and new rank

    # Check for CUDA availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move data to GPU
    modified_target_query_cross_reranker = modified_target_query_cross_reranker.to(device)
    target_query_id = torch.tensor(target_query_id, device=device)
    target_query = target_query.to(device)
    validator_document_id = torch.tensor(validator_document_id, device=device)
    
    # Initialize list to collect reranked results
    target_reranked_run = []
    
    # Iterate over each unique query in the dataset
    for query_id, group in tqdm(modified_target_query_cross_reranker.groupby('query_id'), desc="Reranking Queries"):
        # Create document pairs with the target query for scoring
        list_of_docs = [(target_query, doc_content) for doc_content in group['doc_content']]
        
        # Compute scores using the CrossEncoder model
        with torch.no_grad():
            group_scores = model.predict(list_of_docs).tolist()
        
        # Add scores to the group and sort by score
        group = group.copy()
        group['score'] = group_scores
        group = group.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        # Assign new ranking and distance metrics
        group['new_rank'] = group.index + 1
        group['new_distance'] = group['score']
        
        # Append the reranked group to the result list
        target_reranked_run.append(group)

    # Combine all reranked groups into a single DataFrame
    target_query_reranked_run_df = pd.concat(target_reranked_run, ignore_index=True)
    target_query_reranked_run_df = target_query_reranked_run_df[['query_id', 'doc_id', 'rank', 'distance', 'new_rank', 'new_distance']]
    
    # Retrieve the new rank for the validator document
    new_calculated_rank = target_query_reranked_run_df.loc[
        (target_query_reranked_run_df['query_id'] == target_query_id) & 
        (target_query_reranked_run_df['doc_id'] == validator_document_id), 'new_rank'
    ].iloc[0]

    return target_query_reranked_run_df, new_calculated_rank


# ___________________________________________________________________________________________________________________________________________

def dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query, sent_position):

    # Filter rows where 'new_sent_position' is equal to 'sent_position'
    filtered_df = df_dataset_per_query[df_dataset_per_query['new_sent_position'] == sent_position]
    
    # Check if any row in the filtered dataframe has 'new_rank' equal to 1
    return (filtered_df['new_rank'] <= 10).any()

# ___________________________________________________________________________________________________________________________________________

def dataset_per_query_has_rank_below_n_without_sent_position(df_dataset_per_query):

    filtered_df = df_dataset_per_query
    # Check if any row in the filtered dataframe has 'new_rank' equal to 1
    return (filtered_df['new_rank'] <= 10).any()

# ___________________________________________________________________________________________________________________________________________

def feedback_generated_sentences_per_query_rank_below_10_separated_with_sent_position(df_dataset_per_query, sent_position, rank_threshold):

    # Filter rows where 'new_sent_position' is equal to 'sent_position'
    filtered_df = df_dataset_per_query[
        (df_dataset_per_query['new_sent_position'] == sent_position) &
        (df_dataset_per_query['new_rank'] <= rank_threshold)  # Include only rows with 'rank' <= rank_threshold (like 10)
    ]
    
    # Join all 'new_sent' values in the filtered dataframe separated by '-'
    return ' - '.join(filtered_df['new_sent'].astype(str))

# ___________________________________________________________________________________________________________________________________________

def feedback_generated_sentences_per_query_rank_below_5_separated_without_sent_position(df_dataset_per_query): #No Duplicate
    # Filter rows where new_rank <= 5
    filtered_df = df_dataset_per_query[df_dataset_per_query['new_rank'] <= 5]
    
    # Remove duplicates in the 'new_sent' column
    unique_new_sent = filtered_df['new_sent'].drop_duplicates()
    
    # Join the unique 'new_sent' values with '-'
    return '-'.join(unique_new_sent.astype(str))

# ___________________________________________________________________________________________________________________________________________


def create_per_query_dataset(df_dataset_per_query, validator_document_id, validator_document, target_document_rank, model, 
                                boosting_sentences, candidate_docs_full_query_loop, sent_position, target_context):

    # Create dataset_per_query (different position, repalce content, rerank and append to the per query dataset)

    for sent in boosting_sentences: #different new sent

        # new content is created with placing sent in sent_position
        new_validator_document = sent_position_function(sent, sent_position, validator_document) 
        #replace content in df
        modified_candidate_docs_full_query_loop  = doc_content_replacement_with_llm_sent(candidate_docs_full_query_loop, validator_document_id, new_validator_document) 
        #prepare df for modified document rerank 
        modified_candidate_docs_full_query_for_reranker = modified_candidate_docs_full_query_loop[modified_candidate_docs_full_query_loop['doc_id'] == validator_document_id]
        #Get the new score for modifed document
        new_score = cross_encoder_rank_per_query(modified_candidate_docs_full_query_for_reranker, model) 
        #append to the dataset for this query
        df_dataset_per_query = append_to_df_dataset_per_query(candidate_docs_full_query_loop, validator_document_id, new_score, target_document_rank, 
                                                              sent, sent_position, new_validator_document, df_dataset_per_query, target_context)
        
    return df_dataset_per_query
# ___________________________________________________________________________________________________________________________________________

def append_to_df_dataset_per_query(candidate_docs_full_query_loop, validator_document_id, new_score, target_document_rank, 
                                   sent, sent_position, new_validator_document, df_dataset_per_query, target_context):

    # 'query_id', 'query', 'doc_id', 'doc_content', 'score', 'rank'
    full_target_reranked_run_df = candidate_docs_full_query_loop
    # new_score = score
    full_target_reranked_run_df['new_score'] = full_target_reranked_run_df['score']
    #replace target document new_score with new_score
    full_target_reranked_run_df.loc[full_target_reranked_run_df['doc_id'] == validator_document_id, 'new_score'] = new_score
    # Rerank with new_score per query_id
    full_target_reranked_run_df['new_rank'] = full_target_reranked_run_df.groupby('query_id')['new_score'].rank(ascending=False, method='first')

    full_target_reranked_run_df = full_target_reranked_run_df[full_target_reranked_run_df['doc_id'] == validator_document_id]
    
    full_target_reranked_run_df['doc_context'] = target_context
    full_target_reranked_run_df['new_sent'] = sent
    full_target_reranked_run_df['new_sent_position'] = sent_position
    full_target_reranked_run_df['new_doc_content'] = new_validator_document
    

    full_target_reranked_run_df = full_target_reranked_run_df[['query_id', 'query', 'doc_id','doc_content', 'score','rank', 'doc_context', 'new_sent' , 'new_sent_position' ,'new_doc_content', 'new_score', 'new_rank']]

    df_dataset_per_query = pd.concat([df_dataset_per_query, full_target_reranked_run_df], ignore_index=True)

    return df_dataset_per_query


# ___________________________________________________________________________________________________________________________________________

def cross_encoder_rank_per_query(modified_candidate_docs_full_query_for_reranker, model):

    query_doc_pairs = list(zip(modified_candidate_docs_full_query_for_reranker['query'], modified_candidate_docs_full_query_for_reranker['doc_content']))
    scores = model.predict(query_doc_pairs, convert_to_numpy=True, show_progress_bar=True)
    
    return scores

# ___________________________________________________________________________________________________________________________________________

def cross_encoder_rank_documents(candidate_docs_full, model):

    query_doc_pairs = list(zip(candidate_docs_full['query'], candidate_docs_full['doc_content']))
    
    # Compute relevance scores
    scores = model.predict(query_doc_pairs, convert_to_numpy=True, show_progress_bar=True)
    
    # Add scores to DataFrame
    candidate_docs_full['score'] = scores
    
    # Rank documents per query
    candidate_docs_full['rank'] = candidate_docs_full.groupby('query_id')['score'].rank(ascending=False, method='first')
    ranked_candidate_docs_full = candidate_docs_full
    
    return ranked_candidate_docs_full

# ___________________________________________________________________________________________________________________________________________

def re_org_df_dataset_per_query_with_score(df_dataset_per_query_with_score_new, avg_perplexity, avg_coh_score, avg_top_n_grammar_issues):

    df_dataset_per_query_with_score_new['top_n_avg_perplexity_score'] = avg_perplexity
    df_dataset_per_query_with_score_new['top_n_avg_coh_score'] = avg_coh_score
    df_dataset_per_query_with_score_new['top_n_avg_grammar_issues'] = avg_top_n_grammar_issues
    df_dataset_per_query_with_score_new = calculate_and_add_perplexity_coh_gramm_cola_score_modified_document(df_dataset_per_query_with_score_new)
    df_dataset_per_query_with_score_new = df_dataset_per_query_with_score_new[['query_id', 'query', 'doc_id','doc_content', 'top_n_avg_perplexity_score', 'top_n_avg_coh_score', 'top_n_avg_grammar_issues', 
                                                                               'score','rank', 'doc_context',
                                                                               'new_sent' , 'new_sent_position' ,'new_doc_content','new_perplexity_score', 'new_coh_score', 'new_grammar_issues', 'new_cola_score', 
                                                                               'new_score', 'new_rank']]
    
    return df_dataset_per_query_with_score_new

# ___________________________________________________________________________________________________________________________________________

def remove_highest_new_rank_rows(df_dataset_per_query, sent_position, n_sent):

    # Filter rows where new_sent_position is equal to sent_position
    filtered_df = df_dataset_per_query[df_dataset_per_query['new_sent_position'] == sent_position]
    # Sort by new_rank in descending order to get the highest new_rank values
    filtered_df = filtered_df.sort_values(by='new_rank', ascending=False)
    # Get the indices of the top n_sent rows with the highest new_rank
    indices_to_remove = filtered_df.head(n_sent).index
    # Drop these rows from the original dataframe
    updated_df = df_dataset_per_query.drop(indices_to_remove)
    updated_df = updated_df.sort_values(by='new_sent_position')
    
    return updated_df

# ___________________________________________________________________________________________________________________________________________

def get_best_new_sent(df):

    return df.loc[df['new_rank'].idxmin(), 'new_sent']

# ___________________________________________________________________________________________________________________________________________

def create_best_sent_dataset(df_dataset_per_query, validator_document_id, rephrased_validator_document, target_document_rank, model, 
                              candidate_docs_full_query_loop, sent_position, target_context):


    new_validator_document = rephrased_validator_document
    #replace content in df
    modified_candidate_docs_full_query_loop  = doc_content_replacement_with_llm_sent(candidate_docs_full_query_loop, validator_document_id, new_validator_document) 
    #prepare df for modified document rerank 
    modified_candidate_docs_full_query_for_reranker = modified_candidate_docs_full_query_loop[modified_candidate_docs_full_query_loop['doc_id'] == validator_document_id]
    #Get the new score for modifed document
    new_score = cross_encoder_rank_per_query(modified_candidate_docs_full_query_for_reranker, model) 
    #append to the dataset for this query
    df_dataset_per_query = append_to_df_dataset_per_query(candidate_docs_full_query_loop, validator_document_id, new_score, target_document_rank, 
                                                            "rephrased", sent_position, new_validator_document, df_dataset_per_query, target_context)
        
    return df_dataset_per_query

# ___________________________________________________________________________________________________________________________________________

def target_doc_content_replacement(cross_reranker_sample_1000_query, target_query_id, validator_document_id, collection, new_validator_document):
    
    #cross_reranker for that query_id only
    target_document_cross_reranker_sample_1000_query = cross_reranker_sample_1000_query[cross_reranker_sample_1000_query['query_id'] == target_query_id]
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query.merge(collection, on="doc_id", how="inner")
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query[['query_id', 'query', 'doc_id', 'doc_content', 'rank', 'distance']]

    # Replace content
    target_document_cross_reranker_sample_1000_query.loc[target_document_cross_reranker_sample_1000_query['doc_id'] == validator_document_id, 'doc_content'] = new_validator_document

    return target_document_cross_reranker_sample_1000_query 

# ___________________________________________________________________________________________________________________________________________

def rerank_modified_document(model, target_document_cross_reranker_sample_1000_query, target_query_id, target_query, validator_document_id):

    target_reranked_run = []
    for query_id, group in tqdm(target_document_cross_reranker_sample_1000_query.groupby('query_id')):
        query = target_query  
        list_of_docs = [(query, doc_content) for doc_content in group['doc_content']]
        group_scores = model.predict(list_of_docs).tolist()
        group['score'] = group_scores
        group = group.sort_values(by='score', ascending=False).reset_index(drop=True)
        group['new_rank'] = group.index + 1  
        group['new_distance'] = group['score']  
        target_reranked_run.append(group)

    target_reranked_run_df = pd.concat(target_reranked_run, ignore_index=True)
    target_reranked_run_df = target_reranked_run_df[['query_id', 'doc_id', 'rank', 'distance', 'new_rank', 'new_distance']]

    #target_reranked_run_df[(target_reranked_run_df['query_id'] == target_query_id) & (target_reranked_run_df['doc_id'] == validator_document_id)]
    new_calculated_rank = target_reranked_run_df[(target_reranked_run_df['query_id'] == target_query_id) & (target_reranked_run_df['doc_id'] == validator_document_id)]['new_rank'].values[0]
    #print(new_calculated_rank)
    #print(target_reranked_run_df)
    return target_reranked_run_df, new_calculated_rank

# ___________________________________________________________________________________________________________________________________________