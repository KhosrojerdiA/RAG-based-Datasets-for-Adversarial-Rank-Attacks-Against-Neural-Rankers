import sys
import os
os.system("nvidia-smi")

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"


project_path = './'
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
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.schema import HumanMessage
import pyarrow
import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline

import language_tool_python

nlp = spacy.load("en_core_web_sm")
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




def validator_generation(group):
    # Get the top `n` documents based on rank
    top_n_ranks = group[((group["rank"] == 1000))]
    return top_n_ranks


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




def tokenize_function(examples, tokenizer):
    tokenizer = tokenizer
    return tokenizer(
        [f"Query: {q} Sentence: {s} Last Rank Doc: {d}" for q, s, d in zip(examples['query'], examples['sentence'], examples['last_rank_doc'])],
        padding="max_length", truncation=True, return_tensors='pt'
    )

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


# Function for inference

def predict_best_sentence(query, context_sentences, last_rank_doc, model_path, device):
    # Load the saved model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Send model to GPU

    # Tokenize inputs and send to GPU
    inputs = tokenizer(
        [f"Query: {query} Sentence: {s} Last Rank Doc: {last_rank_doc}" for s in context_sentences],
        return_tensors="pt", truncation=True, padding="max_length"
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=-1)[:, 1]  # Probability of label 1
    
    # Find the best sentence
    best_idx = torch.argmax(scores).item()
    return context_sentences[best_idx]



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


# Function to generate sentences
def generate_boosting_sentences(llm, prompt):
    # Wrap the prompt in a HumanMessage and get a single response

    #response = llm([HumanMessage(content=prompt)]).content.strip()
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # Split the response into individual sentences
    sentences = response.split('\n')
    # Extract and clean the first n sentences
    clean_sentences = [sentence.lstrip('12345. ').strip() for sentence in sentences[:5]]
    return clean_sentences




def create_candidate_docs_full(cross_reranker_sample_1000_query, collection):

    candidate_docs_full = cross_reranker_sample_1000_query[['query_id', 'query', 'doc_id', 'rank' ,'distance']]  
    candidate_docs_full = candidate_docs_full.merge(collection, on="doc_id", how="inner")
    candidate_docs_full = candidate_docs_full[['query_id', 'query', 'doc_id', 'doc_content','rank', 'distance']]
    #print(candidate_docs_full)
    return candidate_docs_full



def create_target_query(candidate_docs_full, target_query_id):
    target_query = candidate_docs_full[candidate_docs_full['query_id'] == target_query_id][['query']]
    target_query= target_query.drop_duplicates(subset=['query']).reset_index(drop=True)
    target_query = target_query.loc[0, 'query']
    #print(target_query)
    return target_query

def create_validator_document_info(candidate_docs_full, target_query_id):
    validator_document_id = candidate_docs_full[(candidate_docs_full['query_id'] == target_query_id) & (candidate_docs_full['rank'] == 1000)]['doc_id'].values[0]
    #print(validator_document_id)
    validator_document = candidate_docs_full[candidate_docs_full['doc_id'] == validator_document_id]['doc_content'].values[0]
    #print(validator_document)
    return validator_document_id, validator_document

def create_target_document_rank(candidate_docs_full, target_query_id, validator_document_id):
    target_document_rank = candidate_docs_full[(candidate_docs_full['query_id'] == target_query_id) & (candidate_docs_full['doc_id'] == validator_document_id)]['rank'].values[0]
    #print(target_document_rank)
    return target_document_rank



def context_generation(cross_reranker_sample_1000_query, target_query_id, top_n_context):
    # Filter rows where rank is between 1 to 5 or 995 to 999
    #top_n_ranks = group[group["rank"].isin(list(range(1, 6)) + list(range(995, 1000)))]

    # Filter rows where rank is between 1 to 5
    group = cross_reranker_sample_1000_query[cross_reranker_sample_1000_query['query_id'] == target_query_id]
    top_n_ranks = group[group["rank"].isin(range(1, top_n_context + 1))]
    return top_n_ranks



def create_target_context(cross_reranker_sample_1000_query, collection, target_query_id, top_n_context):

    context_docs_df = context_generation(cross_reranker_sample_1000_query, target_query_id, top_n_context)
    context_docs_df = context_docs_df[['query_id', 'query', 'doc_id', 'rank', 'distance']]
    context_docs_full = context_docs_df.merge(collection, on="doc_id", how="inner")
    context_docs_full = context_docs_full[
        ['query_id', 'query', 'doc_id', 'doc_content', 'rank', 'distance']
    ]
    context_docs_full['doc_context'] = ' - '.join(context_docs_full['doc_content'])
    target_context = context_docs_full.loc[0, 'doc_context']

    return target_context






def generate_boosting_sentences(llm, prompt, n):

    #response = llm([HumanMessage(content=prompt)]).content.strip()
    response = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    sentences = response.split('\n')
    
    # Filter out sentences that are empty or start with clarification text
    valid_sentences = [sentence.lstrip('123456789. ').strip() for sentence in sentences if sentence.strip()]
    
    # Extract the first `n` valid sentences or add placeholders if not enough
    boosting_sentences = valid_sentences[:n]
    while len(boosting_sentences) < n:
        boosting_sentences.append("[PLACEHOLDER: Generated sentence missing]")
    
    return boosting_sentences


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


def create_llm_initial_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent):

    prompt_template = (
        "The target query is: '{query}'. "
        "The validator document is currently ranked at position 1000. Its content is: '{validator}'. "
        "The top-ranked documents are: '{context}', with each document separated by '-'. "
        "Your task is to identify {n_sent} sentences from the top-ranked documents that, when added to the validator document, "
        "will make it rank 1 instead of 1000 for the query. "
        "Ensure each sentence is relevant, persuasive, and addresses the query context."
    )
    prompt = prompt_template.format(
        query=target_query, validator=validator_document, context=target_context, n_sent=n_sent)
    boosting_sentences = generate_boosting_sentences(llm, prompt, n_sent)
    return boosting_sentences


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


def feedback_llm_without_sent_position(llm, target_query, validator_document, target_context, n_sent, already_generated_new_sentences_separated):

    # Feedback prompt template with enhanced clarity and specificity
    feedback_prompt_template = (
        "The target query is: '{query}'.\n"
        "The validator document is currently ranked at position 1000. Its content is:\n"
        "'{validator}'.\n"
        "The top-ranked documents for the query are as follows (separated by '-'): '{context}'.\n\n"
        "Previously, the following sentences were generated (separated by '-'): '{previous_sentences}', "
        "to be added to the validator document. "
        "However, these sentences failed to improve the ranking of the validator document to position 1 for the query.\n\n"
        "Your task:\n"
        "1. Analyze the target query, validator document, and the context provided.\n"
        "2. Generate {n_sent} new, unique, and highly optimized sentences.\n"
        "3. These sentences should be highly relevant to the query, persuasive, and address the context of the top-ranked documents.\n"
        "4. Ensure these sentences, when added to the validator document, "
        "significantly improve its ranking to position 1 for the target query."
    )
    
    # Format the feedback prompt with the provided parameters
    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=target_context,
        previous_sentences=already_generated_new_sentences_separated,
        n_sent=n_sent
    )
    
    # Generate improved sentences using the feedback prompt
    improved_sentences = generate_boosting_sentences(llm, feedback_prompt, n_sent)
    return improved_sentences


def llm_with_best_sent(llm, target_query, validator_document, best_sentence, context):

    # Feedback prompt template for the language model
    feedback_prompt_template = (
        "Your goal is to optimize the validator document to make it rank at position 1 for the given query. Here are the details:\n\n"
        "1. **Target Query:** '{query}'\n"
        "2. **Validator Document (Current Rank: 1000):**\n'{validator}'\n"
        "3. **Top-ranked Context (Examples of top-ranked documents, separated by '-'):**\n'{context}'\n"
        "4. **Best Sentence Identified:** '{best_sentence}'\n\n"
        "### Task Details\n"
        "Using the information above, you need to:\n"
        "1. Rephrase or revise the best sentence and seamlessly integrate it into the validator document.\n"
        "2. Ensure the revised validator document directly addresses the target query with clarity and relevance.\n"
        "3. Incorporate elements from the top-ranked context to match the language patterns, structure, and specificity of successful documents.\n"
        "4. The length of the validator document can only increase or decrease by 5%.\n"
        "5. Ensure at least 80% similarity between the revised validator document and the original.\n\n"
        "### Success Criteria\n"
        "- The revised validator document must be persuasive, informative, and highly relevant to the target query.\n"
        "- It should align with the content and intent of the top-ranked documents while maintaining originality.\n"
        "- The goal is to significantly improve the validator document's ranking to position 1 for the query."
    )
 

    feedback_prompt = feedback_prompt_template.format(
        query=target_query,
        validator=validator_document,
        context=context,
        best_sentence=best_sentence
    )

    #response = llm([HumanMessage(content=feedback_prompt)]).content.strip()
    response = llm.invoke([HumanMessage(content=feedback_prompt)]).content.strip()

    return response




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





def target_doc_content_replacement(cross_reranker_sample_1000_query, target_query_id, validator_document_id, collection, new_validator_document):
    
    #cross_reranker for that query_id only
    target_document_cross_reranker_sample_1000_query = cross_reranker_sample_1000_query[cross_reranker_sample_1000_query['query_id'] == target_query_id]
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query.merge(collection, on="doc_id", how="inner")
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query[['query_id', 'query', 'doc_id', 'doc_content', 'rank', 'distance']]

    # Replace content
    target_document_cross_reranker_sample_1000_query.loc[target_document_cross_reranker_sample_1000_query['doc_id'] == validator_document_id, 'doc_content'] = new_validator_document

    return target_document_cross_reranker_sample_1000_query 


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


def calculate_avg_top_n_perplexity_coh_cola_score(top_n_df):

    # Initialize scorers
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



def perplexity_coh_score(cross_reranker_sample_1000_query, collection, target_query_id):

    #Calculate the avg of perpelixty and coh_score
    target_document_cross_reranker_sample_1000_query = cross_reranker_sample_1000_query[cross_reranker_sample_1000_query['query_id'] == target_query_id]
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query.merge(collection, on="doc_id", how="inner")
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query[['query_id', 'query', 'doc_id', 'doc_content', 'rank', 'distance']]

    ppl_coh_target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query
    ppl_coh_target_document_cross_reranker_sample_1000_query = calculate_and_add_perplexity_coh_score(ppl_coh_target_document_cross_reranker_sample_1000_query)

    return ppl_coh_target_document_cross_reranker_sample_1000_query





def avg_top_n_perplexity_coh_cola_score(cross_reranker_sample_1000_query, collection, target_query_id, top_n_context):

    #Calculate the avg of perpelixty and coh_score of top n rank
    target_document_cross_reranker_sample_1000_query = cross_reranker_sample_1000_query[cross_reranker_sample_1000_query['query_id'] == target_query_id]
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query.merge(collection, on="doc_id", how="inner")
    target_document_cross_reranker_sample_1000_query = target_document_cross_reranker_sample_1000_query[['query_id', 'query', 'doc_id', 'doc_content', 'rank', 'distance']]

    top_n_df  = target_document_cross_reranker_sample_1000_query.head(top_n_context)
    avg_perplexity, avg_coh_score, avg_top_n_grammar_issues = calculate_avg_top_n_perplexity_coh_cola_score(top_n_df)
    return avg_perplexity, avg_coh_score, avg_top_n_grammar_issues


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


def append_to_df_dataset_per_query(target_reranked_run_df, target_query, target_document_rank, collection, sent, sent_position, new_validator_document, df_dataset_per_query):

    full_target_reranked_run_df = target_reranked_run_df
    full_target_reranked_run_df['query'] = target_query

    full_target_reranked_run_df = full_target_reranked_run_df.merge(collection, on="doc_id", how="inner")
    full_target_reranked_run_df = full_target_reranked_run_df[['query_id', 'query', 'doc_id','doc_content', 'rank','distance','new_rank', 'new_distance']]
    #full_target_reranked_run_df.columns

    selected_row =  full_target_reranked_run_df[full_target_reranked_run_df['rank'] == target_document_rank]
    selected_row = selected_row.copy()
    selected_row['new_sent'] = sent
    selected_row['new_sent_position'] = sent_position
    selected_row['new_doc_content'] = new_validator_document

    selected_row = selected_row[['query_id', 'query', 'doc_id','doc_content', 'rank','distance', 'new_sent' , 'new_sent_position' ,'new_doc_content', 'new_rank', 'new_distance']]
    #print(selected_row)

    #selected_row['doc_content'].values[0]
    #selected_row['new_doc_content'].values[0]

    df_dataset_per_query = pd.concat([df_dataset_per_query, selected_row], ignore_index=True)
    return df_dataset_per_query


def dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query, sent_position):
    # Filter rows where 'new_sent_position' is equal to 'sent_position'
    filtered_df = df_dataset_per_query[df_dataset_per_query['new_sent_position'] == sent_position]
    
    # Check if any row in the filtered dataframe has 'new_rank' equal to 1
    return (filtered_df['new_rank'] <= 10).any()


def dataset_per_query_has_rank_below_n_without_sent_position(df_dataset_per_query):

    filtered_df = df_dataset_per_query
    # Check if any row in the filtered dataframe has 'new_rank' equal to 1
    return (filtered_df['new_rank'] <= 10).any()



def feedback_generated_sentences_per_query_rank_below_10_separated_with_sent_position(df_dataset_per_query, sent_position):
    # Filter rows where 'new_sent_position' is equal to 'sent_position'
    filtered_df = df_dataset_per_query[
        (df_dataset_per_query['new_sent_position'] == sent_position) &
        (df_dataset_per_query['new_rank'] <= 10)  # Include only rows with 'rank' <= 5
    ]
    
    # Join all 'new_sent' values in the filtered dataframe separated by '-'
    return '-'.join(filtered_df['new_sent'].astype(str))



def feedback_generated_sentences_per_query_rank_below_5_separated_without_sent_position(df_dataset_per_query): #No Duplicate
    # Filter rows where new_rank <= 5
    filtered_df = df_dataset_per_query[df_dataset_per_query['new_rank'] <= 5]
    
    # Remove duplicates in the 'new_sent' column
    unique_new_sent = filtered_df['new_sent'].drop_duplicates()
    
    # Join the unique 'new_sent' values with '-'
    return '-'.join(unique_new_sent.astype(str))


def create_per_query_dataset(df_dataset_per_query, target_query_id, target_query, validator_document_id, validator_document, target_document_rank, model, boosting_sentences, cross_reranker_sample_1000_query, collection, sent_position):

    # Create dataset_per_query (different position, repalce content, rerank and append to the per query dataset)

    for sent in boosting_sentences: #different new sent
        
        new_validator_document = sent_position_function(sent, sent_position, validator_document) # new content is created with placing sent in sent_position
        target_document_cross_reranker_sample_1000_query = target_doc_content_replacement(cross_reranker_sample_1000_query, target_query_id, validator_document_id, collection, new_validator_document) #replace content
        
        target_reranked_run_df, new_calculated_rank = rerank_modified_document(model, target_document_cross_reranker_sample_1000_query, target_query_id, target_query, validator_document_id) #rerank with modified
        df_dataset_per_query = append_to_df_dataset_per_query(target_reranked_run_df, target_query, target_document_rank, collection, sent, sent_position, new_validator_document, df_dataset_per_query) #append to the dataset for this query
        
    return df_dataset_per_query


def create_best_sent_dataset(df_dataset_per_query, target_query_id, target_query, validator_document_id, target_document_rank, model, cross_reranker_sample_1000_query, collection, best_new_sent, rephrased_validator_document):

   
    target_document_cross_reranker_sample_1000_query = target_doc_content_replacement(cross_reranker_sample_1000_query, target_query_id, validator_document_id, collection, rephrased_validator_document) #replace content
    
    target_reranked_run_df, new_calculated_rank = rerank_modified_document(model, target_document_cross_reranker_sample_1000_query, target_query_id, target_query, validator_document_id) #rerank with modified
    df_dataset_per_query = append_to_df_dataset_per_query(target_reranked_run_df, target_query, target_document_rank, collection, best_new_sent, "rephrased", rephrased_validator_document, df_dataset_per_query) #append to the dataset for this query
    
    return df_dataset_per_query






def re_org_df_dataset_per_query_with_score(df_dataset_per_query_with_score_new, avg_perplexity, avg_coh_score, avg_top_n_grammar_issues):
    df_dataset_per_query_with_score_new['top_n_avg_perplexity_score'] = avg_perplexity
    df_dataset_per_query_with_score_new['top_n_avg_coh_score'] = avg_coh_score
    df_dataset_per_query_with_score_new['top_n_avg_grammar_issues'] = avg_top_n_grammar_issues
    df_dataset_per_query_with_score_new = calculate_and_add_perplexity_coh_gramm_cola_score_modified_document(df_dataset_per_query_with_score_new)
    df_dataset_per_query_with_score_new = df_dataset_per_query_with_score_new[['query_id', 'query', 'doc_id','doc_content', 'top_n_avg_perplexity_score', 'top_n_avg_coh_score', 'top_n_avg_grammar_issues', 'rank','distance', 'new_sent' , 'new_sent_position' ,'new_doc_content','new_perplexity_score', 'new_coh_score', 'new_grammar_issues', 'new_cola_score', 'new_rank', 'new_distance']]
    return df_dataset_per_query_with_score_new




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





def get_best_new_sent(df):

    return df.loc[df['new_rank'].idxmin(), 'new_sent']



