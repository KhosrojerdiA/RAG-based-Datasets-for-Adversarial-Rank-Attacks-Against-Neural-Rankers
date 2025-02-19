import sys
import os
import time

os.system("nvidia-smi")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

project_path = '/mnt/data/khosro/Amin/code'
sys.path.append(project_path)

# Set a seed for reproducibility
#torch.manual_seed(3708) 
from Amin.archived.utils_local import *

from Amin.archived.utils_local import RankDataset
from Amin.archived.utils_local import GPT2PPLScorer
from Amin.archived.utils_local import DocumentEvaluator
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
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document
from langchain.schema import HumanMessage


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ___________________________________________________________________________________________________________________________________________

data_folder = "/mnt/data/khosro/Amin/data"
checkpoint = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

#model_path = "/mnt/data//akhosrojerdi/Amin/trained_model"
#data_path = f'{model_path}/train_dataset.pt'

model_name = checkpoint  
model = CrossEncoder(checkpoint, max_length=512, device=device)
#os.environ["OPENAI_API_KEY"] = "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"
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0.6, max_tokens = 30)
#llm = ChatOpenAI(model_name="gpt-4o", temperature=0.6)

# Hugging Face TGI endpoint URL
TGI_URL = "https://ai.ls3.rnet.torontomu.ca/llm/"
# Use the updated HuggingFaceEndpoint class
llm = HuggingFaceEndpoint(
    endpoint_url=TGI_URL,
    top_k=5,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.6,
    repetition_penalty=1.03,
)


top_n_context = 5
n_sent = 5
llm_state = "without sent_position" #["with sent_position", "without sent_position"]
max_feedback_iteration = 5

# ___________________________________________________________________________________________________________________________________________

# Define input and output file paths
#input_file = "/mnt/data/khosro/Amin/data/msmarco-passage.dev.small.1k-query.bm25-default.top1k.features.json_rerank.trec"
#output_file = "/mnt/data/khosro/Amin/data/msmarco_passage_rerank.csv"
#msmarco_passage_rerank = pd.read_csv(input_file, delim_whitespace=True, header=None, names=["query_id", "q0", "doc_id", "rank", "score", "ranker"])
#msmarco_passage_rerank = msmarco_passage_rerank[["query_id", "doc_id", "score", "rank"]]
#msmarco_passage_rerank.to_csv(output_file, index=False)
#collection = pd.read_csv('/mnt/data/khosro/Amin/data/collection.tsv', sep='\t', header=None, names=['doc_id', 'doc_content'])
#queries = pd.read_csv('/mnt/data/khosro/Amin/data/queries.dev.small.tsv', sep='\t', header=None, names=['query_id', 'query'])
#merged_df = msmarco_passage_rerank.merge(queries, on="query_id", how="left")
#merged_df = merged_df.merge(collection, on="doc_id", how="left")
#candidate_docs_full = merged_df[['query_id', 'query', 'doc_id', 'doc_content', 'score', 'rank']]
#candidate_docs_full.to_csv('/mnt/data/khosro/Amin/output/candidate_docs_full.csv')
#missing_rows = candidate_docs_full[(candidate_docs_full["query"].isna()) | (candidate_docs_full["query"] == " ") | 
#                  (candidate_docs_full["doc_content"].isna()) | (candidate_docs_full["doc_content"] == " ")]

# ___________________________________________________________________________________________________________________________________________

#candidate_docs_full = pd.read_csv('/mnt/data/khosro/Amin/output/candidate_docs_full.csv')
#candidate_docs_full = candidate_docs_full.drop(columns=["Unnamed: 0"], errors="ignore")
#candidate_docs_full.columns
# 'query_id', 'query', 'doc_id', 'doc_content', 'score', 'rank'
#candidate_docs_full

# ___________________________________________________________________________________________________________________________________________

chunk_1 = pd.read_csv('/mnt/data/khosro/Amin/data/query_documents_chunk_1.tsv', sep='\t')
candidate_docs = chunk_1

candidate_docs = (candidate_docs.rename(columns={'qid': 'query_id', 'pid': 'doc_id', 'document': 'doc_content', 'new_rank': 'rank'})
    [['query_id', 'query', 'doc_id', 'doc_content', 'score', 'rank']])

# ___________________________________________________________________________________________________________________________________________


#target_query_id = 16860 -> all will be found (rank 1)

#target_query_id = 1048876


target_query_id = 2

target_doc_rank = 1000

#target_query_id = 1048995

df_dataset_final = pd.DataFrame()

# ___________________________________________________________________________________________________________________________________________

####### Start Target Query Loop

count_query = 0
start_time = time.time()

candidate_docs_full_query = candidate_docs[candidate_docs['query_id'] == target_query_id]


df_dataset_per_query = pd.DataFrame()
df_dataset_per_query_with_feedback = pd.DataFrame()
count_query += 1
print(f"___________________________________________________________")
print(f"Count Query ID -> {count_query}")

# ___________________________________________________________________________________________________________________________________________

# Target Query (query content)
target_query = create_target_query(candidate_docs_full_query)

# ___________________________________________________________________________________________________________________________________________

# Validator Document
validator_document_id, validator_document = create_validator_document_info(candidate_docs_full_query, target_doc_rank)

# ___________________________________________________________________________________________________________________________________________

print(f"___________________________________________________________")
print(f"Starting the Process for Query ID -> {target_query_id}")
print(f"___________________________________________________________")
print(f"Starting the Process for Document ID -> {validator_document_id}")
print(f"___________________________________________________________")

# ___________________________________________________________________________________________________________________________________________

# Sent Position List for Validator Document (Number of sentences)
select_sent_postion = generate_sent_position_list(validator_document)

# ___________________________________________________________________________________________________________________________________________

# Target Rank
target_document_rank = create_target_document_rank(candidate_docs_full_query, validator_document_id)

#___________________________________________________________________________________________________________________________________________

# Target Context (top 5 documents in crossencoder rank)

target_context = create_target_context(candidate_docs_full_query, top_n_context)
candidate_docs_full_query['doc_context'] = target_context
candidate_docs_full_query = candidate_docs_full_query[['query_id', 'query', 'doc_id', 'doc_content', 'doc_context', 'score', 'rank']]
# 'query_id', 'query', 'doc_id', 'doc_content', 'doc_context', 'score','rank'
     
#___________________________________________________________________________________________________________________________________________

# Avg top n Perplexity and coh_score

#avg_perplexity, avg_coh_score, avg_top_n_grammar_issues = avg_top_n_perplexity_coh_cola_score(candidate_docs_full_query, top_n_context)

#___________________________________________________________________________________________________________________________________________

# LLM - Create Initial Response with and without sent_position

count_boosting_sentences = 0


boosting_sentences = create_llm_initial_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent)
count_boosting_sentences = count_boosting_sentences + len(boosting_sentences)
print(f"___________________________________________________________")
print(f"________________________Initial LLM___________________________________")
print(f"{count_boosting_sentences} sentences has been generated!")
print(f"___________________________________________________________")
print(f"{count_boosting_sentences} Rerank!")
for sent_position in select_sent_postion:
    candidate_docs_full_query_loop = candidate_docs_full_query    
    df_dataset_per_query = create_per_query_dataset(df_dataset_per_query, validator_document_id, validator_document, target_document_rank, 
                                                    model, boosting_sentences, candidate_docs_full_query_loop, sent_position, target_context)


df_dataset_per_query.to_csv('/mnt/data/khosro/Amin/output/phase_1_initial_LLM.csv', index=False)

#___________________________________________________________________________________________________________________________________________

# Check if we do not have rank 1 in per query dataset -> second LLM Agent

df_dataset_per_query_with_feedback = df_dataset_per_query
#'query_id', 'query', 'doc_id','doc_content', 'score','rank', 'doc_context', 'new_sent' , 'new_sent_position' ,'new_doc_content', 'new_score', 'new_rank'

feedback_counter = 0

for sent_position in select_sent_postion:
    feedback_counter = 0
    while not dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query_with_feedback, sent_position): #does not have rank 1
        feedback_counter += 1
        if feedback_counter > max_feedback_iteration:
            print(f"___________________________________________________________")
            print(f"Maximum retries reached for sent_position -> {sent_position}")
            print(f"Breaking out of the loop. Could not find rank below 10 for -> {sent_position}")
            print(f"___________________________________________________________")
            break  # Exit the loop after maximum retries

        print(f"___________________________________________________________")
        print(f"Does not have Rank 1 for sent_position -> {sent_position}")
        print(f"___________________________________________________________")
        print(f"Attempt {feedback_counter} for -> {sent_position}")
        print(f"___________________________________________________________")
        print(f"Generating New Sentences")
        #get new_sentence for this sent_position seperated by - 
        already_generated_new_sentences_separated = feedback_generated_sentences_per_query_rank_below_10_separated_with_sent_position(df_dataset_per_query_with_feedback, 
                                                                                                                                      sent_position, 100) 
        #print(already_generated_new_sentences_separated)
        improved_sentences = feedback_llm_without_sent_position(llm, target_query, validator_document, target_context, n_sent, already_generated_new_sentences_separated)
        #print(improved_sentences)
        print(f"___________________________________________________________")
        print(f"{len(improved_sentences)} sentences has been generated!")
        print(f"___________________________________________________________")
        print(f"Rerank New Documents")
        candidate_docs_full_query_loop = candidate_docs_full_query  
        df_dataset_per_query_with_feedback = create_per_query_dataset(df_dataset_per_query_with_feedback, validator_document_id, validator_document, target_document_rank, 
                                                                      model, improved_sentences, candidate_docs_full_query_loop, sent_position, target_context)
        print(f"___________________________________________________________")
        print(f"Remove high rank Documents")
        df_dataset_per_query_with_feedback = remove_highest_new_rank_rows(df_dataset_per_query_with_feedback, sent_position, n_sent)
        df_dataset_per_query_with_feedback = df_dataset_per_query_with_feedback.reset_index(drop=True)

    if dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query_with_feedback, sent_position): #does have rank 1
        print(f"___________________________________________________________")
        print(f"Rank below 10 Found for sent_position -> {sent_position}")



best_new_sent = get_best_new_sent(df_dataset_per_query_with_feedback)
rephrased_validator_document = llm_with_best_sent(llm, target_query, validator_document, best_new_sent)
candidate_docs_full_query_loop = candidate_docs_full_query 
df_dataset_per_query_with_feedback = create_best_sent_dataset(df_dataset_per_query_with_feedback, validator_document_id, 
                                                              rephrased_validator_document, target_document_rank, 
                                                              model, candidate_docs_full_query_loop, "rephrased", target_context)


#df_dataset_per_query_with_feedback.to_csv('/mnt/data/khosro/Amin/output/phase_2.csv', index=False)

#___________________________________________________________________________________________________________________________________________


df_dataset_per_query_with_feedback_with_score = df_dataset_per_query_with_feedback.copy()
#df_dataset_per_query_with_feedback_with_score = re_org_df_dataset_per_query_with_score(df_dataset_per_query_with_feedback_with_score, avg_perplexity, avg_coh_score, avg_top_n_grammar_issues)

# ___________________________________________________________________________________________________________________________________________

df_dataset_final = pd.concat([df_dataset_final, df_dataset_per_query_with_feedback_with_score], ignore_index=True)
    
###### End of Target Query Loop

# ___________________________________________________________________________________________________________________________________________

end_time = time.time()
duration = end_time - start_time

print(duration) 

df_dataset_final.to_csv('/mnt/data/khosro/Amin/output/phase_final.csv', index=False)

# ___________________________________________________________________________________________________________________________________________


