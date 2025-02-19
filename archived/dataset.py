import sys
import os

os.system("nvidia-smi")
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

project_path = '/mnt/data/khosro/Amin/code'
sys.path.append(project_path)

# Set a seed for reproducibility
#torch.manual_seed(3708) 
from utils import *
from utils import RankDataset
from utils import GPT2PPLScorer
from utils import DocumentEvaluator
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
from langchain.schema import Document
from langchain.schema import HumanMessage
import time

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ___________________________________________________________________________________________________________________________________________

data_folder = "/mnt/data/khosro/Amin/data"
checkpoint = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
#model_path = "/mnt/data//akhosrojerdi/Amin/trained_model"
#data_path = f'{model_path}/train_dataset.pt'

model_name = checkpoint  
model = CrossEncoder(model_name, max_length = 512)
os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6)
#llm = ChatOpenAI(model_name="gpt-4o", temperature=0.6)

top_n_context = 5
n_sent = 5
llm_state = "without sent_position" #["with sent_position", "without sent_position"]
max_feedback_iteration = 5

# ___________________________________________________________________________________________________________________________________________

#Ranker
cross_reranker_sample_1000_query = pd.read_excel('/mnt/data/khosro/Amin/output/index_rank_1000_sample_query.xlsx')

#Documents
collection = pd.read_csv('/mnt/data/khosro/Amin/data/collection.tsv', sep='\t', header=None, names=['doc_id', 'doc_content'])

# Candidate doc full
candidate_docs_full = create_candidate_docs_full(cross_reranker_sample_1000_query, collection)
# ___________________________________________________________________________________________________________________________________________


#target_query_id = 16860 -> all will be found (rank 1)

#target_query_id = 1048876


target_query_id = 262232

#target_query_id = 1048995

df_dataset_final = pd.DataFrame()
# ___________________________________________________________________________________________________________________________________________


# Extract the first 5 unique query IDs and store them in a list
unique_100_query_ids_list = candidate_docs_full['query_id'].drop_duplicates().head(3).tolist()

# ___________________________________________________________________________________________________________________________________________

#Start Target Query Loop
count_query = 0
start_time = time.time()

#for target_query in unique_100_query_ids_list:

#    target_query_id = target_query #from here

df_dataset_per_query = pd.DataFrame()
df_dataset_per_query_with_feedback = pd.DataFrame()
count_query += 1
print(f"___________________________________________________________")
print(f"Count Query ID -> {count_query}")

# ___________________________________________________________________________________________________________________________________________

# Target Query
target_query = create_target_query(candidate_docs_full, target_query_id)

# ___________________________________________________________________________________________________________________________________________

# Validator Document
validator_document_id, validator_document = create_validator_document_info(candidate_docs_full, target_query_id)

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
target_document_rank = create_target_document_rank(candidate_docs_full, target_query_id, validator_document_id)

#___________________________________________________________________________________________________________________________________________

# Target Context (top 5 documents in crossencoder rank)

target_context = create_target_context(cross_reranker_sample_1000_query, collection, target_query_id, top_n_context)

#___________________________________________________________________________________________________________________________________________

# Avg top n Perplexity and coh_score

#ppl_coh_gramm_cola_target_document_cross_reranker_sample_1000_query = perplexity_coh_score(cross_reranker_sample_1000_query, collection, target_query_id)
avg_perplexity, avg_coh_score, avg_top_n_grammar_issues = avg_top_n_perplexity_coh_cola_score(cross_reranker_sample_1000_query, collection, target_query_id, top_n_context)

#___________________________________________________________________________________________________________________________________________

# LLM - Create Initial Response with and without sent_position

count_boosting_sentences = 0
target_query_cross_reranker = cross_reranker_sample_1000_query[cross_reranker_sample_1000_query['query_id'] == target_query_id] #1000 docs for a query

if llm_state == "without sent_position":
    boosting_sentences = create_llm_initial_response_without_sent_position(llm, target_query, validator_document, target_context, n_sent)
    count_boosting_sentences = count_boosting_sentences + len(boosting_sentences)
    print(f"___________________________________________________________")
    print(f"________________________Initial LLM___________________________________")
    print(f"{count_boosting_sentences} sentences has been generated!")
    print(f"___________________________________________________________")
    print(f"{count_boosting_sentences} Rerank!")
    for sent_position in select_sent_postion:    
        df_dataset_per_query = create_per_query_dataset(df_dataset_per_query, target_query_id, target_query, validator_document_id, validator_document, target_document_rank, model, boosting_sentences, target_query_cross_reranker, collection, sent_position)


#df_dataset_per_query
#df_dataset_per_query.to_csv('/mnt/data/khosro/Amin/output/dataset_v3.csv', index=False)
#___________________________________________________________________________________________________________________________________________

# Check if we do not have rank 1 in per query dataset -> second LLM Agent

df_dataset_per_query_with_feedback = df_dataset_per_query
feedback_counter = 0


if llm_state == "without sent_position":
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
            already_generated_new_sentences_separated = feedback_generated_sentences_per_query_rank_below_10_separated_with_sent_position(df_dataset_per_query_with_feedback, sent_position) #get new_sentence for this sent_position seperated by - 
            improved_sentences = feedback_llm_without_sent_position(llm, target_query, validator_document, target_context, n_sent, already_generated_new_sentences_separated)
            print(f"___________________________________________________________")
            print(f"{len(improved_sentences)} sentences has been generated!")
            print(f"___________________________________________________________")
            print(f"Rerank New Documents")
            df_dataset_per_query_with_feedback = create_per_query_dataset(df_dataset_per_query_with_feedback, target_query_id, target_query, validator_document_id, validator_document, target_document_rank, model, improved_sentences, cross_reranker_sample_1000_query, collection, sent_position)
            print(f"___________________________________________________________")
            print(f"Remove high rank Documents")
            df_dataset_per_query_with_feedback = remove_highest_new_rank_rows(df_dataset_per_query_with_feedback, sent_position, n_sent)
            df_dataset_per_query_with_feedback = df_dataset_per_query_with_feedback.reset_index(drop=True)

        if dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query_with_feedback, sent_position): #does have rank 1
            print(f"___________________________________________________________")
            print(f"Rank below 10 Found for sent_position -> {sent_position}")



best_new_sent = get_best_new_sent(df_dataset_per_query_with_feedback)
rephrased_validator_document = llm_with_best_sent(llm, target_query, validator_document, best_new_sent, target_context)
df_dataset_per_query_with_feedback = create_best_sent_dataset(df_dataset_per_query_with_feedback, target_query_id, target_query, validator_document_id, target_document_rank, model, cross_reranker_sample_1000_query, collection, best_new_sent, rephrased_validator_document)
#df_dataset_per_query_with_feedback.to_csv('/mnt/data/khosro/Amin/output/dataset_v1.csv', index=False)

#get the best sentence (does not matter where sent_position is), ask LLM to generate 

#df_dataset_per_query
#df_dataset_per_query_with_feedback.columns
# ___________________________________________________________________________________________________________________________________________

df_dataset_per_query_with_feedback_with_score= df_dataset_per_query_with_feedback.copy()
df_dataset_per_query_with_feedback_with_score = re_org_df_dataset_per_query_with_score(df_dataset_per_query_with_feedback_with_score, avg_perplexity, avg_coh_score, avg_top_n_grammar_issues)
#df_dataset_per_query_with_feedback_with_score.columns
#df_dataset_per_query_with_feedback_with_score
#df_dataset_per_query_with_feedback_with_score.to_csv('/mnt/data/khosro/Amin/output/dataset_v1.csv', index=False)

# ___________________________________________________________________________________________________________________________________________

df_dataset_final = pd.concat([df_dataset_final, df_dataset_per_query_with_feedback_with_score], ignore_index=True)
    

#End of Target Query Loop

# ___________________________________________________________________________________________________________________________________________

end_time = time.time()
duration = end_time - start_time
duration #71644.20554327965

df_dataset_final.to_csv('/mnt/data/khosro/Amin/output/dataset_v2.csv', index=False)

#Add a column and per query, capture the best rank and put "best rank for the query"
#Add a column per query and sent_position, capture the best rank of the sent_position and put "best rank for the query and sent_position"
# ___________________________________________________________________________________________________________________________________________

#11:43
##11:47 for one sentence (4 min)




#change rerank_modified_document

#change create_per_query_dataset so 
#it will give distance
#replace new doc content and distance
#get new rank by sort by distance