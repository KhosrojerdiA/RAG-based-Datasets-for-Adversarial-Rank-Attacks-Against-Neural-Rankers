import sys
import os
import time

os.system("nvidia-smi")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

project_path = '/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval'
sys.path.append(project_path)

# Set a seed for reproducibility
#torch.manual_seed(3708) 
from utils import *
from utils.llm_agent import *

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
from langchain_huggingface import HuggingFaceEndpoint

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chunk_number = 2


num_max_token = 30

# ___________________________________________________________________________________________________________________________________________

checkpoint = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

#model_path = "/mnt/data//akhosrojerdi/Amin/trained_model"
#data_path = f'{model_path}/train_dataset.pt'

model_name = checkpoint  
model = CrossEncoder(checkpoint, max_length=512, device=device)

llm_initial = ChatOpenAI(model_name="unsloth/Qwen2.5-72B-Instruct-bnb-4bit", base_url = "https://ai.ls3.rnet.torontomu.ca/llm2/v1/")
llm_feedback = ChatOpenAI(model_name="unsloth/Qwen2.5-72B-Instruct-bnb-4bit", base_url = "https://ai.ls3.rnet.torontomu.ca/llm2/v1/",temperature=0.6)

#llm_initial = ChatOpenAI(model_name="o1-mini")
#llm_feedback = ChatOpenAI(model_name="gpt-4o", temperature=0.6)

top_n_context = 5
n_sent = 5
max_feedback_iteration = 5


# ___________________________________________________________________________________________________________________________________________

chunk = pd.read_csv(f'/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/data/IDEM_query_documents_chunk_{chunk_number}.tsv', sep='\t')
#598 queries, for each query, 20 documents (10 context and 10 target documents)

candidate_docs = chunk
candidate_docs = (candidate_docs.rename(columns={'qid': 'query_id', 'pid': 'doc_id', 'document': 'doc_content', 'new_rank': 'rank'})
    [['query_id', 'query', 'doc_id', 'doc_content', 'score', 'rank']])

# ___________________________________________________________________________________________________________________________________________

target_query_id_list = candidate_docs['query_id'].unique().tolist()


target_doc_rank_list = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009] #[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]

# ___________________________________________________________________________________________________________________________________________

count_doc = 0
count_query = 0
start_time = time.time()
df_dataset_final = pd.DataFrame()

################ Start Target Query Loop

for target_query_id in target_query_id_list:

    df_dataset_per_query = pd.DataFrame()
    df_dataset_per_query_with_feedback = pd.DataFrame()
    count_doc += 1
    print("###########################################################################")
    print(f"Count Doc -> {count_doc}")
    print("###########################################################################")

    for target_doc_rank in target_doc_rank_list:

        candidate_docs_full_query = candidate_docs[candidate_docs['query_id'] == target_query_id]
        df_dataset_per_query = pd.DataFrame()
        df_dataset_per_query_with_feedback = pd.DataFrame()
        count_query += 1
        print(f"___________________________________________________________")
        print(f"Count Query -> {count_query}")
        print(f"___________________________________________________________")

        # ___________________________________________________________________________________________________________________________________________

        # Target Query (query content)
        target_query = create_target_query(candidate_docs_full_query)

        # ___________________________________________________________________________________________________________________________________________

        # Validator Document
        validator_document_id, validator_document = create_validator_document_info(candidate_docs_full_query, target_doc_rank)

        # ___________________________________________________________________________________________________________________________________________

        #print(f"___________________________________________________________")
        #print(f"Starting the Process for Query ID -> {target_query_id}")
        #print(f"___________________________________________________________")
        #print(f"Starting the Process for Document ID -> {validator_document_id}")
        #print(f"___________________________________________________________")

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


        boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B = create_initial_llm_response_without_sent_position(llm_initial, target_query, validator_document, 
                                                                                                                        target_context, n_sent, num_max_token)
        count_boosting_sentences = count_boosting_sentences + len(boosting_sentences)
        print(f"___________________________________________________________")
        print(f"________________________Initial LLM___________________________________")
        print(f"{count_boosting_sentences} sentences has been generated!")
        print(f"___________________________________________________________")
        print(f"{count_boosting_sentences} Rerank!")

        if count_boosting_sentences == 0:
              boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B = create_initial_llm_response_without_sent_position(llm_initial, target_query, validator_document, 
                                                                                                                        target_context, n_sent, num_max_token)
              print(f"{count_boosting_sentences} sentences has been generated!")

        for sent_position in select_sent_postion:
            candidate_docs_full_query_loop = candidate_docs_full_query    
            df_dataset_per_query = create_per_query_dataset(df_dataset_per_query, validator_document_id, validator_document, target_document_rank, 
                                                            model, boosting_sentences, candidate_docs_full_query_loop, sent_position, target_context)


        #df_dataset_per_query.to_csv('/mnt/data/khosro/Amin/idem 4 chunk results/phase_1_chunk_number_.csv', index=False)

        #___________________________________________________________________________________________________________________________________________

        # Check if we do not have rank 1 in per query dataset -> second LLM Agent

        df_dataset_per_query_with_feedback = df_dataset_per_query
        #'query_id', 'query', 'doc_id','doc_content', 'score','rank', 'doc_context', 'new_sent' , 'new_sent_position' ,'new_doc_content', 'new_score', 'new_rank'

        feedback_counter = 0

        for sent_position in select_sent_postion:
            feedback_counter = 0
            if not  dataset_per_query_has_rank_below_n(df_dataset_per_query_with_feedback): # does not have rank 1 for this query
                    while not dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query_with_feedback, sent_position): #does not have rank 1 for this sent_position for this query
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
                                #print(f"___________________________________________________________")
                                #print(f"Generating New Sentences")
                                #get new_sentence for this sent_position seperated by - 
                                already_generated_new_sentences_separated = feedback_generated_sentences_per_query_rank_below_10_separated_with_sent_position(df_dataset_per_query_with_feedback, 
                                                                                                                                                            sent_position, 100) 
                                #print(already_generated_new_sentences_separated)
                                improved_sentences = feedback_llm_without_sent_position(llm_feedback, 
                                                            target_query, 
                                                            validator_document, 
                                                            target_context, 
                                                            n_sent, 
                                                            already_generated_new_sentences_separated, 
                                                            key_phrases_buffer_A, 
                                                            key_phrases_buffer_B,
                                                            num_max_token)
                                #print(improved_sentences)
                                #print(f"___________________________________________________________")
                                #print(f"sentences has been generated!")
                                #print(f"___________________________________________________________")
                                #print(f"Rerank New Documents")
                                candidate_docs_full_query_loop = candidate_docs_full_query  
                                df_dataset_per_query_with_feedback = create_per_query_dataset(df_dataset_per_query_with_feedback, validator_document_id, validator_document, target_document_rank, 
                                                                                            model, improved_sentences, candidate_docs_full_query_loop, sent_position, target_context)
                                #print(f"___________________________________________________________")
                                #print(f"Remove high rank Documents")
                                df_dataset_per_query_with_feedback = remove_highest_new_rank_rows(df_dataset_per_query_with_feedback, sent_position, n_sent)
                                df_dataset_per_query_with_feedback = df_dataset_per_query_with_feedback.reset_index(drop=True)

                    if dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query_with_feedback, sent_position): #does have rank 1
                        print(f"___________________________________________________________")
                        print(f"Rank below 10 Found for sent_position -> {sent_position}")




        best_new_sent = get_best_new_sent(df_dataset_per_query_with_feedback)
        rephrased_validator_document = llm_with_best_sent(llm_feedback, target_query, validator_document, best_new_sent, num_max_token)
        candidate_docs_full_query_loop = candidate_docs_full_query 
        df_dataset_per_query_with_feedback = create_best_sent_dataset(df_dataset_per_query_with_feedback, validator_document_id, 
                                                                    rephrased_validator_document, target_document_rank, 
                                                                    model, candidate_docs_full_query_loop, best_new_sent, target_context)


        #df_dataset_per_query_with_feedback.to_csv('/mnt/data/khosro/Amin/idem 4 chunk results/phase_2_chunk_number_.csv', index=False)

        #___________________________________________________________________________________________________________________________________________


        df_dataset_per_query_with_feedback_with_score = df_dataset_per_query_with_feedback.copy()
        #df_dataset_per_query_with_feedback_with_score = query_top_3(df_dataset_per_query_with_feedback_with_score)

        #df_dataset_per_query_with_feedback_with_score = re_org_df_dataset_per_query_with_score(df_dataset_per_query_with_feedback_with_score, avg_perplexity, avg_coh_score, avg_top_n_grammar_issues)

        # ___________________________________________________________________________________________________________________________________________

        df_dataset_final = pd.concat([df_dataset_final, df_dataset_per_query_with_feedback_with_score], ignore_index=True)
        df_dataset_final = add_best_query_doc_columns(df_dataset_final)
        df_dataset_final.to_csv(f'/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/output/phase_3_chunk_{chunk_number}.csv', index=False)
        print("###########################################################################")

################ End of Target Query Loop

# ___________________________________________________________________________________________________________________________________________

df_dataset_final.to_csv(f'/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/output/phase_4_chunk_{chunk_number}.csv', index=False)

end_time = time.time()
duration = end_time - start_time

print(duration) 

# ___________________________________________________________________________________________________________________________________________