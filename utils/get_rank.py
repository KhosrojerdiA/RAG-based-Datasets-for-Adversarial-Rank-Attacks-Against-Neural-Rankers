import sys
import os
import time

os.system("nvidia-smi")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

project_path = '/mnt/data/khosro/Amin/code'
sys.path.append(project_path)

# Set a seed for reproducibility
#torch.manual_seed(3708) 
from utils import *
from llm_agent import *

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

chunk_number = 3

candidate_docs_full = pd.read_csv('/mnt/data/khosro/Amin/output/candidate_docs_full.csv')
candidate_docs_full = candidate_docs_full.drop(columns=["Unnamed: 0"], errors="ignore")
my_data = pd.read_csv(f'/mnt/data/khosro/Amin/get_rank_data/chunk_{chunk_number}_FINAL.csv')
new_ranking_df = new_rank(my_data, candidate_docs_full)
my_data['actual_new_rank'] = new_ranking_df['actual_rank']
my_data.to_csv(f'/mnt/data/khosro/Amin/get_rank_data/ranked_chunk_{chunk_number}.csv')

# ___________________________________________________________________________________________________________________________________________



candidate_docs_full = pd.read_csv('/mnt/data/khosro/Amin/output/candidate_docs_full.csv')
candidate_docs_full = candidate_docs_full.drop(columns=["Unnamed: 0"], errors="ignore")
my_data = pd.read_csv(f'/mnt/data/khosro/Amin/get_rank_data/FULL_v2.csv')
old_ranking_df = create_old_rank(my_data, candidate_docs_full)
my_data['actual_old_rank'] = old_ranking_df['old_rank']
my_data.to_csv(f'/mnt/data/khosro/Amin/get_rank_data/FULL_v3.csv')

# ___________________________________________________________________________________________________________________________________________

