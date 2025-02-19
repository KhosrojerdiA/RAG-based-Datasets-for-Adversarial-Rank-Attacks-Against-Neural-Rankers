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
model_name = checkpoint  
model = CrossEncoder(checkpoint, max_length=512, device=device)

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.6)
#llm = ChatOpenAI(model_name="gpt-4o", temperature=0.6)

# ___________________________________________________________________________________________________________________________________________

#Ranker
cross_reranker_sample_1000_query = pd.read_excel('/mnt/data/khosro/Amin/output/index_rank_1000_sample_query.xlsx')
#Index(['query_id', 'query', 'doc_id', 'rank', 'distance'], dtype='object')
NEW_cross_reranker_sample_1000_query = cross_reranker_sample_1000_query
NEW_cross_reranker_sample_1000_query = NEW_cross_reranker_sample_1000_query.drop(columns=['rank', 'distance'])

#Documents
collection = pd.read_csv('/mnt/data/khosro/Amin/data/collection.tsv', sep='\t', header=None, names=['doc_id', 'doc_content'])

# Candidate doc full
candidate_docs_full = create_candidate_docs_full_for_cross_encoder(NEW_cross_reranker_sample_1000_query, collection)
#Index(['query_id', 'query', 'doc_id', 'doc_content'], dtype='object')

# ___________________________________________________________________________________________________________________________________________



ranked_candidate_docs_full = cross_encoder_rank_documents(candidate_docs_full, model)
ranked_candidate_docs_full.to_csv('/mnt/data/khosro/Amin/output/ranked_candidate_docs_full.csv', index=False)