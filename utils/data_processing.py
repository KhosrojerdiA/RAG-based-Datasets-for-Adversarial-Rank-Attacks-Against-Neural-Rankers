import sys
import os
import time

os.system("nvidia-smi")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

project_path = '/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval'
sys.path.append(project_path)

# Set a seed for reproducibility
#torch.manual_seed(3708) 
from utils.utils import *
from utils.llm_agent import *

from utils.utils  import RankDataset
from utils.utils  import GPT2PPLScorer
from utils.utils  import DocumentEvaluator
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
from unidecode import unidecode

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#_______________________________________________________________________________________________________________________

chunk_number = 1

chunk = pd.read_csv(f'/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/data/IDEM_query_documents_chunk_{chunk_number}.tsv', sep='\t')
#598 queries, for each query, 20 documents (10 context and 10 target documents)

candidate_docs = chunk
candidate_docs = (candidate_docs.rename(columns={'qid': 'query_id', 'pid': 'doc_id', 'document': 'doc_content', 'new_rank': 'rank'})
    [['query_id', 'query', 'doc_id', 'doc_content', 'score', 'rank']])



# Function to check for non-ASCII characters
def has_non_ascii(s):
    return any(ord(c) >= 128 for c in s) if isinstance(s, str) else False

# Function to check for quotes
def has_quotes(s):
    return "'" in s or '"' in s if isinstance(s, str) else False

# Apply both conditions to filter rows
filtered_rows = candidate_docs[
    candidate_docs[['query', 'doc_content']].applymap(has_non_ascii).any(axis=1) |
    candidate_docs[['query', 'doc_content']].applymap(has_quotes).any(axis=1)
]
#non_ascii_rows.to_csv(f'/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/data/non_ascii.csv')
len(filtered_rows)
len(candidate_docs)
#_______________________________________________________________________________________________________________________

# Convert to ASCII
ascii_candidate_docs_full = candidate_docs
def convert_to_ascii(s):
    return unidecode(s) if isinstance(s, str) else s

# Apply transformation to 'query' and 'doc_content' columns
ascii_candidate_docs_full['query'] = ascii_candidate_docs_full['query'].apply(convert_to_ascii)
ascii_candidate_docs_full['doc_content'] = ascii_candidate_docs_full['doc_content'].apply(convert_to_ascii)

#Check 
non_ascii_rows = ascii_candidate_docs_full[
    ascii_candidate_docs_full[['query', 'doc_content']].applymap(has_non_ascii).any(axis=1)
]

#_______________________________________________________________________________________________________________________


filtered_df = ascii_candidate_docs_full[
    (ascii_candidate_docs_full['query_id'] == 2235) & 
    (ascii_candidate_docs_full['doc_id'] == 4528126)
]['doc_content'].values[0]
#_______________________________________________________________________________________________________________________

#utf8

def convert_to_utf8(df, columns):
    """
    Converts specified columns in a DataFrame to UTF-8 encoding.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to be converted.

    Returns:
        pd.DataFrame: DataFrame with UTF-8 encoded text.
    """
    def safe_decode(text):
        if isinstance(text, bytes):  # If the text is in byte format
            return text.decode('utf-8', errors='replace')  # Decode using UTF-8
        elif isinstance(text, str):  # If already a string
            return text.encode('latin-1', errors='replace').decode('utf-8', errors='replace')
        return text  # Return unchanged if it's not a string or bytes

    for col in columns:
        df[col] = df[col].astype(str).apply(safe_decode)  # Convert each column safely

    return df

ascii_candidate_docs_full = candidate_docs
ascii_candidate_docs_full = convert_to_utf8(ascii_candidate_docs_full, ["query", "doc_content"])

filtered_df = ascii_candidate_docs_full[
    (ascii_candidate_docs_full['query_id'] == 2235) & 
    (ascii_candidate_docs_full['doc_id'] == 4528126)
]['doc_content'].values[0]
filtered_df

#_______________________________________________________________________________________________________________________
# has quotes
#'''

def has_quotes(s):
    return '"' in s if isinstance(s, str) else False

# Filter rows where 'query' or 'doc_content' contains single or double quotes
quote_rows = ascii_candidate_docs_full[
    ascii_candidate_docs_full[['query', 'doc_content']].applymap(has_quotes).any(axis=1)
]
#quote_rows.to_csv(f'/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/data/quote_rows.csv')

#_______________________________________________________________________________________________________________________

def json_safe_text(text):
    if isinstance(text, str):
        text = text.replace('"', '\\"')  # Escape double quotes
        text = text.replace("'", "\\'")  # Escape single quotes (optional)
        return text.encode("unicode_escape").decode("utf-8")  # Ensure other special characters are escaped
    return text

converted_candidate_docs_full['query'] = converted_candidate_docs_full['query'].apply(json_safe_text)
converted_candidate_docs_full['doc_content'] = converted_candidate_docs_full['doc_content'].apply(json_safe_text)
