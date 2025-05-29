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



full = pd.read_csv('/mnt/data/khosro/Amin/get_rank_data/FULL_v3.csv')
full.columns


device = "cuda" if torch.cuda.is_available() else "cpu"
ppl_scorer = GPT2PPLScorer(device)

# Compute perplexity for each new_doc_content
tqdm.pandas()
full['perplexity_new_doc'] = full['new_doc_content'].progress_apply(lambda x: ppl_scorer.perplexity(x)[0] if isinstance(x, str) else None)

tqdm.pandas()
full['perplexity_old_doc'] = full['doc_content'].progress_apply(lambda x: ppl_scorer.perplexity(x)[0] if isinstance(x, str) else None)
full.to_csv(f'/mnt/data/khosro/Amin/get_rank_data/FULL_v4.csv')


# ___________________________________________________________________________________________________________________________________________

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Load RoBERTa-based acceptability model
model_name = "textattack/roberta-base-CoLA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # Set model to evaluation mode

def predict_acceptability(document):
    """Predicts the linguistic acceptability score and classification label (acceptable or not)."""
    if not isinstance(document, str) or len(document.strip()) == 0:
        return None, None  # Handle empty/missing values gracefully

    inputs = tokenizer(document, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        acceptability_score = probabilities[1].item()  # Probability of being "acceptable"
        predicted_label = torch.argmax(probabilities).item()  # 1 = acceptable, 0 = not acceptable
    return acceptability_score, predicted_label

# Compute acceptability scores for `new_doc_content`
tqdm.pandas()
full[['acceptability_score_new_doc', 'predicted_label_new_doc']] = full['new_doc_content'].progress_apply(
    lambda x: pd.Series(predict_acceptability(x))
)

tqdm.pandas()
full[['acceptability_score_old_doc', 'predicted_label_old_doc']] = full['doc_content'].progress_apply(
    lambda x: pd.Series(predict_acceptability(x))
)
full.to_csv(f'/mnt/data/khosro/Amin/get_rank_data/FULL_v4.csv')