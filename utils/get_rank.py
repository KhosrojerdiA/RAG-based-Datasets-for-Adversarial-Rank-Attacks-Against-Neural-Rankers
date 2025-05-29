import sys
import os
import time

os.system("nvidia-smi")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

project_path = '/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval'
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

chunk_number = 1

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
chunk_number = 1
candidate_docs_full = pd.read_csv('/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/data/candidate_docs_full.csv')
candidate_docs_full = candidate_docs_full.drop(columns=["Unnamed: 0"], errors="ignore")

my_data = pd.read_csv(f'/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/test_output/phase_1_chunk_{chunk_number}_no_think.csv')


my_data_old_and_new_rank = add_old_and_new_unique_ranks(my_data, candidate_docs_full)
# ___________________________________________________________________________________________________________________________________________

# Load GPT2-based perplexity scorer (assuming GPT2PPLScorer is defined somewhere)
# You must define GPT2PPLScorer yourself based on your original script.

# Example placeholder class:
class GPT2PPLScorer:
    def __init__(self, device):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()
        self.device = device

    def perplexity(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return [None]
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            loss = outputs.loss
        return [torch.exp(loss).item()]

ppl_scorer = GPT2PPLScorer(device)

tqdm.pandas()

full = my_data_old_and_new_rank.copy()
full['perplexity_new_doc'] = full['new_doc_content'].progress_apply(lambda x: ppl_scorer.perplexity(x)[0] if isinstance(x, str) else None)
full['perplexity_old_doc'] = full['doc_content'].progress_apply(lambda x: ppl_scorer.perplexity(x)[0] if isinstance(x, str) else None)

# Load acceptability model
model_name = "textattack/roberta-base-CoLA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

def predict_acceptability(document):
    if not isinstance(document, str) or len(document.strip()) == 0:
        return None, None
    inputs = tokenizer(document, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        acceptability_score = probabilities[1].item()
        predicted_label = torch.argmax(probabilities).item()
    return acceptability_score, predicted_label

tqdm.pandas()
full[['acceptability_score_new_doc', 'predicted_label_new_doc']] = full['new_doc_content'].progress_apply(lambda x: pd.Series(predict_acceptability(x)))
full[['acceptability_score_old_doc', 'predicted_label_old_doc']] = full['doc_content'].progress_apply(lambda x: pd.Series(predict_acceptability(x)))

# ___________________________________________________________________________________________________________________________________________

full = full[[
    'query_id', 'query', 'doc_id', 'doc_content', 'score', 'rank', 'actual_old_rank',
    'doc_context', 'new_sent', 'new_sent_position', 'new_doc_content', 'new_score',
    'new_rank', 'query_doc_top_ranks', 'query_doc_for_begining_top_ranks','actual_new_rank', 
    'perplexity_new_doc', 'perplexity_old_doc', 
    'acceptability_score_new_doc', 'predicted_label_new_doc','acceptability_score_old_doc', 'predicted_label_old_doc',
    'key_phrases_buffer_A', 'key_phrases_buffer_B'
]]

output_path = f'/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval/test_output/reranked_perplexity_phase_1_chunk_{chunk_number}_no_think.csv'
full.to_csv(output_path, index=False)


# ___________________________________________________________________________________________________________________________________________

