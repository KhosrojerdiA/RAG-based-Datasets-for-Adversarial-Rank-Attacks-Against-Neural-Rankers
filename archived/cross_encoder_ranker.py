import sys
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"


project_path = '/mnt/data/khosro/Amin/code'
sys.path.append(project_path)

#_________________________________________________________________________________________
import torch
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document 
from langchain.schema import HumanMessage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#_________________________________________________________________________________________

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Name of the Sentence Transformer model")
parser.add_argument("--faiss_index_path", type=str, required=True, help="Path to the FAISS index file")
parser.add_argument("--query_file", type=str, required=True, help="Path to the file containing queries (one per line)")
parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
parser.add_argument("--query_counts", type=int, default=5, help="Number of top results to retrieve")
args = parser.parse_args()



model_name  = "sentence-transformers/msmarco-distilbert-base-tas-b"
faiss_index_path = "/home/akhosrojerdi/Amin/output/collection_faiss.index"
query_file =  "/home/akhosrojerdi/Amin/data/queries.dev.small.tsv"
sample_query = "/home/akhosrojerdi/Amin/data/msmarco-passage.dev.small.attack.1000-query-id.txt"
top_k = 1000 
#_________________________________________________________________________________________

# Load Sentence Transformer model

print(f"Loading Sentence Transformer model: {model_name}")
model = SentenceTransformer(model_name).to(device)

#_________________________________________________________________________________________

# Load FAISS index

print(f"Loading FAISS index from {faiss_index_path}...")
index = faiss.read_index(faiss_index_path)  
print("FAISS index loaded successfully!")


#_________________________________________________________________________________________

# Load queries

with open(sample_query, 'r') as f:
    query_ids = set(line.strip() for line in f)

print(f"Loaded {len(query_ids)} query IDs from Sample File.")

print(f"Loading queries from {query_file}...")
queries = []
with open(query_file, 'r') as f:
    for line in f:
        query_id, query_text = line.strip().split('\t', 1)  # Assuming tab-separated values
        if query_id in query_ids:
            queries.append((query_id, query_text))

print(f"Total queries loaded: {len(queries)}")

#_________________________________________________________________________________________

# Encode queries

print("Encoding queries...")
query_texts = [q[1] for q in queries]  # Extract query texts for encoding
query_embeddings = model.encode(query_texts, convert_to_numpy=True, show_progress_bar=True)



#_________________________________________________________________________________________

# Perform retrieval

print(f"Retrieving top-{top_k} results for each query...")
results = []
for i, query_embedding in enumerate(query_embeddings):
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    results.append({
        "query": queries[i],
        "results": indices[0].tolist(),  # Document indices in the collection
        "distances": distances[0].tolist()
    })

#_________________________________________________________________________________________

# Result as DataFrame

result_info = []
for result in results:
    query_id, query_text = result['query']
    for rank, (doc_id, distance) in enumerate(zip(result['results'], result['distances']), start=1):
        result_info.append({
            'query_id': query_id,
            'query': query_text,
            'doc_id': doc_id,
            'rank': rank,
            'distance': distance
        })

result_df = pd.DataFrame(result_info)

print(result_df)

result_df.to_excel("/home/akhosrojerdi/Amin/output/1000_query_1000_documents_Rank.xlsx", index=False)