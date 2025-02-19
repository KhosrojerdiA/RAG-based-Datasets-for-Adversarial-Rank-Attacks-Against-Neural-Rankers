import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


project_path = '/home/akhosrojerdi/Amin'
sys.path.append(project_path)



import json
import faiss
from sentence_transformers import SentenceTransformer, util
import torch
import os
import argparse


if not torch.cuda.is_available():
    print("Warning: No GPU found. The encoding process may be slower.")


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Name of the Sentence Transformer model")
parser.add_argument("--collection_folder", type=str, required=True, help="Folder containing the passage collection")
parser.add_argument("--collection_name", type=str, required=True, help="Name of the collection file (without .tsv)")
parser.add_argument("--output_folder", type=str, default="output", help="Folder to store embeddings and index")
args = parser.parse_args()


model_name = args.model_name
collection_folder = args.collection_folder
collection_name = args.collection_name
output_folder = args.output_folder


print(f"Loading Sentence Transformer model: {model_name}")
bi_encoder = SentenceTransformer(model_name)


collection_filepath = os.path.join(collection_folder, collection_name + ".tsv")
if not os.path.exists(collection_filepath):
    raise FileNotFoundError(f"Collection file not found at {collection_filepath}")


os.makedirs(output_folder, exist_ok=True)

passages = []
print(f"Loading passages from {collection_filepath}...")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        try:
            pid, passage = line.strip().split("\t")
            passages.append(passage)
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")

print(f"Total passages loaded: {len(passages)}")


batch_size = 128
embedding_dim = bi_encoder.get_sentence_embedding_dimension() 
index = faiss.IndexFlatL2(embedding_dim)

print("Encoding and saving passage embeddings in chunks...")
num_passages = len(passages)
chunk_size = 1_000_000
for x in range(0, num_passages, chunk_size):
    chunk = passages[x:x + chunk_size]
    print(f"Encoding chunk {x // chunk_size + 1}: {len(chunk)} passages")
    corpus_embeddings = bi_encoder.encode(
        chunk,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size
    )
    

    chunk_filename = os.path.join(output_folder, f"corpus_tensor_{x // chunk_size + 1}.pt")
    torch.save(corpus_embeddings, chunk_filename)
    print(f"Saved embeddings to {chunk_filename}")


    print("Adding embeddings to FAISS index...")
    index.add(corpus_embeddings.cpu().numpy())

print(f"Total embeddings added to FAISS index: {index.ntotal}")


faiss_index_path = os.path.join(output_folder, f"{collection_name}_faiss.index")
print(f"Saving FAISS index to {faiss_index_path}...")
faiss.write_index(index, faiss_index_path)

print("Indexing complete!")
#_______________________________________________________________________________

