"""
This script chunks documents, generates embeddings and stores & saves them via FAISS vector database.
"""

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
import torch
import random
import pandas as pd
from tqdm import tqdm
import json
import pickle



# Set seed for vector database
random.seed(42)
torch.manual_seed(42)

# Check device
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(device)

if has_gpu:
    torch.cuda.manual_seed_all(42)


def generate_embeddings(model_name, chunk_path=None):

    # Load data
    loader_1 = CSVLoader(
        file_path='processed_data_part1.csv',
        metadata_columns=['PMID', 'Title', 'Authors', 'Publication Date', 'DOI'])
    loader_2 = CSVLoader(
        file_path='processed_data_part2.csv',
        metadata_columns=['PMID', 'Title', 'Authors', 'Publication Date', 'DOI'])

    loader_all = MergedDataLoader(loaders=[loader_1, loader_2])
    docs_all = loader_all.load()

    # Define embedding model
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    embed_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    # Apply text splitting into chunks to prevent truncation of longer abstracts
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400,
                                                                            chunk_overlap=100,)
    chunked_docs = text_splitter.split_documents(docs_all)

    if chunk_path:
        # Save chunked docs for BM25 retriever later
        with open(chunk_path, "wb") as file:  # Note the 'wb' mode for writing binary
            pickle.dump(chunked_docs, file)

    # Set up Faiss vector database
    db = FAISS.from_documents(chunked_docs, embedding=embed_model)

    #Save embeddings locally
    index_save_path = f"faiss_indices/{model_name}"
    db.save_local(index_save_path)



if __name__ == "__main__":
    generate_embeddings("thenlper/gte-base", chunk_path="chunked_docs.pkl")


