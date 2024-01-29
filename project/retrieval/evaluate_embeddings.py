"""
This script generates embeddings with different models and evaluates them using a Pubmed QA-set.
"""

print("start")

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json


print("imports loaded")

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



# Load data
loader_1 = CSVLoader(
    file_path='processed_data_part1.csv',
    metadata_columns=['PMID', 'Title', 'Authors', 'Publication Date', 'DOI'])
loader_2 = CSVLoader(
    file_path='processed_data_part2.csv',
    metadata_columns=['PMID', 'Title', 'Authors', 'Publication Date', 'DOI'])

loader_all = MergedDataLoader(loaders=[loader_1, loader_2])
docs_all = loader_all.load()

eval_data = pd.read_csv('questions_answers.csv')
eval_data.rename(columns={eval_data.columns[0]: 'PMID'}, inplace=True)


# Evaluate
def evaluate_and_plot(model_name, tokenizer, model, plot_title, file_name):
    # Define embedding model
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    embed_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    # Apply text splitting into chunks to prevent truncation of longer abstracts
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def token_len(text):
        tokens = tokenizer.tokenize(text)
        tokens_length = len(tokens)

        return tokens_length

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,              #target size for each chunk of text
    #                                             chunk_overlap=100,            #specifies how much overlap there should be between consecutive chunks
    #                                             length_function=token_len,   #counts the number of characters in the text using the token_len function
    #                                             is_separator_regex=False,)   #whether the splitter should treat the separators as regular expressions

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, 
                                                                         chunk_overlap=100,)
    chunked_docs = text_splitter.split_documents(docs_all)


    # Set up Faiss vector database
    db = FAISS.from_documents(chunked_docs, embedding=embed_model)

    # Extract gold labels and queries
    gold_pmids = eval_data['PMID'].to_list()
    eval_queries = eval_data['QUESTION'].to_list()

    accuracies = []

    for k in range(1, 21):
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        matches = 0

        for query, gold_label in zip(eval_queries, gold_pmids):
            top_k_results = retriever.get_relevant_documents(query)
            retrieved_pmids = [int(result.metadata['PMID']) for result in top_k_results]

            if gold_label in retrieved_pmids:
                matches += 1

        accuracy = matches / len(eval_queries)
        accuracies.append(accuracy)
    
    print(accuracies)

    # Plotting the results
    plt.plot(range(1, 21), accuracies, marker='o')
    plt.xlabel('k (Number of Top Results Considered)')
    plt.ylabel('Accuracy')
    plt.title(plot_title)
    plt.xticks(range(1, 21))

    plt.savefig(file_name)


# List of models to evaluate
models_to_evaluate = [
    # {
    #     'model_name': 'Muennighoff/SGPT-125M-weightedmean-nli-bitfit',
    #     'tokenizer': AutoTokenizer.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit"),
    #     'model': SentenceTransformer("Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit"),
    #     'plot_title': 'Retriever Model Accuracy with SGPT-125M',
    #     'file_name': 'retriever_accuracy_sgpt_msmarco.png'
    # },
    # {
    #     'model_name': 'dmis-lab/biobert-base-cased-v1.1',
    #     'tokenizer': AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1"),
    #     'model': AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1"),
    #     'plot_title': 'Retriever Model Accuracy with BioBERT',
    #     'file_name': 'retriever_accuracy_biobert.png'
    # },
    # {
    #     'model_name': 'intfloat/e5-base-v2',
    #     'tokenizer': AutoTokenizer.from_pretrained("intfloat/e5-base-v2"),
    #     'model': SentenceTransformer("intfloat/e5-base-v2"),
    #     'plot_title': 'Retriever Model Accuracy with e5-base-v2',
    #     'file_name': 'retriever_accuracy_e5-base-v2.png'
    # },
    # {
    #     'model_name': 'BAAI/bge-base-en-v1.5',
    #     'tokenizer': AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5"),
    #     'model': SentenceTransformer("BAAI/bge-base-en-v1.5"),
    #     'plot_title': 'Retriever Model Accuracy with bge-base-en-v1.5',
    #     'file_name': 'retriever_accuracy_bge-base-en-v1.5.png'
    # },
    # {
    #     'model_name': 'llmrails/ember-v1',
    #     'tokenizer': AutoTokenizer.from_pretrained("llmrails/ember-v1"),
    #     'model': SentenceTransformer("llmrails/ember-v1"),
    #     'plot_title': 'Retriever Model Accuracy with llmrails/ember-v1',
    #     'file_name': 'retriever_accuracy_ember-v1.png'
    # },
    {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'tokenizer': AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
        'model': SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        'plot_title': 'Retriever Model Accuracy with all-MiniLM-L6-v2',
        'file_name': 'retriever_accuracy_all-MiniLM-L6-v2.png'
    },
    {
        'model_name': 'jamesgpt1/sf_model_e5',
        'tokenizer': AutoTokenizer.from_pretrained("jamesgpt1/sf_model_e5"),
        'model': SentenceTransformer("jamesgpt1/sf_model_e5"),
        'plot_title': 'Retriever Model Accuracy with sf_model_e5',
        'file_name': 'retriever_accuracy_sf_model_e5.png'
    },
    {
        'model_name': 'thenlper/gte-base',
        'tokenizer': AutoTokenizer.from_pretrained("thenlper/gte-base"),
        'model': SentenceTransformer("thenlper/gte-base"),
        'plot_title': 'Retriever Model Accuracy with gte-base',
        'file_name': 'retriever_accuracy_gte-base.png'
    },    
    
    
]




# Loop over the models
for model_info in models_to_evaluate:
    evaluate_and_plot(model_info['model_name'], model_info['tokenizer'], model_info['model'], model_info['plot_title'], model_info['file_name'])


