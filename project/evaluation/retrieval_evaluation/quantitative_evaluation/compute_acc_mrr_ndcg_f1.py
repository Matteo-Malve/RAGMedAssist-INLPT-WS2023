
"""
This script computes retrieval accuracy, MRR, nDCG, and F1 scores for different embedding models, using the FAISS vector base as retriever, and plots & saves results.
"""

from generate_embeddings import generate_embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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


# Eval data
def load_eval_data(eval_path):
    eval_data = pd.read_csv(eval_path)
    eval_data.rename(columns={eval_data.columns[0]: 'PMID'}, inplace=True)
    gold_pmids = eval_data['PMID'].to_list()
    eval_queries = eval_data['QUESTION'].to_list()

    return gold_pmids, eval_queries

# Initialize vector database
def get_vector_database(index_path=None):
    # Check if embeddings already saved in FAISS index
    if index_path:
        # Load embedded 
        return FAISS.load_local(f"{index_path}/{model_name}", embed_model)
    else:
        generate_embeddings(model_name)
        return FAISS.load_local(f"faiss_indices/model_name", embed_model)


# Accuracy
def compute_plot_accuracy(model_name, eval_path, plot_title, file_name):
    # Extract gold labels and queries
    gold_pmids, eval_queries = load_eval_data(eval_path)

    # Compute accuracy
    accuracies = []
    for k in range(1, 21):
        # Define retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        matches = 0
        for query, gold_label in zip(eval_queries, gold_pmids):
            top_k_results = retriever.get_relevant_documents(query)
            retrieved_pmids = [int(result.metadata['PMID']) for result in top_k_results]
            if gold_label in retrieved_pmids:
                matches += 1

        accuracy = matches / len(eval_queries)
        accuracies.append(accuracy)

    # Save accuracies to a text file
    file_path = file_name.replace('.png', '.txt')
    with open(file_path, 'w') as f:
        for accuracy in accuracies:
            f.write(f"{accuracy}\n")

    # Plotting the results
    plt.plot(range(1, 21), accuracies, marker='o')
    plt.xlabel('k (Number of Top Results Considered)')
    plt.ylabel('Accuracy')
    plt.title(plot_title)
    plt.xticks(range(1, 21))

    plt.savefig(file_name)


# MRR
def compute_mrr(eval_path, file_name):
    # Extract gold labels and queries
    gold_pmids, eval_queries = load_eval_data(eval_path)

    # Define retriever
    retriever = db.as_retriever(search_type="similarity")

    mrr = 0
    for query, gold_label in zip(eval_queries, gold_pmids):
        top_k_results = retriever.get_relevant_documents(query)
        retrieved_pmids = [int(result.metadata['PMID']) for result in top_k_results]
        for rank, pmid in enumerate(retrieved_pmids, start=1):
            if pmid == gold_label:
                mrr += 1 / rank
                break

    mrr /= len(eval_queries)
    print(mrr)

    # Save MRRs to a text file
    with open(f"retriever_MRR_{file_name}.txt", 'w') as f:
        f.write(f"{mrr}")


# nDCG
def compute_ndcg(eval_path, file_name):
    # Extract gold labels and queries
    gold_pmids, eval_queries = load_eval_data(eval_path)

    ndcg_list = []
    for k in range(1, 4):
         # Define retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

        dcg_values = []
        for query, gold_label in zip(eval_queries, gold_pmids):
            top_k_results = retriever.get_relevant_documents(query)
            retrieved_pmids = [int(result.metadata['PMID']) for result in top_k_results]
            # DCG calculation
            dcg = sum([1 / np.log2(rank + 1) if pmid == gold_label else 0 for rank, pmid in enumerate(retrieved_pmids, start=2)])
            dcg_values.append(dcg)

        # Assuming ideal DCG (iDCG) is the best case where the first document is always relevant
        idcg = 1 / np.log2(2)  # Best case for top-1 result being relevant
        ndcg = np.mean([dcg / idcg for dcg in dcg_values])
        ndcg_list.append(ndcg)

    # Save nDCGs to a text file
    with open(f"retriever_nDCG_{file_name}.txt", 'w') as f:
        for ndcg in ndcg_list:
            f.write(f"{ndcg}\n")


# F1
def compute_plot_f1(eval_path, file_name):
    # Extract gold labels and queries
    gold_pmids, eval_queries = load_eval_data(eval_path)

    f1_list = []
    for k in range(1, 11):
        precision_at_k = []
        recall_at_k = []
        for query, gold_label in zip(eval_queries, gold_pmids):
            # Define retriever
            retriever = db.as_retriever(search_type="similarity")

            top_k_results = retriever.get_relevant_documents(query)
            retrieved_pmids = [int(result.metadata['PMID']) for result in top_k_results[:k]]
            true_positives = sum([1 for pmid in retrieved_pmids if pmid == gold_label])
            precision = true_positives / k
            recall = true_positives / 1  #assuming one relevant document per query

            precision_at_k.append(precision)
            recall_at_k.append(recall)

        mean_precision = np.mean(precision_at_k)
        mean_recall = np.mean(recall_at_k)
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0
        f1_list.append(f1_score)

    # Save F1s to a text file
    with open(f"retriever_F1_{file_name}.txt", 'w') as f:
        for f1 in f1_list:
            f.write(f"{f1}\n")

    # Plot results
    plt.plot(range(1, 11), f1_list, marker='o')
    plt.xlabel('k (Number of Top Results Considered)')
    plt.ylabel('F1 at k')
    plt.title(f'Retriever Model F1 Score with {model__name}')
    plt.xticks(range(1, 11))
    plt.savefig(f"retriever_f1_{file_name}.png")



if __name__ == "__main__":

    # List of models to evaluate
    models_to_evaluate = [
        # {
        #     'model_name': 'Muennighoff/SGPT-125M-weightedmean-nli-bitfit',
        #     'plot_title': 'Retriever Model Accuracy with SGPT-125M',
        #     'acc_file': 'retriever_accuracy_Muennighoff_SGPT-125M-weightedmean-nli-bitfit.png',
        #     'mrr_file': 'retriever_MRR_tMuennighoff_SGPT-125M-weightedmean-nli-bitfit',
        #     'ndcg_file': 'retriever_nDCG_tMuennighoff_SGPT-125M-weightedmean-nli-bitfit',
        #     'f1_file': 'retriever_F1_Muennighoff_SGPT-125M-weightedmean-nli-bitfit',
        # },
        # {
        #     'model_name': 'dmis-lab/biobert-base-cased-v1.1',
        #     'plot_title': 'Retriever Model Accuracy with BioBERT',
        #     'acc_file': 'retriever_accuracy_dmis-lab_biobert-base-cased-v1.1.png',
        #     'mrr_file': 'retriever_MRR_dmis-lab_biobert-base-cased-v1.1',
        #     'ndcg_file': 'retriever_nDCG_dmis-lab_biobert-base-cased-v1.1',
        #     'f1_file': 'retriever_F1_dmis-lab_biobert-base-cased-v1.1',
        # },
        # {
        #     'model_name': 'intfloat/e5-base-v2',
        #     'plot_title': 'Retriever Model Accuracy with e5-base-v2',
        #     'acc_file': 'retriever_accuracy_intfloat_e5-base-v2.png',
        #     'mrr_file': 'retriever_MRR_intfloat_e5-base-v2',
        #     'ndcg_file': 'retriever_nDCG_intfloat_e5-base-v2',
        #     'f1_file': 'retriever_F1_intfloat_e5-base-v2',
        # },
        # {
        #     'model_name': 'BAAI/bge-base-en-v1.5',
        #     'plot_title': 'Retriever Model Accuracy with bge-base-en-v1.5',
        #     'acc_file': 'retriever_accuracy_BAAI_bge-base-en-v1.5.png',
        #     'mrr_file': 'retriever_MRR_BAAI_bge-base-en-v1.5',
        #     'ndcg_file': 'retriever_nDCG_BAAI_bge-base-en-v1.5',
        #     'f1_file': 'retriever_F1_BAAI_bge-base-en-v1.5',
        # },
        # {
        #     'model_name': 'llmrails/ember-v1',
        #     'plot_title': 'Retriever Model Accuracy with llmrails/ember-v1',
        #     'acc_file': 'retriever_accuracy_llmrails_ember-v1.png',
        #     'mrr_file': 'retriever_MRR_llmrails_ember-v1',
        #     'ndcg_file': 'retriever_nDCG_llmrails_ember-v1',
        #     'f1_file': 'retriever_F1_llmrails_ember-v1',
        # },
        # {
        #     'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        #     'plot_title': 'Retriever Model Accuracy with all-MiniLM-L6-v2',
        #     'acc_file': 'retriever_accuracy_sentence-transformers_all-MiniLM-L6-v2.png',
        #     'mrr_file': 'retriever_MRR_tsentence-transformers_all-MiniLM-L6-v2',
        #     'ndcg_file': 'retriever_nDCG_sentence-transformers_all-MiniLM-L6-v2',
        #     'f1_file': 'retriever_F1_sentence-transformers_all-MiniLM-L6-v2',
        # },
        # {
        #     'model_name': 'jamesgpt1/sf_model_e5',
        #     'plot_title': 'Retriever Model Accuracy with sf_model_e5',
        #     'acc_file': 'retriever_accuracy_jamesgpt1_sf_model_e5.png',
        #     'mrr_file': 'retriever_MRR_jamesgpt1_sf_model_e5',
        #     'ndcg_file': 'retriever_nDCG_jamesgpt1_sf_model_e5',
        #     'f1_file': 'retriever_F1_jamesgpt1_sf_model_e5',
        # },
        {
            'model_name': 'thenlper/gte-base',
            'plot_title': 'Retriever Model Accuracy with gte-base',
            'acc_file': 'retriever_accuracy_thenlper_gte-base.png',
            'mrr_file': 'retriever_MRR_thenlper_gte-base',
            'ndcg_file': 'retriever_nDCG_thenlper_gte-base',
            'f1_file': 'retriever_F1_thenlper_gte-base',
        },
    ]


    for model_info in models_to_evaluate:
        # Define embedding model
        model_name = model_info['model_name']
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        embed_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

        db = get_vector_database(index_path="faiss_indices")

        compute_plot_accuracy(model_name, "questions_answers.csv", model_info['plot_title'], model_info['acc_file'])
        
        compute_mrr("questions_answers.csv", model_info['mrr_file'])

        compute_ndcg("questions_answers.csv", model_info['ndcg_file'])

        compute_plot_f1("questions_answers.csv", model_info['f1_file'])