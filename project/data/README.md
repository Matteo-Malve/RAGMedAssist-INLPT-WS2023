# 📊 data

- 🗂️ [`embeddings`](embeddings): This folder contains code for generating, uploading, and saving embeddings.

    - 🗂️ [`faiss_indices`](embeddings/faiss_indices): This folder contains the saved FAISS embeddings.

    - 💽 [`chunked_docs.pkl`](embeddings/chunked_docs.pkl): Documents chunked via text splitting, later used by the BM25 retriever.

    - 💻 [`generate_embeddings.py`](embeddings/generate_embeddings.py): This code chunks documents and generates embeddings, finally saving them in [`faiss_indices`](embeddings/faiss_indices).

    - 💻 [`upload_thenlper_gte_embs_into_pinecone.ipynb`](embeddings/upload_thenlper_gte_embs_into_pinecone.ipynb): This code, while not currently integrated into our chatbot, uploads embeddings to Pinecone. It serves as an alternative for those preferring Pinecone over FAISS for vector search.

- 🗂️ [`evaluation_data`](evaluation_data): This folder contains data used for evaluation.

    - 🗂️ [`retrieval_eval_data`](evaluation_data/retrieval_eval_data): This folder contains data used to evaluate our retrieval system.

        - 💻 [`download_questions_answers.ipynb`](evaluation_data/retrieval_eval_data/download_questions_answers.ipynb): This code downloads the QA evaluation set.
        - 💽 [`questions_answers.csv`](evaluation_data/retrieval_eval_data/questions_answers.csv): QA pairs used as gold truths for retrieval evaluation.


- 🗂️ [`original_pubmed_data`](original_pubmed_data): this folder contains the original PubMed data.

    - 💻 [`download_pubmed_data.ipynb`](original_pubmed_data/download_pubmed_data.ipynb): This code downloads our original dataset from PubMed.
    - 💽 [`pubmed_data_part1.csv`](original_pubmed_data/pubmed_data_part1.csv): Part 1 of our dataset.
    - 💽 [`pubmed_data_part2.csv`](original_pubmed_data/pubmed_data_part2.csv): Part 2 of our dataset.

- 🗂️ [`preprocessing_and_analytics`](preprocessing_and_analytics): This folder contains code for pre-processing and analyzing our data.

    - 💻 [`data_anaylitics.ipynb`](preprocessing_and_analytics/data_anaylitics.ipynb): This code analyzes our data and generates various plots.
    - 💻 [`preprocess_data.ipynb`](preprocessing_and_analytics/preprocess_data.ipynb): This code pre-processes and cleans our dataset.
    - 💽 [`processed_data_part1.csv`](preprocessing_and_analytics/processed_data_part1.csv): Part 1 of our pre-processed dataset.
    - 💽 [`processed_data_part2.csv`](preprocessing_and_analytics/processed_data_part2.csv): Part 2 of our pre-processed dataset.