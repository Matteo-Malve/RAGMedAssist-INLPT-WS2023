# 📈 evaluation

- 🗂️ [`llm_evaluation`](evaluation/llm_evaluation): This folder contains all evaluation files of our chatmodel. 

    - 🗂️ [`compare_retrievers`](evaluation/llm_evaluation/compare_retrievers): This folder contains evaluation of our retrievers (FAISS vs. Pinecone).
    - 🗂️ [`edge_cases`](evaluation/llm_evaluation/edge_cases): This folder contains evaluation of query edge cases.
    - 🗂️ [`hybrid_search`](evaluation/llm_evaluation/hybrid_search): This folder contains evaluation of the hybrid search method: we tested different weights for keyword and vector search.
    - 🗂️ [`prompt_engineering`](evaluation/llm_evaluation/prompt_engineering): This folder contains evaluation of our prompt engineering in order to find the best prompt.
    - 🗂️ [`question_types`](evaluation/llm_evaluation/question_types): This folder contains evaluation of different question types (e.g. causal, complex, factoid).
    - 📜 [`EVAL_QUESTION.md`](evaluation/llm_evaluation/RETRIEVAL_EVALUATION.md): This README contains a list of all the questions we used for the evaluation of our LLM.


- 🗂️ [`retrieval_evaluation`](evaluation/retrieval_evaluation): This folder contains all evaluation files of our retrieval system.

    - 🗂️ [`qualitative_evaluation`](evaluation/retrieval_evaluation/qualitative_evaluation): Folder for our qualitative evaluation.

        - 📸 [`images`](evaluation/retrieval_evaluation/qualitative_evaluation/images): Folder containing images related to our qualitative evaluation.
        - 🗂️ [`results`](evaluation/retrieval_evaluation/qualitative_evaluation/results): Folder containing result files.
        - 💻 [`qualitative_evaluation.ipynb`](valuation/retrieval_evaluation/qualitative_evaluation/qualitative_evaluation.ipynb): This code evaluates our retrieval system qualitatively.

    - 🗂️ [`quantitative_evaluation`](evaluation/retrieval_evaluation/quantitative_evaluation): Folder for our quantitative evaluation.

        - 📸 [`images`](evaluation/retrieval_evaluation/quantitative_evaluation/images): Folder containing images related to our quantitative evaluation.
        - 🗂️ [`results`](evaluation/retrieval_evaluation/quantitative_evaluation/results): Folder containing result files.
        - 💻 [`compare_against_keyword_search.ipynb`](evaluation/retrieval_evaluation/quantitative_evaluation/compare_against_keyword_search.ipynb): This code computes scores of the keyword search retriever in order to compare it to our vector search.
        - 💻 [`compute_acc_mrr_ndcg_f1.py`](evaluation/retrieval_evaluation/quantitative_evaluation/compute_acc_mrr_ndcg_f1.py): This code computes quantitative scores to evaluate the different embedding models we have tested.

    - 📜 [`RETRIEVAL_EVALUATION.md`](evaluation/retrieval_evaluation/RETRIEVAL_EVALUATION.md): This README contains our detailed retrieval evaluation results.

