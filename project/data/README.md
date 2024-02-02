## Code Files

- **📄 [`download_pubmed_data.py`](download_pubmed_data.py):** Downloads abstracts and metadata from the PubMed website and saves it to a csv file ([pubmed_data.csv](pubmed_data.csv)).

- **📄 [`preprocess_data.py`](preprocess_data.py):** Applies preprocessing steps to the parsed data.


## Data Files

- **💽 [`pubmed_data.csv`](pubmed_data.csv):** Contains the downloaded abstracts with all their metadata.

- **💽 [`processed_data.csv`](processed_data.csv):** Contains preprocessed abstracts with relevant metadata such as authors, title, publication date and DOI.

- **🗂️ [`question_answers`](question_answers):**

    - **🗂️ [`download_questions_answers.ipynb`](questions_answers/download_questions_answers.ipynb):** Downloads evaluation data set.

     - **🗂️ [`questions_answers.csv`](questions_answers/questions_answers.csv):** Evaluation data set used for our quantitative and qualitative experiments to determine the best embedding model for retrieval.
