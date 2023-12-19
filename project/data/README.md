## Code Files

- **📄 [`download_pubmed_data.py`](project/data/download_pubmed_data.py):** Downloads abstracts and metadata from the PubMed website and saves it to a csv file ([abstracts.csv](project/data/abstracts.csv)).

- **📄 [`parse_abstracts.py`](project/data/parse_abstracts.py):** Parses the [abstracts.csv](project/data/abstracts.csv) file to bring it in the desired Pandas dataframe format used for further processing.

- **📄 [`preprocess_parsed_abstracts.py`](project/data/preprocess_parsed_abstracts.py):** Applies preprocessing steps to the parsed data.


## Data Files

- **💽 [`pmid-intelligen-set.txt`](project/data/pmid-intelligen-set.txt):** Contains all PMIDs of our documents.

- **💽 [`pubmed-intelligen-set.txt`](project/data/pubmed-intelligen-set.txt):** Contains the downloaded, unparsed abstracts with all their metadata.

- **💽 [`csv-intelligen-set.csv`](project/data/csv-intelligen-set.csv):** Contains the downloaded metadata without the abstracts.

- **💽 [`abstracts.csv`](project/data/abstracts.csv):** Contains the downloaded, unparsed abstracts with all their metadata.

- **💽 [`parsed_abstracts.csv`](project/data/parsed_abstracts.csv):** Contains abstracts with relevant metadata such as authors, title, publication date and DOI.
