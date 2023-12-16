import pandas as pd
import re


def parse_document(document):
    pmid = re.search(r'PMID- (\d+)', document)
    date = re.search(r'DP  - (\d{4} \w{3} \d{1,2})', document)
    title = re.search(r'TI  - (.+?)\n', document, re.DOTALL)
    abstract = re.search(r'AB  - (.*?)\n[A-Z]{2,4}  - ', document, re.DOTALL)
    authors = re.findall(r'FAU - (.+)', document)
    doi = re.search(r'LID - (.+?) \[doi\]', document)

    return {
        'PMID': pmid.group(1) if pmid else None,
        'Date': date.group(1) if date else None,
        'Title': title.group(1).replace('\n      ', ' ') if title else None,
        'Abstract': abstract.group(1).replace('\n      ', ' ') if abstract else None,
        'Authors': '; '.join(authors) if authors else None,
        'DOI': doi.group(1) if doi else None
    }

# Read the entire file into a single string
with open('pubmed-intelligen-set.txt', 'r') as file:
    file_content = file.read()

# Split the file content into individual documents
documents = file_content.strip().split('\n\nPMID- ')

# Parse each document
parsed_documents = [parse_document('PMID- ' + doc) for doc in documents]

# Create a DataFrame
df = pd.DataFrame(parsed_documents)

# Save to CSV
df.to_csv('parsed_abstracts.csv')


