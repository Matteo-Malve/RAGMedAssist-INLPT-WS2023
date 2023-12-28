from Bio import Entrez
import csv
from tqdm import tqdm


def fetch_abstracts(pmids):
    Entrez.email = "your_email@example.com"  # Always provide your email
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="text")
    abstracts = handle.read()
    handle.close()
    return abstracts

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['PMID', 'Abstract'])
        # Write the data
        for pmid, abstract in data.items():
            writer.writerow([pmid, abstract])

# Fetch PMIDs based on the query
Entrez.email = "sandra_friebolin@proton.me"
query = "intelligence[Title/Abstract] AND (\"2013\"[Date - Publication] : \"2023\"[Date - Publication])"
handle = Entrez.esearch(db="pubmed", term=query, retmax=100000)  # Adjust retmax as needed
record = Entrez.read(handle)
handle.close()
pmid_list = record["IdList"]

# Fetch abstracts for the PMIDs
abstracts_data = {}
for pmid in tqdm(pmid_list, desc="Downloading Abstracts"):
    try:
        abstract = fetch_abstracts(pmid)
        abstracts_data[pmid] = abstract
    except Exception as e:
        print(f"Error fetching abstract for PMID {pmid}: {e}")
        
# Save the data to a CSV file
save_to_csv(abstracts_data, 'abstracts.csv')
