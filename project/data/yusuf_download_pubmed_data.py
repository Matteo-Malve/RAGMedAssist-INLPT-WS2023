# from Bio import Entrez
# import csv
# from tqdm import tqdm
#
#
# def fetch_abstracts(pmids):
#     Entrez.email = "your_email@example.com"  # Always provide your email
#     handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="text")
#     abstracts = handle.read()
#     handle.close()
#     return abstracts
#
#
# def save_to_csv(data, filename):
#     with open(filename, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         # Write the header
#         writer.writerow(['PMID', 'Abstract'])
#         # Write the data
#         for pmid, abstract in data.items():
#             writer.writerow([pmid, abstract])
#
#
# # Fetch PMIDs based on the query
# Entrez.email = "sandra_friebolin@proton.me"
# query = "intelligence[Title/Abstract] AND (\"2013\"[Date - Publication] : \"2023\"[Date - Publication])"
# handle = Entrez.esearch(db="pubmed", term=query, retmax=100000)  # Adjust retmax as needed
# record = Entrez.read(handle)
# handle.close()
# pmid_list = record["IdList"]
#
# # Fetch abstracts for the PMIDs
# abstracts_data = {}
# for pmid in tqdm(pmid_list, desc="Downloading Abstracts"):
#     try:
#         abstract = fetch_abstracts(pmid)
#         abstracts_data[pmid] = abstract
#     except Exception as e:
#         print(f"Error fetching abstract for PMID {pmid}: {e}")
#
# # Save the data to a CSV file
# save_to_csv(abstracts_data, 'abstracts.csv')




from Bio import Entrez
import pandas as pd

def search(query):
    Entrez.email = "sandra_friebolin@proton.me"
    handle = Entrez.esearch(db='pubmed',
    # sort='relevance',
    retmax='10000',
    retmode='xml',
    term=query, rettype="abstract")
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = "sandra_friebolin@proton.me"
    handle = Entrez.efetch(db='pubmed',
    retmode='xml',
    id=ids)
    results = Entrez.read(handle)
    return results


query = "intelligence[Title/Abstract] AND (\"2013\"[Date - Publication] : \"2013\"[Date - Publication])"
studies = search(query)
studiesIdList = studies['IdList']
len(studiesIdList)


from tqdm import tqdm

pmid_list = []
title_list = []
abstract_list =[]
papers = fetch_details(studiesIdList)
for i, paper in enumerate(papers['PubmedArticle']):

    if paper['MedlineCitation']["Article"].get("Abstract") is not None:
        # some abstracts are divided into multiple sections.
        # For example on the page https://pubmed.ncbi.nlm.nih.gov/38108232/ , tha abstract is divided into 4 parts (Objective, Methods, Results, Conclusion)
        # In this case, we only retrieve the first text section under 'AbstractText'. But these approach should be adapted as needed in the future
        abstract_list.append(paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0])
        title_list.append(paper['MedlineCitation']['Article']['ArticleTitle'])
        pmid_list.append(paper['MedlineCitation']['PMID'][:])


df = pd.DataFrame(list(zip(pmid_list, title_list, abstract_list)), columns=['PMID','Title', 'Abstract'])

df.isna().sum()
df.head()

