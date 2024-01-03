
from Bio import Entrez
import pandas as pd
from tqdm import tqdm
import time



def search(query, retstart, retmax):
    Entrez.email = "sandra_friebolin@proton.me"
    handle = Entrez.esearch(db='pubmed', 
                            retstart=retstart, 
                            retmax=retmax, 
                            retmode='xml', 
                            term=query)
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


#query = "intelligence[Title/Abstract] AND (\"2013\"[Date - Publication] : \"2013\"[Date - Publication])"
query = "intelligence[Title/Abstract] AND (\"2013\"[Date - Publication] : \"2023\"[Date - Publication])"

retmax = 1000  # Number of results to fetch in each request

def get_total_count(query):
    Entrez.email = "sandra_friebolin@proton.me"
    handle = Entrez.esearch(db='pubmed', retmax=1, retmode='xml', term=query)
    results = Entrez.read(handle)
    print(results)  # Temporarily added to inspect the raw result
    return int(results['Count'])

total_count = get_total_count(query)
print("Total records to fetch:", total_count)

# Add a delay to avoid hitting the API too hard
delay = 1  # seconds

pmid_list = []
title_list = []
abstract_list =[]
author_list = []
date_list = []
doi_list = []


# Loop to fetch results in batches
for start in range(0, total_count, retmax):
    print(f"Fetching records {start+1} to {start+retmax}")
    studies = search(query, start, retmax)
    studiesIdList = studies['IdList']
    papers = fetch_details(studiesIdList)

    for paper in papers['PubmedArticle']:
        article = paper['MedlineCitation']['Article']

        # Check if Abstract is present
        if article.get("Abstract") is not None:
            abstract_texts = article['Abstract']['AbstractText']
            full_abstract = ' '.join([str(text) for text in abstract_texts])
            
            title_list.append(article['ArticleTitle'])
            pmid_list.append(paper['MedlineCitation']['PMID'])

            # Fetch authors
            if 'AuthorList' in article:
                authors = article['AuthorList']
                author_names = [author.get('ForeName') + " " + author.get('LastName') \
                                if author.get('ForeName') else author.get('LastName') \
                                    for author in authors if 'LastName' in author]
                author_list.append("; ".join(author_names))
            else:
                author_list.append("")

            # Fetch Publication Date
            medline_citation = paper.get('MedlineCitation', {})
            article = medline_citation.get('Article', {})
            pub_date = None

            # Check various fields for publication date
            if 'ArticleDate' in article:
                pub_date = article['ArticleDate']
            elif 'PubDate' in article:
                pub_date = article['PubDate']
            elif 'DateCompleted' in medline_citation:
                pub_date = medline_citation['DateCompleted']
            elif 'DateRevised' in medline_citation:
                pub_date = medline_citation['DateRevised']

            # Format the publication date
            if pub_date:
                date_str = f"{pub_date[0]['Year']}-{pub_date[0].get('Month', '01')}-{pub_date[0].get('Day', '01')}"
            else:
                date_str = ""

            date_list.append(date_str)

            # Fetch DOI
            article_id_list = paper.get('PubmedData', {}).get('ArticleIdList', [])
            doi = next((id_ for id_ in article_id_list if id_.attributes.get('IdType') == 'doi'), None)
            doi_list.append(doi if doi is not None else "")

            # Append Abstract
            abstract_list.append(full_abstract)

            time.sleep(delay)  # Delay between each request


# Create DataFrame
df = pd.DataFrame({
    'PMID': pmid_list,
    'Title': title_list,
    'Abstract': abstract_list,
    'Authors': author_list,
    'Publication Date': date_list,
    'DOI': doi_list
})

print(df.head())

        
# Save the data to a CSV file
df.to_csv('pubmed_data.csv', index=False)
