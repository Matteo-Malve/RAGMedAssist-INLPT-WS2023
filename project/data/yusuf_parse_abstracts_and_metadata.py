# import pandas as pd
# import re
#
#
# def parse_document(document):
#     pmid = re.search(r'PMID- (\d+)', document)
#     date = re.search(r'DP  - (\d{4} \w{3} \d{1,2})', document)
#     title = re.search(r'TI  - (.+?)\n', document, re.DOTALL)
#     abstract = re.search(r'AB  - (.*?)\n[A-Z]{2,4}  - ', document, re.DOTALL)
#     authors = re.findall(r'FAU - (.+)', document)
#     doi = re.search(r'LID - (.+?) \[doi\]', document)
#
#     return {
#         'PMID': pmid.group(1) if pmid else None,
#         'Date': date.group(1) if date else None,
#         'Title': title.group(1).replace('\n      ', ' ') if title else None,
#         'Abstract': abstract.group(1).replace('\n      ', ' ') if abstract else None,
#         'Authors': '; '.join(authors) if authors else None,
#         'DOI': doi.group(1) if doi else None
#     }
#
# # Read the entire file into a single string
# with open('pubmed-intelligen-set.txt', 'r') as file:
#     file_content = file.read()
#
# # Split the file content into individual documents
# documents = file_content.strip().split('\n\nPMID- ')
#
# # Parse each document
# parsed_documents = [parse_document('PMID- ' + doc) for doc in documents]
#
# # Create a DataFrame
# df = pd.DataFrame(parsed_documents)
#
# # Save to CSV
# df.to_csv('parsed_abstracts.csv')


import pandas as pd

"""
Note: Some records contain the same attribute multiple times such as:
    IS - 2045-2322 (Electronic)
    IS - 2045-2322 (Linking)
In these case, the implementation adds only the last one to the returned data. 
These can be easily changed according to the needs 
"""
def parse_pubmed_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the data into records
    records = data.strip().split('\n\n')

    # Parse each record into a dictionary
    parsed_records = []
    for record in records:
        record_dict = {}
        for line in record.split('\n'):
            if len(line) - len(line.lstrip()) >= 6:
                record_dict[last_added_key.strip()]+=(" "+line.strip())
                continue
            if line:  # Skip empty lines
                key, value = line.split('-', 1)
                record_dict[key.strip()] = value.strip()
                last_added_key = key

        parsed_records.append(record_dict)

    return parsed_records


parsed_data = parse_pubmed_data('pubmed-intelligen-set.txt')
dff = pd.DataFrame(parsed_data)
dff.head()

dff.isna().sum()
dff = dff[~dff["AB"].isna()]
dff.isna().sum()
small_dff = dff[["PMID", "TI","AB"]]

