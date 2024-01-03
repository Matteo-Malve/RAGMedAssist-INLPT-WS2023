# Define the query
query='intelligence[Title/Abstract] AND ("2013"[Date - Publication] : "2023"[Date - Publication])'

# Determine the total number of records
total_records=$(esearch -db pubmed -query "$query" | xtract -element Count)

# Set batch size
batch_size=1000

# Initialize a file to store all results
output_file="all_results.txt"
> "$output_file" # This clears the file if it already exists

# Loop through in batches
for ((i=0; i<total_records; i+=batch_size))

do
    esearch -db pubmed -query "$query" -retstart $i -retmax $batch_size |
    efetch -format xml |
    xtract -pattern PubmedArticle -if Abstract \
    -element MedlineCitation/PMID -tab "\t" \
    -block Article -element ArticleTitle -tab "\t" \
    -block Abstract -sep " " -element AbstractText -tab "\t" \
    -block Journal -block JournalIssue -block PubDate -tab "\t" -element Year,Month,Day -sep "-" -tab "\t" \
    -group AuthorList -sep ", " -element Author/LastName,Author/ForeName -tab "\n" >> "$output_file"
done
